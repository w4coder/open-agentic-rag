# Tutoriel — lire `agentic-rag` fichier par fichier

À lire **à côté du code**. Prérequis : le tutoriel de `open-rag` (la couche retrieval est réutilisée telle quelle).

## 0. Vue d'ensemble

```
API (clé → Principal user/tenant/roles) ─► AgentService (propriété du thread, budgets) ─► graphe :

START ─► load_memory ─► route ─┬─ direct ──► direct_answer ─► END
                               ├─ memory ──► remember ─► END
                               ├─ sql ─────► propose_sql ─► approve_sql ⏸ ─► run_sql ─┐
                               └─ docs/web ► plan ═Send×N═► research (CRAG) ═══════════┤
                                              ▲                                         ▼
                                              └── replan (borné) ◄── verify ◄── synthesize
                                                                       └─ ok/budget ─► finalize ─► END
```

Principe (ADR 0001/0002) : **le code fixe le flux, le « cerveau » prend les décisions**. Le graphe est déterministe et testable ; chaque décision est un appel isolé, évaluable, avec une valeur par défaut sûre en cas de panne.

**Ordre de lecture** : `config` → `tools` → `brain` → `budget` → `crag` → `graph` → `persistence` → `service` → `api`/`cli` → `evals` → `tests`.

---

## 1. `config.py`
`AgentSettings` (préfixe `AGENT_`), par groupe :
- **Cerveau** : `llm_provider` (`heuristic`/`openai`), `llm_model`, `llm_fallback_model`, `llm_timeout_s`.
- **Budgets** : `max_loops`, `max_retrieval_attempts`, `max_steps` (→ `recursion_limit`), `max_tokens_per_run`, `tenant_daily_tokens`, `rate_limit_per_min`.
- **HITL** : `require_approval`, `four_eyes`.
- **Persistance** : `checkpoint_url` (`memory` / `sqlite:///…` / `postgresql://…`), `memory_max_items`, `memory_max_chars`.
- **Outils** : `corpus_path`, `ingest_on_start` (dev), `tenant` (tenant par défaut en CLI), `sql_timeout_s`, `web_provider` (`stub`/`tavily`), `tavily_api_key`.
- **API** : `api_keys` = `"key:user:tenant:role1|role2,…"`, `redis_url`, `log_json`.

---

## 2. `tools.py` — actions de l'agent
Format unique des évidences : `{"id", "text", "source", "kind": doc|sql|web, "sub_question"}`.

| Élément | Rôle |
|---|---|
| `DocsTool(settings, rag_settings=None)` | Avec `RAG_QDRANT_URL` : se connecte à **l'index partagé de open-rag** (mêmes embeddings, aucune ingestion). Sinon : Qdrant en mémoire + ingestion du corpus (dev). Les embeddings viennent de l'environnement `RAG_*`. |
| `DocsTool.search(query, sub_question, tenant)` | Top-k hybride **dans le tenant de l'appelant**. |
| `validate_sql(sql)` | Filtre regex rapide, puis **sqlglot** : une seule instruction, de type requête, aucun nœud d'écriture, **toutes** les tables (jointures, sous-requêtes, `main.x`, CTE) dans `ALLOWED_TABLES`, fonctions dans `ALLOWED_FUNCTIONS`. Renvoie le SQL **normalisé**, celui qui sera exécuté. |
| `SqlTool._authorize` | **Authorizer SQLite** : la base n'autorise que `SELECT`, la lecture des tables permises et les fonctions permises ; tout le reste est refusé (`sqlite_master`, CTE récursif…). C'est l'équivalent d'un rôle en lecture seule avec GRANT par table. |
| `SqlTool.run(sql, sub_question)` | Valide → lock (connexion partagée) → *progress handler* (timeout) → `LIMIT 50` → évidence `kind=sql`. |
| `WebTool` | Stub hors ligne déterministe. |
| `TavilyWebTool(api_key, timeout_s, transport)` | Vraie recherche web ; une erreur donne zéro évidence, jamais un crash. `transport` permet les tests (httpx `MockTransport`). |
| `build_tools(settings)` | Assemble les trois outils. |

---

## 3. `brain.py` — les décisions
`Brain` (Protocol) : `route`, `plan`, `grade_many`, `rewrite`, `text_to_sql`, `synthesize`, `verify`. Sorties : `Synthesis(answer, cited_ids, missing)`, `Verdict(grounded, complete, missing)`.

| Méthode | `HeuristicBrain` | `LLMBrain` | Défaut si panne |
|---|---|---|---|
| `route` | Regex | `_Route` | `docs` |
| `plan` | Découpe sur and/et/; | `_Plan` (≤ 4) | `[question]` |
| `grade_many` | Recouvrement de mots-clés | **batch concurrent** (1 latence au lieu de N) | garder le passage (la vérification contrôle) |
| `rewrite` | Mots-clés | `_Rewrite` | sous-question |
| `text_to_sql` | Gabarits | `_SQL` | `""` → bloqué par le validateur |
| `synthesize` | Extractif (open-rag) | `Synthesis` ; préférences dans `<preferences>` comme **données** | refus |
| `verify` | Citations ∈ évidences | Check déterministe puis LLM | **échec fermé** (ungrounded) |

`LLMBrain(settings, llm=None, fallback=None)` : `_chain` applique `with_structured_output` puis `with_fallbacks`. `_ask(schema, system, human, default)` renvoie `default` en cas d'erreur, sauf pour `BudgetExceeded`, qui est propagé. `llm`/`fallback` sont injectables pour les tests.

## 4. `budget.py`
- `BudgetExceeded`.
- `TokenBudget(max_tokens)` : callback LangChain (`raise_error=True`) attaché au run. Il voit tous les appels LLM (sous-graphes, branches `Send`), compte `total_tokens` même sans `model_name`, et lève l'exception au dépassement. Il hérite de `UsageMetadataCallbackHandler` (usage par modèle → métriques).

## 5. `crag.py` — sous-graphe Corrective RAG
État : `sub_question`, `tenant`, `mode`, `query`, `attempt`, `candidates`, `relevant`, `path`.
`entry` (web direct ?) → `retrieve` (docs du tenant) → `grade` (`grade_many`) → `after_grade` : pertinent → END, sinon `rewrite` (borné par `max_retrieval_attempts`), sinon `web_fallback`. `path` sert à l'éval de trajectoire.

## 6. `graph.py` — graphe principal
- `extend_or_reset` : reducer de liste. Les branches parallèles ajoutent, une liste qui commence par `RESET` remplace.
- `RUN_RESET` : état de run vierge envoyé à chaque `ask`. Il corrige le bug de réutilisation de `thread_id` (statut, évidences, `missing` hérités).
- `AgentState` : + `tenant`.

| Nœud | Rôle |
|---|---|
| `load_memory` | Lit `("memory", tenant, user)`, au plus `memory_max_items` éléments. |
| `remember` | Écriture **non fiable** : `check_input` (injection, longueur), `redact_pii`, éviction des plus anciens au-delà du plafond. |
| `route`, `direct_answer`, `plan` | Routage, small talk, décomposition. |
| `fan_out` | Un `Send("research", {sub_question, mode, tenant})` par sous-question. |
| `research` | Sous-graphe CRAG. |
| `propose_sql` → `approve_sql` → `run_sql` | Text-to-SQL → `interrupt` (HITL, sans effet de bord) → exécution sécurisée. |
| `synthesize`, `verify`, `replan`, `finalize` | Synthèse, réflexion bornée, statut final (ok / incomplete / ungrounded / rejected / blocked). |

## 7. `persistence.py`
`build_persistence(url)` → `(checkpointer, store)` :
- `memory` : dev.
- `sqlite:///…` : un seul nœud, durable (WAL).
- `postgresql://…` : production ; pool psycopg (`autocommit`, `dict_row`), `setup()` crée les tables. Une pause HITL sur un réplica peut être reprise sur un autre.

## 8. `service.py`
| Élément | Rôle |
|---|---|
| `Principal(user_id, tenant, roles)` | Identité, issue de la clé API. |
| `Forbidden`, `Conflict`, `UpstreamError` | → 403 / 409 / 503. |
| `local_principal(user)` | CLI/tests : l'opérateur au terminal est aussi approbateur. |
| `_claim_thread` | Registre global `("threads",)` : thread → `{tenant, user_id}`. Un autre utilisateur ou tenant → `Forbidden`. |
| `_config` | `thread_id = "tenant:thread"`, `recursion_limit`, callback `TokenBudget`, tags/métadonnées LangSmith (+ request id). |
| `_pending` | Le thread a-t-il une interruption en attente ? |
| `_run` | Budget journalier → `graph.invoke` → `budget_exceeded` / `step_budget_exceeded` / `UpstreamError` → comptage des tokens → métriques `agent_runs_total`, `agent_run_seconds`, `agent_run_steps`. |
| `ask(question, user_id, thread_id, principal)` | Revendique le thread, refuse s'il attend une approbation (`Conflict`), lance avec `RUN_RESET`. |
| `resume(thread_id, approved, user_id, principal)` | Même tenant, rôle `approver`, *four eyes* optionnel, une interruption doit être en attente ; **journal d'audit** (`agentic_rag.audit`) avec le SQL, le demandeur et l'approbateur. |

## 9. `api.py` et `cli.py`
- `parse_api_keys` : `key:user:tenant:roles` → `Principal`. **L'identité ne vient jamais du body.**
- Dépendance `principal` : 401 si clé inconnue, 429 au-delà du rate limit (`open_rag.limits`, partagé via Redis).
- `POST /v1/ask` (`thread_id` validé par regex), `POST /v1/resume` ; `/healthz`, `/readyz` (index docs + store), `/metrics` ; middleware `X-Request-ID`.
- `cli.py` : boucle `ask` → approbation `y/N` au terminal → `resume`.

## 10. Déploiement
- `pyproject.toml` : `open-rag @ https://github.com/w4coder/open-rag/archive/refs/tags/v0.3.0.tar.gz`, une URL directe épinglée sur un tag, pour qu'un paquet homonyme sur PyPI ne soit jamais résolu (*dependency confusion*). Extras `sqlite`, `postgres`.
- `requirements*.lock` : versions exactes.
- `deploy/Dockerfile` : contexte = dossier parent (monorepo), multi-stage, non-root.
- `deploy/docker-compose.yml` : 2 réplicas d'agent + Postgres + Qdrant + Redis.
- `.github/workflows/ci.yml` : lock → ruff → pytest → éval avec gate déterministe → `pip-audit` → build ; sur `main`, éval du `LLMBrain` avec gate.

## 11. Évaluation et tests
| Fichier | Rôle |
|---|---|
| `evals/run_eval.py` | Réponse / trajectoire / coût vs baseline single-shot ; `--min-correct`, `--min-trajectory` = gate ; provider lu depuis l'environnement. |
| `tests/test_agent.py` | Topologie : routage, parallélisme, CRAG borné, HITL, mémoire, SQL. |
| `tests/test_production.py` | Thread réutilisé propre, `Conflict`, threads privés + rôle approbateur, *four eyes*, isolation des docs par tenant, contournements SQL (7 cas), authorizer sans validateur, timeout SQL, mémoire (injection, PII, plafond, tenant), `LLMBrain` (défauts sûrs, fallback, refus honnête de bout en bout), budget tokens (callback réel), thread réutilisable après un dépassement, budgets de steps et journalier, **reprise après redémarrage (SQLite)**, Tavily (succès/panne), API (identité par clé, 403/409, request id, readyz, metrics). |

## 12. Exercices
1. Avec `AGENT_CHECKPOINT_URL=sqlite:///agent.db` : lancez une question SQL, tuez le process, puis reprenez avec `cli`/`/v1/resume`.
2. Écrivez une requête SQL qui passe `validate_sql` mais que l'authorizer refuse (indice : il y en a de moins en moins).
3. Fixez `AGENT_MAX_TOKENS_PER_RUN=500` en mode `openai` et observez `budget_exceeded`.
