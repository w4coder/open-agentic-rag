# agentic-rag — RAG agentique (LangGraph · LangChain · LangSmith)

[English](README.md) · Partie retrieval : [open-rag](https://github.com/w4coder/open-rag)

Projet de référence **pédagogique**, suite de [`open-rag`](https://github.com/w4coder/open-rag). Il montre comment passer d'un pipeline RAG linéaire à un **agent** qui route, planifie, vérifie et agit — avec les garde-fous d'un système de production. Commencez par [docs/patterns.md](docs/patterns.md) (catalogue + schéma du graphe), puis lisez `graph.py`.

## Démarrage (offline, sans clé API)

```bash
# 1. le projet 1 fournit la couche retrieval (hybride, chunking, index) comme bibliothèque
pip install -r requirements-dev.lock && pip install --no-deps -e .   # versions verrouillées (inclut open-rag v0.3.0 depuis GitHub)
cp .env.example .env
pytest -q                                   # 41 tests : topologie, sécurité, budgets, durabilité, résilience LLM
python -m evals.run_eval                    # agent vs RAG single-shot, réponse + trajectoire + coût
python -m agentic_rag.cli "What is the Business plan rate limit and what is the Enterprise availability SLA?"
python -m agentic_rag.cli "What was the total revenue in Q1 by region?"   # demande une approbation humaine
uvicorn agentic_rag.api:app --reload        # /v1/ask puis /v1/resume
```

Mode `heuristic` (défaut) : les décisions (routeur, planner, grader, vérificateur) sont des règles déterministes → la **topologie** du graphe est testable sans LLM. Mode réel : `AGENT_LLM_PROVIDER=openai` + `OPENAI_API_KEY` → les mêmes décisions sont prises par un LLM via sorties structurées (`LLMBrain`). Le mode LLM n'est **pas** couvert par les tests automatiques (pas de clé dans ce dépôt) : évaluez-le avec `python -m evals.run_eval --langsmith`.

## Ce que fait l'agent

| Question | Chemin |
|---|---|
| « Hello » | route → réponse directe (aucune recherche) |
| « Quelle est l'erreur NB-4021 ? » | route docs → plan (1) → CRAG → synthèse → vérif |
| « Limite Business **et** SLA Enterprise ? » | plan (2 sous-questions) → **2 recherches en parallèle** → synthèse |
| Question sans réponse | CRAG : rewrite → web fallback → replan (borné) → refus honnête |
| « Revenu Q1 par région ? » | text-to-SQL → **validation** → ⏸ **approbation humaine** → exécution lecture seule |
| « Statut des incidents ? » | route web → recherche web (stub offline) |
| « Remember that I prefer French » | mémoire long terme (store) |

## Carte du code

| Fichier | Rôle |
|---|---|
| `graph.py` | **graphe principal** : routeur, plan, `Send`, synthèse, vérification, replan, HITL, mémoire |
| `crag.py` | sous-graphe Corrective RAG (retrieve → grade → rewrite/web) |
| `brain.py` | toutes les décisions derrière une interface (`HeuristicBrain` / `LLMBrain`) |
| `tools.py` | outils : docs (retrieval de open-rag), SQL sécurisé, web ; schéma d'évidence uniforme |
| `service.py` | exécution, reprise après interruption, config LangSmith (tags, thread, budget) |
| `api.py`, `cli.py` | interfaces |
| `evals/` | dataset, évaluation réponse/trajectoire/coût, baseline single-shot, LangSmith |
| `docs/` | patterns, évaluation, ADR |

## LangSmith
`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` : chaque run produit un arbre de traces (`agentic_rag` → nœuds → sous-graphe CRAG → appels LLM), tagué par `brain:*`, groupé par `thread_id`. Les expériences (`--langsmith`) comparent réponse, trajectoire et nombre d'étapes.

## Production
Identité par clé API (`AGENT_API_KEYS`), threads privés, rôle `approver` (+ *four eyes*), état durable (`AGENT_CHECKPOINT_URL` sqlite/postgres), index partagé avec open-rag (`RAG_QDRANT_URL`), budgets (steps, tokens/run, tokens/jour, rate limit), SQL validé par parseur + authorizer de la base, mémoire filtrée. Voir [docs/tutorial.md](docs/tutorial.md) et [docs/production-review.md](docs/production-review.md).

## Limites assumées
- Le `LLMBrain` est **évalué** (`evals --langsmith`, CI sur `main`) et sa résilience testée avec des stubs, mais aucun chiffre de qualité avec un vrai LLM n'est publié ici (pas de clé dans ce dépôt).
- Le backend Postgres n'est pas couvert par les tests automatiques (SQLite l'est, avec le même contrat LangGraph).
- La base SQL de démo est un SQLite en mémoire : sur un vrai entrepôt, recréez la même défense avec un rôle en lecture seule + GRANT par table + `statement_timeout`.
- Le corpus de démonstration est minuscule : les chiffres de `evals/` valident le comportement, pas la performance d'un LLM réel.
