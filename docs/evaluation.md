# Évaluation d'un agent

Un agent peut donner la bonne réponse par un mauvais chemin (coûteux, fragile) : on évalue donc **trois niveaux**.

| Niveau | Question | Mesure (`evals/run_eval.py`) |
|---|---|---|
| Réponse finale | correcte ? refus correct si non répondable ? | `correct` |
| **Trajectoire** | bons nœuds (route, plan, SQL…), parallélisme attendu, rien d'interdit, budget respecté ? | `trajectory` |
| Coût | combien d'étapes (≈ appels LLM, latence) ? | `avg_steps` |

Chaque exemple du dataset déclare `expect_nodes`, `forbid_nodes`, `min_research` (nombre de recherches parallèles attendues). Le service expose une `trace` (liste des nœuds et chemins CRAG), qui sert aussi dans LangSmith (évaluateur `trajectory`, métrique `n_steps`).

## Résultats (mode heuristic offline, 10 questions)

| | correct | trajectoire | étapes moy. |
|---|---|---|---|
| **Agent** | 1.00 | 1.00 | 8.1 |
| RAG single-shot (baseline) | 0.40 | n/a | 3 |

| Type | Agent | Baseline |
|---|---|---|
| simple | 1.0 | 1.0 |
| multi-hop | 1.0 | 0.0 |
| sql / web | 1.0 / 1.0 | 0.0 / 0.0 |
| sans réponse | 1.0 | 1.0 |
| chit-chat | 1.0 | 0.0 |

### Comment lire ces chiffres (honnêtement)
- Le « cerveau » heuristique et le dataset de 10 questions sont **conçus ensemble** : ce score valide la **topologie** (le routeur route, les boucles se terminent, le HITL interrompt…), pas la qualité d'un LLM. Un score de 1.0 en conditions réelles serait suspect.
- La baseline n'a ni SQL ni web : elle échoue naturellement sur ces catégories. Le vrai enseignement est le **coût** : l'agent utilise ~2,7× plus d'étapes. Sur les questions simples, les deux sont équivalentes → *ne payez l'agentique que là où elle apporte quelque chose*.
- Pour une conclusion fiable : `AGENT_LLM_PROVIDER=openai python -m evals.run_eval --langsmith`, avec ≥ 50 questions réelles, plusieurs exécutions (les agents sont non déterministes) et comparaison dans LangSmith.

## Tests de comportement (pytest) : ce qui est garanti
Boucle bornée (`replan` = `max_loops`), CRAG (rewrite puis fallback web), SQL non sûr bloqué (5 attaques paramétrées), HITL (interruption, approbation, rejet), mémoire inter-threads, chit-chat sans retrieval.

## Évaluations par étape (à ajouter avec un vrai LLM)
Chaque décision de `brain.py` est isolable : dataset de routage (précision), de planification (qualité des sous-questions), de grading (précision/rappel de la pertinence), de vérification (détection d'hallucinations injectées). Un agent qui échoue → localiser **quelle décision** est fausse avant de toucher aux prompts.
