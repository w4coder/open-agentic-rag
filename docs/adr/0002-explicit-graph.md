# ADR 0002 — Workflow à graphe explicite plutôt qu'agent ReAct libre

**Décision** : topologie fixe et lisible (routeur → plan → recherche parallèle → synthèse → vérification → replan borné) implémentée en LangGraph, avec des *boucles bornées* et des budgets durs (`max_loops`, `max_retrieval_attempts`, `recursion_limit`).

**Pourquoi** : le RAG agentique de production demande prévisibilité, auditabilité (traces LangSmith lisibles) et coûts bornés. Un agent entièrement libre décide seul du nombre d'appels d'outils : puissant, mais difficile à borner et à évaluer.

**Où le LLM garde la main** : le *contenu* de chaque décision (quelle route, quelles sous-questions, ce document est-il pertinent ?), pas l'existence du flux.

**Choix de LangGraph** : `Send` (map-reduce parallèle), sous-graphes, `interrupt` (HITL), checkpointer (reprise), store (mémoire long terme), streaming.

**HITL** : `approve_sql` est un nœud isolé sans effet de bord, car à la reprise LangGraph ré-exécute le nœud interrompu depuis le début.

**Production** : remplacer `InMemorySaver`/`InMemoryStore` par Postgres ; exposer reprise via `/v1/resume`.
