# ADR 0001 — Interface « Brain » : décisions séparées de la topologie

**Contexte** : un agent mélange *flux de contrôle* (code déterministe) et *décisions* (LLM, non déterministes). Si tout est mélangé dans des prompts monolithiques, on ne peut ni tester ni évaluer précisément.

**Décision** : chaque décision (route, plan, grade, rewrite, text-to-SQL, synthèse, vérification) est une méthode d'une interface `Brain`, à sortie **structurée** (Pydantic). Deux implémentations : `LLMBrain` (production) et `HeuristicBrain` (tests déterministes).

**Conséquences** : (+) la topologie est testée en CI sans clé ni flakiness ; (+) chaque décision a son dataset d'évaluation ; (+) on peut mixer des modèles (petit modèle pour grade/route, grand pour synthèse). (−) le `HeuristicBrain` doit rester simple, sinon on évalue nos règles et non le système ; (−) plus d'appels LLM que l'agent « un seul prompt ReAct » (compromis assumé : contrôle > concision).

**Alternatives** : agent ReAct/tool-calling générique (flexible, peu prévisible, plus dur à borner) ; à envisager pour des tâches ouvertes, pas pour un RAG où le chemin est prévisible.
