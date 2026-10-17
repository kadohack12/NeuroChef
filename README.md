# NeuroChef

NeuroChef est un moteur de décodage automatique orienté **flux d’octets**. Il recherche des chaînes de transformations imbriquées, détecte les formats binaires courants et classe les résultats texte ou fichiers.

## Démarrage

```bash
uv sync
uv run pytest
uv run neurochef '...' --max-depth 5
```

Architecture :

- `src/neurochef/transforms/` : transformations indépendantes `bytes -> Optional[bytes]`.
- `src/neurochef/detectors/` : entropie de Shannon, signatures PE/ELF/ZIP/GZIP et charset.
- `src/neurochef/evaluators/` : score heuristique texte français/anglais et headers binaires.
- `src/neurochef/core/` : recherche BFS bornée, déduplication et traçage des étapes.
- `src/neurochef/ai/` : extraction de features, ranking heuristique et connecteur LLM optionnel.
- `tests/` : tests unitaires du pipeline et des composants.
- `benchmarks/` : points d’entrée pour futurs benchmarks de profondeur et de fan-out.

Le MVP privilégie la robustesse : un transform invalide renvoie `None`, la recherche est bornée par profondeur et nombre de nœuds, et les résultats gardent la chaîne de transformations appliquée.

## Ranker ONNX local

Le modèle optionnel `models/neuro_ranker_optimized.onnx` est chargé par `ONNXRanker`. Sa signature inspectée est `float_input: float32[None, 14]`, avec les sorties `output_label` et `output_probability` (séquence de dictionnaires classe→probabilité). L’ordre de features est versionné dans `neurochef.ai.features.FEATURE_ORDER` et contrôlé à l’initialisation. Si le modèle est absent, invalide ou si l’inférence échoue, le ranker revient automatiquement à `HeuristicRanker`.

## Ranking et recherche guidée

Le moteur utilise maintenant une file de priorité A* best-first et injecte un `HeuristicRanker` par défaut. Les sorties exposent `nodes_explored` pour mesurer l’efficacité du guidage. `LLMRanker` peut utiliser Ollama local ou une API compatible OpenAI/Anthropic, mais revient automatiquement à l’heuristique si le service est indisponible ou incertain.
