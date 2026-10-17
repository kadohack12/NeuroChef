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
- `tests/` : tests unitaires du pipeline et des composants.
- `benchmarks/` : points d’entrée pour futurs benchmarks de profondeur et de fan-out.

Le MVP privilégie la robustesse : un transform invalide renvoie `None`, la recherche est bornée par profondeur et nombre de nœuds, et les résultats gardent la chaîne de transformations appliquée.
