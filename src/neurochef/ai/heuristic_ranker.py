"""Fast deterministic baseline ranker."""

from __future__ import annotations

from neurochef.transforms import Transform

from .base import BaseRanker
from .features import FeatureExtractor


class HeuristicRanker(BaseRanker):
    """Order operations using format conformity and byte-level evidence."""

    def __init__(self, extractor: FeatureExtractor | None = None) -> None:
        self.extractor = extractor or FeatureExtractor()

    def rank(self, data: bytes, transforms: list[Transform]) -> list[tuple[Transform, float]]:
        features = self.extractor.extract(data)
        return self.normalize(
            (transform, self._score(transform.name.lower(), features))
            for transform in transforms
        )

    @staticmethod
    def _score(name: str, features: dict[str, float]) -> float:
        base64_ratio = features["base64_ratio"]
        hex_ratio = features["hex_ratio"]
        printable = features["printable_ratio"]
        entropy = features["shannon_entropy"]
        if "base64" in name:
            return 1.0 if base64_ratio > 0.95 and features["is_multiple_of_4"] else base64_ratio
        if "hex" in name:
            return 1.0 if hex_ratio > 0.95 and features["is_multiple_of_2"] else hex_ratio
        if "gzip" in name:
            return 1.0 if features["is_gzip"] else 0.05
        if "zlib" in name or "decompress" in name:
            return 1.0 if features["is_zlib"] else max(0.05, entropy / 8.0 - 0.20)
        if "base85" in name or "ascii85" in name:
            return features["base85_ratio"]
        if "url" in name:
            return min(0.90, 0.20 + printable * 0.30 + (1.0 - base64_ratio) * 0.20)
        if "xor" in name:
            return max(0.10, (1.0 - printable) * 0.65 + entropy / 8.0 * 0.35)
        if "rot" in name:
            return 0.20 + printable * 0.20
        if "reverse" in name:
            return 0.15 + (1.0 - printable) * 0.10
        return 0.10
