"""Ranking interfaces for search guidance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from neurochef.transforms import Transform


class BaseRanker(ABC):
    """Rank candidate operations from most to least promising."""

    @abstractmethod
    def rank(self, data: bytes, transforms: list[Transform]) -> list[tuple[Transform, float]]:
        """Return candidates ordered by descending priority score."""
        raise NotImplementedError

    def rank_transforms(
        self, data: bytes, candidates: list[Transform]
    ) -> list[tuple[Transform, float]]:
        """Backward-compatible name used by early integrations."""
        return self.rank(data, candidates)

    @staticmethod
    def normalize(
        ranked: Sequence[tuple[Transform, float]],
    ) -> list[tuple[Transform, float]]:
        return sorted(
            ((transform, max(0.0, min(1.0, float(score)))) for transform, score in ranked),
            key=lambda item: (-item[1], item[0].name),
        )
