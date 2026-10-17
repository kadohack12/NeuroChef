"""Bounded graph search over byte transformations."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from neurochef.evaluators import Evaluation, score_bytes
from neurochef.transforms import DEFAULT_TRANSFORMS, Transform


@dataclass(frozen=True)
class SearchStep:
    transform: str
    data: bytes


@dataclass(frozen=True)
class SearchResult:
    data: bytes
    steps: tuple[SearchStep, ...]
    evaluation: Evaluation


class SearchEngine:
    """Breadth-first decoder with depth and node limits.

    Transforms are deliberately untrusted: invalid inputs simply yield no candidate.
    The best-scoring state is returned, which makes the MVP useful even when no
    candidate reaches a perfect terminal score.
    """

    def __init__(
        self,
        transforms: Sequence[Transform] = DEFAULT_TRANSFORMS,
        max_depth: int = 5,
        max_nodes: int = 10_000,
    ) -> None:
        self.transforms = tuple(transforms)
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def search(self, data: bytes) -> SearchResult:
        queue: deque[tuple[bytes, tuple[SearchStep, ...]]] = deque([(data, ())])
        visited: set[bytes] = {data}
        best = SearchResult(data, (), score_bytes(data))
        nodes = 0

        while queue and nodes < self.max_nodes:
            current, path = queue.popleft()
            current_eval = score_bytes(current)
            candidate = SearchResult(current, path, current_eval)
            if self._rank(candidate) > self._rank(best):
                best = candidate
            if len(path) >= self.max_depth:
                continue
            for transform in self.transforms:
                if not self._applicable(transform, current):
                    continue
                for decoded in transform.candidates(current):
                    nodes += 1
                    if nodes >= self.max_nodes:
                        break
                    if not decoded or decoded == current or decoded in visited:
                        continue
                    visited.add(decoded)
                    next_path = path + (SearchStep(transform.name, decoded),)
                    decoded_eval = score_bytes(decoded)
                    if decoded_eval.score >= 0.75:
                        return SearchResult(decoded, next_path, decoded_eval)
                    queue.append((decoded, next_path))
                if nodes >= self.max_nodes:
                    break
        return best

    @staticmethod
    def _applicable(transform: Transform, data: bytes) -> bool:
        """Apply obvious format gates before expensive fan-out transforms."""
        if len(data) >= 2 and len(data) % 2 == 0:
            try:
                is_hex = all(byte in b"0123456789abcdefABCDEF" for byte in data)
            except TypeError:
                is_hex = False
            if is_hex:
                return transform.name == "from_hex"
        return True

    @staticmethod
    def _rank(result: SearchResult) -> tuple[float, int, int]:
        # A small depth tie-break rewards actually decoding a layer without
        # allowing depth to overpower the evaluator.
        return (result.evaluation.score, len(result.steps), -len(result.data))

    def solve(self, data: bytes) -> SearchResult:
        """Alias intended for CLI callers and future strategy implementations."""
        return self.search(data)
