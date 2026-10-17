"""Bounded A* / best-first graph search over byte transformations."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count

from neurochef.ai import BaseRanker, HeuristicRanker
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
    nodes_explored: int = 0


class SearchEngine:
    """Priority-guided decoder with depth, node and state-duplication limits."""

    def __init__(
        self,
        transforms: Sequence[Transform] = DEFAULT_TRANSFORMS,
        max_depth: int = 5,
        max_nodes: int = 10_000,
        ranker: BaseRanker | None = None,
    ) -> None:
        self.transforms = tuple(transforms)
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.ranker = ranker or HeuristicRanker()

    def search(self, data: bytes) -> SearchResult:
        sequence = count()
        # (estimated total cost, insertion order, data, path, path cost)
        queue: list[tuple[float, int, bytes, tuple[SearchStep, ...], float]] = [
            (0.0, next(sequence), data, (), 0.0)
        ]
        visited: set[bytes] = {data}
        best = SearchResult(data, (), score_bytes(data))
        nodes_explored = 0

        while queue and nodes_explored < self.max_nodes:
            _priority, _order, current, path, path_cost = heapq.heappop(queue)
            nodes_explored += 1
            current_eval = score_bytes(current)
            candidate = SearchResult(current, path, current_eval, nodes_explored)
            if self._rank(candidate) > self._rank(best):
                best = candidate
            if current_eval.score >= 0.75 and path:
                return candidate
            if len(path) >= self.max_depth:
                continue

            applicable = [
                transform for transform in self.transforms if self._applicable(transform, current)
            ]
            ranked = self.ranker.rank(current, applicable)
            for transform, transform_score in ranked:
                # Convert a high probability into a low A* edge cost.
                edge_cost = 1.0 - transform_score
                for decoded in transform.candidates(current):
                    if not decoded or decoded == current or decoded in visited:
                        continue
                    visited.add(decoded)
                    next_path = path + (SearchStep(transform.name, decoded),)
                    next_cost = path_cost + edge_cost
                    decoded_eval = score_bytes(decoded)
                    if decoded_eval.score >= 0.75:
                        return SearchResult(decoded, next_path, decoded_eval, nodes_explored)
                    # Tiny depth penalty prevents endless low-cost expansion ties.
                    priority = next_cost + len(next_path) * 0.01
                    heapq.heappush(
                        queue,
                        (priority, next(sequence), decoded, next_path, next_cost),
                    )
        return SearchResult(best.data, best.steps, best.evaluation, nodes_explored)

    @staticmethod
    def _applicable(transform: Transform, data: bytes) -> bool:
        """Apply obvious format gates before expensive fan-out transforms."""
        if len(data) >= 2 and len(data) % 2 == 0:
            is_hex = all(byte in b"0123456789abcdefABCDEF" for byte in data)
            if is_hex:
                return transform.name == "from_hex"
        return True

    @staticmethod
    def _rank(result: SearchResult) -> tuple[float, int, int]:
        return (result.evaluation.score, len(result.steps), -len(result.data))

    def solve(self, data: bytes) -> SearchResult:
        """Solve an input and retain exploration metrics for diagnostics."""
        return self.search(data)
