"""Metrics shared by experiments and regression tests."""

from __future__ import annotations

from collections.abc import Sequence

from prma.encoding import cycle_edges


def directed_edge_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Jaccard overlap between the directed edge sets of two cycles."""

    left_edges = set(cycle_edges(left))
    right_edges = set(cycle_edges(right))
    union = left_edges | right_edges
    return len(left_edges & right_edges) / len(union) if union else 1.0


def mean_pairwise_edge_overlap(cycles: Sequence[tuple[int, ...]]) -> float:
    if len(cycles) < 2:
        return 0.0
    values = [
        directed_edge_jaccard(cycles[left], cycles[right])
        for left in range(len(cycles))
        for right in range(left + 1, len(cycles))
    ]
    return sum(values) / len(values)
