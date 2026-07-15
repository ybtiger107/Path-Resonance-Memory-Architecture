"""Deterministic mapping from memory identifiers to directed cycles."""

from __future__ import annotations

import hashlib


def _node_score(seed: int, memory_id: int, node: int) -> bytes:
    payload = f"prma-v0:{seed}:{memory_id}:{node}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def encode_cycle(
    memory_id: int, node_count: int, path_length: int, seed: int = 0
) -> tuple[int, ...]:
    """Return a stable simple cycle for ``memory_id``.

    Nodes are ordered by a cryptographic digest rather than a sampled sinusoid.
    The mapping therefore has no temporal sampling alias. Different identifiers
    may still share nodes or edges; that interference is an intended benchmark
    variable, not an implicit uniqueness guarantee.
    """

    if memory_id < 0:
        raise ValueError("memory_id must be non-negative")
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if not 2 <= path_length <= node_count:
        raise ValueError("path_length must be between 2 and node_count")

    ranked = sorted(range(node_count), key=lambda node: _node_score(seed, memory_id, node))
    return tuple(ranked[:path_length])


def cycle_edges(cycle: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return directed edges, including the closing edge, of a cycle."""

    if len(cycle) < 2:
        raise ValueError("a cycle needs at least two nodes")
    return tuple((cycle[index], cycle[(index + 1) % len(cycle)]) for index in range(len(cycle)))
