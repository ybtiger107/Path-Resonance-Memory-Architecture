"""Spatial layouts and propagation delays for dynamical PRMA models."""

from __future__ import annotations

import numpy as np


def positions_1d(node_count: int) -> np.ndarray:
    if node_count < 1:
        raise ValueError("node_count must be positive")
    return np.arange(node_count, dtype=np.float64).reshape(-1, 1)


def positions_2d(node_count: int) -> np.ndarray:
    """Place nodes around expanding square shells in legacy order."""

    if node_count < 1:
        raise ValueError("node_count must be positive")
    coordinates = [(1, 1)]
    shell = 1
    while len(coordinates) < node_count:
        for offset in range(1, 2 * shell + 2):
            if len(coordinates) >= node_count:
                break
            if offset <= shell + 1:
                coordinate = (shell + 1, offset)
            else:
                coordinate = (2 * shell + 2 - offset, shell + 1)
            coordinates.append(coordinate)
        shell += 1
    return np.asarray(coordinates, dtype=np.float64)


def distance_matrix(positions: np.ndarray) -> np.ndarray:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("positions must have shape (nodes, dimensions)")
    difference = array[:, None, :] - array[None, :, :]
    return np.linalg.norm(difference, axis=-1)


def propagation_delays(positions: np.ndarray, delay_scale: float) -> np.ndarray:
    if delay_scale < 0:
        raise ValueError("delay_scale must be non-negative")
    return np.ceil(delay_scale * distance_matrix(positions)).astype(np.int64)
