"""Faithful implementation of the 2025 Evermemory Model-2 mixed dynamics.

The update order and index orientation intentionally match commit f708ad1 of
``ybtiger107/2025-Excool-Evermemory``. The source repository is MIT licensed.
Compatibility quirks are documented rather than silently corrected here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from prma.config import DynamicalConfig
from prma.topology import positions_1d, positions_2d, propagation_delays


@dataclass(frozen=True, slots=True)
class DynamicalTrace:
    """Recorded state with arrays indexed as described in ``docs/DYNAMICS.md``."""

    voltages: np.ndarray
    activations: np.ndarray
    external_inputs: np.ndarray
    weight_history: np.ndarray | None
    contribution_history: np.ndarray | None
    final_weights: np.ndarray
    positions: np.ndarray
    delays: np.ndarray


def legacy_input_series(config: DynamicalConfig) -> np.ndarray:
    """Generate the exact Model-2 sampled sinusoidal input."""

    sample_frequency = 10.0 * config.reference_frequency
    active_steps = min(config.steps, int(config.input_steps))
    series = np.zeros(config.steps, dtype=np.float64)
    time = np.arange(active_steps, dtype=np.float64) / sample_frequency
    series[:active_steps] = (
        config.threshold
        * config.input_amplitude
        * 0.5
        * (np.sin(2.0 * math.pi * (config.memory_id + 1) * (time / 30.0)) + 1.0)
    )
    return series


def legacy_alias_period(reference_frequency: float) -> int | None:
    """Return the exact memory-ID alias period when it is integral.

    The phase increment is ``2π(p+1)/(300*f_ref)`` radians per sample. Thus IDs
    separated by ``300*f_ref`` generate identical samples when that denominator
    is an integer. ``None`` means no integer period is asserted.
    """

    denominator = 300.0 * reference_frequency
    rounded = round(denominator)
    if math.isclose(denominator, rounded, rel_tol=0.0, abs_tol=1e-12):
        return int(rounded)
    return None


def _initial_weights(config: DynamicalConfig, initial_weights: np.ndarray | None) -> np.ndarray:
    shape = (config.node_count, config.node_count)
    if initial_weights is None:
        weights = config.initial_weight * (
            np.ones(shape, dtype=np.float64) - np.eye(config.node_count)
        )
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.shape != shape:
            raise ValueError(f"initial_weights must have shape {shape}")
        if np.any(weights < 0) or np.any(weights > config.saturation):
            raise ValueError("initial_weights must be in [0, saturation]")
        np.fill_diagonal(weights, 0.0)
    return weights


def simulate_dynamics(
    config: DynamicalConfig,
    *,
    initial_weights: np.ndarray | None = None,
    record_synapses: bool = True,
) -> DynamicalTrace:
    """Run the dense reference dynamics in the legacy operation order.

    Set ``record_synapses=False`` to reduce trace storage from O(T*N²) to O(T*N),
    while retaining identical dynamics and final weights.
    """

    positions = (
        positions_2d(config.node_count)
        if config.topology == "2d"
        else positions_1d(config.node_count)
    )
    delays = propagation_delays(positions, config.delay_scale)
    input_series = legacy_input_series(config)

    voltages = np.zeros((config.steps, config.node_count), dtype=np.float64)
    activations = np.zeros((config.steps, config.node_count), dtype=np.int64)
    external_inputs = np.zeros((config.steps, config.node_count), dtype=np.float64)
    weight_history = (
        np.zeros((config.steps, config.node_count, config.node_count), dtype=np.float64)
        if record_synapses
        else None
    )
    contributions = (
        np.zeros((config.steps, config.node_count, config.node_count), dtype=np.float64)
        if record_synapses
        else None
    )
    weights = _initial_weights(config, initial_weights)

    for time_index in range(config.steps):
        external = np.zeros(config.node_count, dtype=np.float64)
        external[0] = input_series[time_index]
        external_inputs[time_index] = external

        recurrent = np.zeros(config.node_count, dtype=np.float64)
        for pre in range(config.node_count):
            for post in range(config.node_count):
                delay = int(delays[pre, post])
                if delay <= 0 or time_index - delay < 0:
                    continue
                contribution = weights[pre, post] * voltages[time_index - delay, pre]
                recurrent[post] += contribution
                if contributions is not None:
                    contributions[time_index, pre, post] = contribution

        current_voltages = np.clip(external + recurrent, 0.0, config.voltage_cap)
        voltages[time_index] = current_voltages
        current_activations = (current_voltages > config.threshold).astype(np.int64)
        activations[time_index] = current_activations

        below_saturation = weights < config.saturation
        weights[below_saturation] *= 1.0 - config.weight_decay

        # Legacy learning is gated by positive external input at node zero.
        if external[0] > 0.0:
            for pre in range(config.node_count):
                for post in range(config.node_count):
                    if pre == post:
                        weights[pre, post] = 0.0
                        continue
                    if weights[pre, post] >= config.saturation:
                        continue
                    delay = int(delays[pre, post])
                    if delay <= 0 or time_index - delay < 0:
                        continue
                    if activations[time_index - delay, pre] == 1 and current_activations[post] == 1:
                        weights[pre, post] += config.learning_rate
                        if weights[pre, post] >= config.saturation:
                            weights[pre, post] = config.saturation
                            weights[post, pre] = 0.0
                            for other in range(config.node_count):
                                if other != pre and other != post:
                                    weights[pre, other] = 0.0
                                    weights[other, post] = 0.0

        np.fill_diagonal(weights, 0.0)
        weights = np.clip(weights, 0.0, config.saturation)
        if weight_history is not None:
            weight_history[time_index] = weights

    return DynamicalTrace(
        voltages=voltages,
        activations=activations,
        external_inputs=external_inputs,
        weight_history=weight_history,
        contribution_history=contributions,
        final_weights=weights.copy(),
        positions=positions,
        delays=delays,
    )
