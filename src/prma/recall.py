"""Training and partial-cue recall protocols for the dynamical model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from prma.config import DynamicalConfig, DynamicalRecallConfig
from prma.dynamics import DynamicalTrace, simulate_dynamics


@dataclass(frozen=True, slots=True)
class DynamicalRecallTrial:
    """Traces and phase boundaries from one cold-start recall trial."""

    protocol: DynamicalRecallConfig
    recall_config: DynamicalConfig
    training_trace: DynamicalTrace
    recall_trace: DynamicalTrace

    @property
    def cue_activations(self) -> np.ndarray:
        return self.recall_trace.activations[: self.protocol.cue_steps]

    @property
    def free_activations(self) -> np.ndarray:
        return self.recall_trace.activations[self.protocol.cue_steps :]


def frozen_recall_config(protocol: DynamicalRecallConfig) -> DynamicalConfig:
    """Derive the recall run while preserving all non-learning model semantics."""

    return replace(
        protocol.training,
        steps=protocol.cue_steps + protocol.free_steps,
        input_steps=protocol.cue_steps,
        learning_rate=0.0,
        weight_decay=0.0,
    )


def run_partial_cue_recall(
    protocol: DynamicalRecallConfig,
    *,
    record_synapses: bool = False,
) -> DynamicalRecallTrial:
    """Train once, then recall from a shorter same-frequency cue.

    Recall starts with zero voltage history and the learned final weight matrix.
    Both learning and decay are disabled, so the recall phase measures dynamics
    supported by stored structure rather than additional consolidation.
    """

    training_trace = simulate_dynamics(
        protocol.training,
        record_synapses=record_synapses,
    )
    recall_config = frozen_recall_config(protocol)
    recall_trace = simulate_dynamics(
        recall_config,
        initial_weights=training_trace.final_weights,
        record_synapses=record_synapses,
    )
    return DynamicalRecallTrial(
        protocol=protocol,
        recall_config=recall_config,
        training_trace=training_trace,
        recall_trace=recall_trace,
    )
