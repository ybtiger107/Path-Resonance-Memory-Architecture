import numpy as np
import pytest

from prma.config import DynamicalConfig, DynamicalRecallConfig
from prma.recall import frozen_recall_config, run_partial_cue_recall


def training_config() -> DynamicalConfig:
    return DynamicalConfig(
        node_count=4,
        memory_id=2,
        steps=20,
        delay_scale=1.0,
        threshold=0.1,
        learning_rate=0.2,
        weight_decay=0.01,
        saturation=0.8,
        initial_weight=0.7,
        topology="1d",
        input_amplitude=2.0,
        input_steps=12,
        reference_frequency=0.1,
    )


def test_recall_config_round_trip() -> None:
    original = DynamicalRecallConfig(training=training_config(), cue_steps=4, free_steps=8)
    assert DynamicalRecallConfig.from_dict(original.to_dict()) == original


def test_partial_cue_must_be_shorter_than_effective_training_input() -> None:
    with pytest.raises(ValueError, match="shorter"):
        DynamicalRecallConfig(training=training_config(), cue_steps=12, free_steps=5)


def test_recall_derivation_freezes_learning_and_decay() -> None:
    protocol = DynamicalRecallConfig(training=training_config(), cue_steps=4, free_steps=8)
    recall = frozen_recall_config(protocol)
    assert recall.steps == 12
    assert recall.input_steps == 4
    assert recall.learning_rate == 0.0
    assert recall.weight_decay == 0.0
    assert recall.memory_id == protocol.training.memory_id
    assert recall.topology == protocol.training.topology


def test_partial_cue_trial_has_frozen_weights_and_explicit_phase_boundary() -> None:
    protocol = DynamicalRecallConfig(training=training_config(), cue_steps=4, free_steps=8)
    trial = run_partial_cue_recall(protocol, record_synapses=True)
    learned = trial.training_trace.final_weights
    history = trial.recall_trace.weight_history

    assert history is not None
    expected_history = np.broadcast_to(learned, history.shape)
    np.testing.assert_array_equal(history, expected_history)
    np.testing.assert_array_equal(trial.recall_trace.final_weights, learned)
    assert np.any(trial.recall_trace.external_inputs[: protocol.cue_steps, 0] > 0)
    assert not np.any(trial.recall_trace.external_inputs[protocol.cue_steps :])
    assert trial.cue_activations.shape == (protocol.cue_steps, protocol.training.node_count)
    assert trial.free_activations.shape == (protocol.free_steps, protocol.training.node_count)


def test_recall_is_cold_started_with_no_preserved_voltage_history() -> None:
    protocol = DynamicalRecallConfig(training=training_config(), cue_steps=4, free_steps=8)
    trial = run_partial_cue_recall(protocol)
    assert trial.recall_trace.voltages[0, 0] > 0
    assert not np.any(trial.recall_trace.voltages[0, 1:])
