import hashlib
import json
from pathlib import Path

import numpy as np

from prma.config import DynamicalConfig
from prma.dynamics import legacy_alias_period, legacy_input_series, simulate_dynamics

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_model2_golden.json"
FIXTURE_SHA256 = "74f879491c5d013a70f8a8ce8090906d44f770d7c200b271528758df15d3efdc"


def test_golden_fixture_checksum() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_trace_matches_archived_legacy_model2_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    trace = simulate_dynamics(DynamicalConfig.from_dict(fixture["config"]))
    expected = fixture["expected"]

    np.testing.assert_allclose(trace.positions, expected["positions"], rtol=0, atol=0)
    np.testing.assert_array_equal(trace.delays, expected["delays"])
    np.testing.assert_allclose(trace.voltages, expected["voltages"], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(trace.activations, expected["activations"])
    np.testing.assert_allclose(
        trace.external_inputs, expected["external_inputs"], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(trace.final_weights, expected["final_weights"], rtol=0, atol=1e-15)


def test_compact_recording_preserves_dynamics() -> None:
    config = DynamicalConfig(
        node_count=5,
        memory_id=3,
        steps=20,
        delay_scale=1.0,
        topology="1d",
        threshold=0.1,
        initial_weight=0.7,
        input_steps=10,
        reference_frequency=0.2,
    )
    full = simulate_dynamics(config, record_synapses=True)
    compact = simulate_dynamics(config, record_synapses=False)
    assert compact.weight_history is None
    assert compact.contribution_history is None
    np.testing.assert_array_equal(compact.voltages, full.voltages)
    np.testing.assert_array_equal(compact.activations, full.activations)
    np.testing.assert_array_equal(compact.final_weights, full.final_weights)


def test_default_frequency_encoding_has_exact_id_aliases() -> None:
    period = legacy_alias_period(30.0)
    assert period == 9000
    first = DynamicalConfig(node_count=2, memory_id=7, steps=40, reference_frequency=30.0)
    alias = DynamicalConfig(
        node_count=2,
        memory_id=7 + period,
        steps=40,
        reference_frequency=30.0,
    )
    np.testing.assert_allclose(legacy_input_series(first), legacy_input_series(alias), atol=1e-14)


def test_strict_threshold_does_not_activate_equal_voltage() -> None:
    config = DynamicalConfig(
        node_count=1,
        memory_id=0,
        steps=1,
        threshold=1.0,
        input_amplitude=1.0,
    )
    trace = simulate_dynamics(config)
    assert trace.voltages[0, 0] == 0.5
    assert trace.activations[0, 0] == 0
