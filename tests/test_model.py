import numpy as np

from prma.config import ModelConfig
from prma.model import PathResonanceMemory


def test_single_memory_is_recalled_from_every_cycle_node() -> None:
    config = ModelConfig(
        node_count=12,
        path_length=6,
        learning_rate=0.5,
        repetitions=2,
        competition=0.0,
    )
    model = PathResonanceMemory(config)
    cycle = model.learn(1)
    assert all(model.evaluate(1, cue=cue).exact for cue in cycle)


def test_learning_is_stateful_across_memories() -> None:
    config = ModelConfig(node_count=20, path_length=4, competition=0.0)
    model = PathResonanceMemory(config)
    model.learn(1)
    first_state = model.weights.copy()
    model.learn(2)
    assert model.presentations == 2 * config.repetitions
    assert np.count_nonzero(model.weights) >= np.count_nonzero(first_state)
    assert np.all(model.weights[first_state > 0] >= first_state[first_state > 0])


def test_saved_model_round_trip(tmp_path) -> None:
    model = PathResonanceMemory(ModelConfig(node_count=10, path_length=4))
    model.learn(9)
    target = tmp_path / "state.npz"
    model.save(target)
    restored = PathResonanceMemory.load(target)
    np.testing.assert_array_equal(restored.weights, model.weights)
    assert restored.config == model.config
    assert restored.presentations == model.presentations
    assert restored.synaptic_events == model.synaptic_events
