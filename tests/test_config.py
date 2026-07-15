import pytest

from prma.config import ExperimentConfig, ModelConfig


def test_model_config_rejects_impossible_path() -> None:
    with pytest.raises(ValueError, match="path_length"):
        ModelConfig(node_count=4, path_length=5)


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown model config"):
        ModelConfig.from_dict({"node_count": 8, "mystery": 1})


def test_experiment_config_round_trip() -> None:
    original = ExperimentConfig(model=ModelConfig(node_count=10, path_length=5), memory_count=3)
    assert ExperimentConfig.from_dict(original.to_dict()) == original
