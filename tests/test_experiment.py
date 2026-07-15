from prma.config import ExperimentConfig, ModelConfig
from prma.experiment import run_capacity_experiment


def test_capacity_experiment_reports_provenance_and_per_memory_results() -> None:
    config = ExperimentConfig(
        model=ModelConfig(node_count=16, path_length=4, seed=11),
        memory_count=3,
    )
    result = run_capacity_experiment(config)
    assert result["schema_version"] == 1
    assert result["config"] == config.to_dict()
    assert result["summary"]["memory_count"] == 3
    assert result["summary"]["cue_count"] == 12
    assert len(result["per_memory"]) == 3
