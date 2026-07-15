"""Sequential capacity experiment with provenance-rich JSON output."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from prma import __version__
from prma.config import ExperimentConfig
from prma.metrics import mean_pairwise_edge_overlap
from prma.model import PathResonanceMemory


def _git_revision() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() or None


def run_capacity_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Learn memories sequentially, then evaluate all of them from every cue."""

    model = PathResonanceMemory(config.model)
    memory_ids = list(range(config.memory_id_start, config.memory_id_start + config.memory_count))
    cycles = [model.learn(memory_id) for memory_id in memory_ids]

    per_memory: list[dict[str, Any]] = []
    all_accuracies: list[float] = []
    exact_cues = 0
    cue_count = 0
    for memory_id, cycle in zip(memory_ids, cycles, strict=True):
        results = [model.evaluate(memory_id, cue=cue) for cue in cycle]
        accuracies = [result.transition_accuracy for result in results]
        exact = sum(result.exact for result in results)
        exact_cues += exact
        cue_count += len(results)
        all_accuracies.extend(accuracies)
        per_memory.append(
            {
                "memory_id": memory_id,
                "target_cycle": list(cycle),
                "mean_recall_accuracy": float(np.mean(accuracies)),
                "minimum_recall_accuracy": float(np.min(accuracies)),
                "exact_cue_rate": exact / len(results),
            }
        )

    return {
        "schema_version": 1,
        "model_version": model.MODEL_VERSION,
        "package_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "config": config.to_dict(),
        "summary": {
            "memory_count": len(memory_ids),
            "cue_count": cue_count,
            "mean_recall_accuracy": float(np.mean(all_accuracies)),
            "minimum_recall_accuracy": float(np.min(all_accuracies)),
            "exact_cue_rate": exact_cues / cue_count,
            "mean_pairwise_directed_edge_jaccard": mean_pairwise_edge_overlap(cycles),
            "synaptic_events": model.synaptic_events,
            "nonzero_weights": int(np.count_nonzero(model.weights)),
        },
        "per_memory": per_memory,
    }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    return ExperimentConfig.from_dict(payload)


def write_result(result: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
