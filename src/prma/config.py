"""Validated configuration objects used by models and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Parameters for the v0 deterministic cycle-learning baseline."""

    node_count: int = 32
    path_length: int = 8
    learning_rate: float = 0.2
    repetitions: int = 5
    passive_decay: float = 0.0
    competition: float = 0.05
    saturation: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ValueError("node_count must be at least 2")
        if not 2 <= self.path_length <= self.node_count:
            raise ValueError("path_length must be between 2 and node_count")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0 <= self.passive_decay < 1:
            raise ValueError("passive_decay must be in [0, 1)")
        if not 0 <= self.competition < 1:
            raise ValueError("competition must be in [0, 1)")
        if self.saturation <= 0:
            raise ValueError("saturation must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown model config fields: {', '.join(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a sequential capacity experiment."""

    model: ModelConfig = ModelConfig()
    memory_count: int = 16
    memory_id_start: int = 1

    def __post_init__(self) -> None:
        if self.memory_count < 1:
            raise ValueError("memory_count must be at least 1")
        if self.memory_id_start < 0:
            raise ValueError("memory_id_start must be non-negative")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        allowed = {"model", "memory_count", "memory_id_start"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown experiment config fields: {', '.join(unknown)}")
        payload = dict(values)
        payload["model"] = ModelConfig.from_dict(payload.get("model", {}))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "memory_count": self.memory_count,
            "memory_id_start": self.memory_id_start,
        }
