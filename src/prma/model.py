"""Stateful reference implementation of path-resonance learning and recall."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prma.config import ModelConfig
from prma.encoding import cycle_edges, encode_cycle


@dataclass(frozen=True, slots=True)
class RecallResult:
    memory_id: int
    cue: int
    expected: tuple[int, ...]
    recalled: tuple[int, ...]
    transition_accuracy: float
    exact: bool


class PathResonanceMemory:
    """Persistent directed-weight memory.

    This v0 baseline exposes its simplifications deliberately: an identifier is
    mapped to a target simple cycle, each presentation reinforces its directed
    edges, competing outgoing edges are inhibited, and recall follows the
    strongest outgoing transition. It is a falsifiable software baseline for
    later temporal/spiking models, not a claim of biological equivalence.
    """

    MODEL_VERSION = "prma-cycle-v0"

    def __init__(self, config: ModelConfig, weights: np.ndarray | None = None) -> None:
        self.config = config
        shape = (config.node_count, config.node_count)
        if weights is None:
            self.weights = np.zeros(shape, dtype=np.float64)
        else:
            array = np.asarray(weights, dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"weights must have shape {shape}")
            self.weights = array.copy()
        np.fill_diagonal(self.weights, 0.0)
        self.presentations = 0
        self.synaptic_events = 0

    def target_cycle(self, memory_id: int) -> tuple[int, ...]:
        return encode_cycle(
            memory_id,
            node_count=self.config.node_count,
            path_length=self.config.path_length,
            seed=self.config.seed,
        )

    def learn(self, memory_id: int) -> tuple[int, ...]:
        """Present one memory repeatedly while preserving all prior weights."""

        cycle = self.target_cycle(memory_id)
        edges = cycle_edges(cycle)
        edge_set = set(edges)

        for _ in range(self.config.repetitions):
            if self.config.passive_decay:
                self.weights *= 1.0 - self.config.passive_decay

            for source, target in edges:
                if self.config.competition:
                    mask = np.ones(self.config.node_count, dtype=bool)
                    mask[target] = False
                    mask[source] = False
                    self.weights[source, mask] *= 1.0 - self.config.competition
                    self.synaptic_events += int(mask.sum())

                self.weights[source, target] = min(
                    self.config.saturation,
                    self.weights[source, target] + self.config.learning_rate,
                )
                self.synaptic_events += 1

            # The closing path should not silently mutate during refactors.
            assert set(cycle_edges(cycle)) == edge_set
            np.fill_diagonal(self.weights, 0.0)

        self.presentations += self.config.repetitions
        return cycle

    def recall_path(self, cue: int, steps: int | None = None) -> tuple[int, ...]:
        """Follow the strongest directed transitions from ``cue``."""

        if not 0 <= cue < self.config.node_count:
            raise ValueError("cue is outside the network")
        step_count = self.config.path_length if steps is None else steps
        if step_count < 1:
            raise ValueError("steps must be positive")

        path = [cue]
        current = cue
        for _ in range(step_count - 1):
            row = self.weights[current]
            if float(row.max()) <= 0:
                break
            current = int(np.argmax(row))
            path.append(current)
        return tuple(path)

    def evaluate(self, memory_id: int, cue: int | None = None) -> RecallResult:
        cycle = self.target_cycle(memory_id)
        cue_node = cycle[0] if cue is None else cue
        if cue_node not in cycle:
            raise ValueError("cue must be a node in the target cycle")
        offset = cycle.index(cue_node)
        expected = cycle[offset:] + cycle[:offset]
        recalled = self.recall_path(cue_node, steps=len(cycle))
        correct = sum(a == b for a, b in zip(expected, recalled, strict=False))
        accuracy = correct / len(expected)
        return RecallResult(
            memory_id=memory_id,
            cue=cue_node,
            expected=expected,
            recalled=recalled,
            transition_accuracy=accuracy,
            exact=recalled == expected,
        )

    def save(self, path: str | Path) -> None:
        """Save weights and the complete model configuration."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            weights=self.weights,
            config=np.array([self.config.to_dict()], dtype=object),
            presentations=np.array([self.presentations], dtype=np.int64),
            synaptic_events=np.array([self.synaptic_events], dtype=np.int64),
            model_version=np.array([self.MODEL_VERSION]),
        )

    @classmethod
    def load(cls, path: str | Path) -> PathResonanceMemory:
        with np.load(path, allow_pickle=True) as archive:
            version = str(archive["model_version"][0])
            if version != cls.MODEL_VERSION:
                raise ValueError(f"unsupported model version: {version}")
            config = ModelConfig.from_dict(dict(archive["config"][0]))
            model = cls(config, weights=archive["weights"])
            model.presentations = int(archive["presentations"][0])
            model.synaptic_events = int(archive["synaptic_events"][0])
            return model
