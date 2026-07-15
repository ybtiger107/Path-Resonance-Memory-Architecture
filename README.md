# Path-Resonance Memory Architecture (PRMA)

PRMA is a reproducible research platform for studying memories represented by
directed activation paths in a shared network. It is the successor to
[2025-Excool-Evermemory](https://github.com/ybtiger107/2025-Excool-Evermemory),
with explicit model contracts, persistent sequential learning, quantitative
benchmarks, tests, and experiment provenance.

> **Research status:** early baseline (`v0.1`). The included model is a reference
> implementation, not evidence for biological plausibility, hardware efficiency,
> or state-of-the-art memory capacity. Such claims require the benchmark and
> ablation program described in `docs/RESEARCH_ROADMAP.md`.

## What is included

- A deterministic memory-to-cycle encoder without frequency aliasing
- A stateful path-learning model that retains weights across memories
- An Evermemory-compatible spatial dynamical simulator with golden-fixture parity
- Recall from any node on a learned cycle
- Capacity, interference, overlap, and event-cost measurements
- JSON experiment configs and machine-readable result manifests
- Unit tests, linting, and GitHub Actions

## Quick start

PRMA requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
pytest
prma experiment --config configs/baseline.json --output runs/baseline.json
prma dynamics --config configs/legacy-dynamics-smoke.json --output runs/dynamics.json
```

During active development, run directly from the source tree so every edit is
picked up without reinstalling:

```bash
PYTHONPATH=src python -m prma experiment \
  --config configs/baseline.json \
  --output runs/baseline.json
```

## Repository layout

```text
src/prma/              versioned model and experiment code
configs/               reviewed experiment configurations
experiments/           research entry points and future sweeps
tests/                 deterministic regression tests
docs/                  architecture, reproducibility, and roadmap
.github/                CI and repository-management templates
```

## Scientific contract

Every reported result should identify the code revision, complete configuration,
seed, environment, per-memory measurements, and aggregate statistics. Generated
run artifacts are ignored by Git by default; publish durable datasets through a
versioned release or an external data registry and record their checksum.

See [Architecture](docs/ARCHITECTURE.md),
[Dynamical equations](docs/DYNAMICS.md),
[Reproducibility](docs/REPRODUCIBILITY.md),
[Legacy migration](docs/LEGACY_MIGRATION.md), and
[Contributing](CONTRIBUTING.md) before extending the model.
