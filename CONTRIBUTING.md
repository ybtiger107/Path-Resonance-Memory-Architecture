# Contributing

PRMA is research software. A change is complete only when its scientific meaning
and its implementation are both reviewable.

## Workflow

1. Open an issue describing the hypothesis, bug, or maintenance goal.
2. Work on a short-lived branch named `feat/…`, `fix/…`, `exp/…`, or `docs/…`.
3. Add or update tests and document changes to model semantics.
4. Run `ruff check .`, `ruff format --check .`, and `pytest`.
5. Submit a pull request using the repository template.

Do not commit generated runs, large arrays, secrets, or API credentials. Small
test fixtures are allowed when they are stable and documented.

## Research changes

Changes to encoding, learning, recall, metrics, or default parameters must include:

- the hypothesis and expected direction of change;
- a baseline comparison using identical seeds;
- negative or null results, not only the best run;
- runtime and event-cost changes;
- a statement about backward compatibility of saved states.

Use semantic versioning for public interfaces. Model-semantic changes must update
the `model_version` recorded in experiment output.
