# Experiments

Experiment logic belongs in the importable `prma` package. This directory holds
reviewed sweep definitions and short entry points only; notebooks must not become
the sole implementation of a reported result.

The first registered experiment is `configs/baseline.json`, a sequential capacity
check for the deterministic cycle baseline. Run artifacts go to `runs/` and are
not committed by default.

`configs/legacy-dynamics-smoke.json` is a compatibility smoke test, not a recall
benchmark. The partial-cue training/recall boundary is now specified in
`docs/RECALL_PROTOCOL.md`; metrics and the first registered recall experiment are
the next research step.
