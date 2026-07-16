# Partial-cue dynamical recall protocol

This document defines what a dynamical recall **trial** means in PRMA. It fixes
the experimental boundary before recall metrics are selected. Passing the
mechanistic tests described here is not, by itself, evidence of successful
memory recall.

## Question

After a memory-specific temporal input modifies the network weights, can a
strictly shorter prefix of that input trigger self-supported dynamics when no
further learning or external input is allowed?

The shorter prefix still contains information about the memory identifier. Later
benchmarks must therefore compare against untrained, shuffled-weight, wrong-ID,
and frequency-alias controls to show that any persistence comes from learned
structure rather than cue energy alone.

## Phase A: training

Run the Evermemory-compatible model using a validated `DynamicalConfig`:

```text
initial weights + full training input -> training trace -> learned final weights
```

Training may contain a post-input stabilization interval when `steps` is greater
than `input_steps`. The effective training input length is
`min(input_steps, steps)`.

## Phase B: frozen-weight recall

Construct a recall configuration from the training configuration with only these
changes:

```text
steps         = cue_steps + free_steps
input_steps   = cue_steps
learning_rate = 0
weight_decay  = 0
```

All topology, delay, threshold, voltage cap, frequency, amplitude, and memory-ID
settings remain identical. The learned final weight matrix is copied into the
recall run.

`cue_steps` must be positive and strictly shorter than the effective training
input. `free_steps` must be positive.

## Initial-state policy

Recall is **cold-started**:

- learned weights are preserved;
- voltage and activation histories are reset to zero;
- the cue sinusoid restarts at phase zero;
- no delayed event from training crosses into recall.

This isolates structural memory in the weights. A warm-start continuation may be
studied later, but it is a different protocol because residual voltage history can
carry memory independently of weights.

## Phase boundary

The recall trace is partitioned into:

```text
[0, cue_steps)                    partial-cue interval
[cue_steps, cue_steps+free_steps) unforced free-dynamics interval
```

Every external input must be zero in the second interval. The weight matrix must
equal the learned matrix at every recall step.

## What this step does not define

The protocol intentionally does not yet choose a success metric or target cycle.
The next benchmark step must define and validate:

- how a target path or attractor is extracted from training;
- transition accuracy and cycle re-entry;
- latency and post-cue persistence;
- minimum useful cue length;
- negative controls and frequency-alias handling.

The implementation is `prma.recall.run_partial_cue_recall`; mechanistic tests are
in `tests/test_recall.py`.

