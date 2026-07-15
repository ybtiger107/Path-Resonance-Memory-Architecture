# Architecture

## Scope

PRMA studies whether persistent memories can be represented and recalled as
directed paths in a shared network. The software is divided into contracts so a
new temporal, spiking, or hardware-aware model can be compared with the baseline
without silently changing encoding or metrics.

```text
memory identifier
      │
      ▼
deterministic encoder ──► target directed cycle
                              │
                              ▼
                       persistent learner
                              │
                              ▼
cue ───────────────────► path recall ──► metrics
```

## Baseline v0

For a memory identifier `m`, the encoder ranks network nodes with a seeded BLAKE2
digest and selects the first `L` nodes as a simple directed cycle. Learning adds
`learning_rate` to each target edge on every presentation, bounded by
`saturation`. Optional passive decay applies globally, while competition inhibits
other outgoing edges from an active source.

Recall starts at a cue node and repeatedly follows the largest outgoing weight.
Ties use the lowest node index through NumPy's deterministic `argmax`. A memory is
evaluated from every node in its cycle, preventing a favorable fixed cue from
hiding failures.

This encoder solves the old frequency-aliasing problem by making mapping discrete
and explicit. It does **not** guarantee disjoint cycles. Shared nodes and edges are
the source of measurable interference.

## Stable interfaces

- `ModelConfig`: validated model semantics
- `PathResonanceMemory.learn`: sequential state mutation
- `PathResonanceMemory.evaluate`: cue-conditioned recall result
- `run_capacity_experiment`: provenance-rich benchmark result

Saved model files include `model_version`; incompatible semantic changes must use
a new version and, where practical, a migration utility.

## Known limitations

- The target cycle is supplied during learning rather than emerging from physical
  neuron dynamics.
- Recall is deterministic winner-take-all traversal.
- The event count is a computational proxy, not joules or measured hardware power.
- There is no noisy or partial cue representation beyond choosing a cycle node.
- The baseline has no spatial topology, propagation delay, or differentiable loss.

These limitations define the next experiments rather than hidden assumptions.

