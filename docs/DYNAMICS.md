# Evermemory-compatible dynamics

This document fixes the semantics reproduced by `prma.dynamics` from legacy
Evermemory commit `f708ad1`. It describes compatibility, including behaviors that
may later be changed in a separately versioned model.

## State and orientation

For `N` nodes and discrete step `t`:

- `V[t, i]` is the voltage of node `i`.
- `A[t, i]` is its binary activation.
- `W[t, i, j]` is the weight **from presynaptic `i` to postsynaptic `j`**, recorded
  after decay and learning at step `t`.
- `C[t, i, j]` is the contribution from `i` to `j` used during step `t`, before
  that step's weight update.

The initial matrix has `W[i,j] = initial_weight` for `i != j` and zero diagonal.

## Topology and propagation

Nodes use either integer positions on a line or the legacy expanding square-shell
ordering. For Euclidean position `x_i`, propagation delay is

```text
D[i,j] = ceil(delay_scale * ||x_i - x_j||₂).
```

The recurrent contribution and total voltage are

```text
C[t,i,j] = W_before[t,i,j] * V[t-D[i,j], i]
R[t,j]   = Σ_i C[t,i,j]
V[t,j]   = clip(I[t,j] + R[t,j], 0, voltage_cap),
```

where a contribution is zero when `D[i,j] <= 0` or `t-D[i,j] < 0`. There is no
leak, reset, or direct carry-over of the previous voltage.

Activation uses a strict threshold:

```text
A[t,j] = 1 when V[t,j] > threshold, else 0.
```

## External input

Only node zero receives input. With sample frequency `f_s = 10*f_ref`, step `t`,
memory identifier `p`, and amplitude `b`:

```text
I[t,0] = threshold * b/2 *
         (sin(2π(p+1)t / (30 f_s)) + 1)
```

for `t < input_steps`; all other inputs are zero. The legacy comment describes
an additional `(p mod 10)` amplitude factor, but the executable code does not use
it, so the compatible implementation does not either.

The digital phase increment is `2π(p+1)/(300*f_ref)`. With the default
`f_ref=30`, identifiers separated by 9000 produce identical sampled input. PRMA
reports this alias period explicitly rather than treating every integer as unique.

## Weight update order

At every step, all unsaturated weights first decay:

```text
W <- W * (1 - weight_decay), when W < saturation.
```

Despite separate branches in the legacy source, this decay occurs both while
external input is present and after it ends. Learning occurs only when
`I[t,0] > 0`. An eligible edge satisfies

```text
A[t-D[i,j], i] = 1 and A[t,j] = 1,
```

and receives `W[i,j] <- W[i,j] + learning_rate`. When it reaches saturation:

1. `W[i,j]` is clipped to saturation;
2. the reverse edge `W[j,i]` becomes zero;
3. every other edge in row `i` becomes zero;
4. every other edge in column `j` becomes zero.

The legacy comment calls step 3 “other outgoing edges from post,” but the code
actually clears outgoing edges from `pre` and incoming edges to `post`. PRMA
matches the code. Nested `pre` then `post` iteration means a saturation event can
affect edges considered later in the same step.

Finally, the diagonal is zeroed and all weights are clipped to `[0, saturation]`.

## Complexity and recording modes

The compatibility engine is dense and runs in `O(T*N²)` time. Full recording uses
`O(T*N²)` memory for weight and contribution histories. Compact recording omits
those histories and uses `O(T*N + N²)` memory without changing trajectories or
final weights; parity is tested.

A truly event-driven sparse engine remains a separate optimization milestone. It
must match this reference on compatible sparse configurations before replacing it
in large experiments.

## Provenance

The golden fixture and checksums are documented in `tests/fixtures/README.md`.
The fixture exercises delayed propagation, threshold activation, Hebbian updates,
saturation, reverse-edge removal, and row/column suppression.

