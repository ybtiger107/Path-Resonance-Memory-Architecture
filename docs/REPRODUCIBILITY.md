# Reproducibility

## Minimum record for a result

Every PRMA result must retain:

- the complete JSON configuration;
- Git revision and PRMA model version;
- Python, NumPy, operating-system information;
- seed and memory identifier range;
- aggregate and per-memory metrics;
- dataset or artifact checksums when external inputs are used.

The standard experiment runner writes these fields automatically. A Git revision
is `null` before the repository has its first commit; published results must come
from a committed revision.

## Comparing models

Use identical encoded memories, memory ordering, cues, and seeds. Report the full
seed distribution with confidence intervals, not only the best seed. Parameter
selection and final evaluation must use separate experiment sets once optimization
is introduced.

## Artifacts

The `runs/` and `artifacts/` directories are ignored because raw sweeps can become
large. Release-worthy artifacts should be immutable, content-addressed, and linked
from a release manifest. Do not claim reproduction from an unversioned local file.

