# Optional diagnostics

`invarlock[diagnostics]` is the optional feature for descriptive
numeric observations. It is useful when a paired decision needs supporting
engineering context. Its canonical JSON can travel inside the authenticated
`evaluate -> verify -> report` evidence transaction while remaining outside
the acceptance calculation.

> **User guide**
>
> **Outcome:** Produce deterministic numerical observations that support an
> investigation without changing an InvarLock acceptance decision.
>
> **Audience:** Engineers comparing numerical artifacts after conversion,
> quantization, or another externally managed transformation.
>
> **Prerequisites:** Fixed baseline and subject arrays, retained input
> provenance, and a completed or planned paired evidence transaction whose
> policy remains authoritative.

## Decide whether you need it

| Question | Use the diagnostics package? |
| --- | --- |
| Does the subject pass the paired metric under the caller-approved policy? | No. Use `invarlock verify`; diagnostics have no acceptance authority. |
| Did a model conversion produce an unexpected numerical shape? | Possibly. Compare observations from fixed baseline and subject arrays. |
| Do you need a calibrated safety, quality, or anomaly verdict? | No. The package produces observations, not calibrated verdicts. |
| Do you need a small deterministic summary for an investigation record? | Yes, provided the input array and its provenance are retained separately. |

Install it independently of the core runtime providers:

```bash
python -m pip install 'invarlock[diagnostics]'
```

The package has no provider hook or policy threshold. Its canonical JSON output
can be supplied as an `EvidenceObservation` when a caller publishes evidence;
the core wraps it in an authenticated, content-addressed envelope and renders
it in a separate report section. This keeps exploratory numerical analysis
visible without silently making it acceptance policy.

## Available observations

All three functions accept NumPy arrays or array-like numeric input, convert it
to `float64`, and reject empty, nonnumeric, non-finite, or oversized input with
`DiagnosticInputError`. The shared upper bound is 5,000,000 values. This is an
allocation guard, not a promise that exact SVD or eigenvalue decomposition at the
limit will be operationally cheap.

| Function | Required shape | Additional constraints | Result type |
| --- | --- | --- | --- |
| `spectral_observation` | Exactly two dimensions | At least one row and column | `SpectralObservation` |
| `rmt_observation` | Exactly two dimensions | At least two rows and one varying column; constant columns are reported and excluded | `RmtObservation` |
| `variance_observation` | One or more dimensions | At least one value; scalar/zero-dimensional input is rejected; singleton input reports population variance `0.0` and sample variance `null` | `VarianceObservation` |

Catch `DiagnosticInputError` when a caller needs to distinguish an invalid
investigation input from an unexpected implementation failure:

```python
from invarlock.diagnostics import DiagnosticInputError, rmt_observation

try:
    observation = rmt_observation([[1.0], [1.0]])
except DiagnosticInputError as exc:
    print(f"diagnostic input rejected: {exc}")
```

### Singular-value summary

`spectral_observation` computes exact singular values and a compact summary for
one finite matrix:

```python
import json

import numpy as np
from invarlock.diagnostics import spectral_observation

matrix = np.diag([3.0, 1.0])
observation = spectral_observation(matrix)

assert observation["status"] == "observation"
print(json.dumps(observation, indent=2, sort_keys=True))
```

Use it to describe the supplied matrix, not to infer model quality. Exact SVD
can be expensive for large matrices; take a fixed slice or projection when
the full calculation is not operationally justified, and record that choice.

### Covariance and Marchenko--Pastur reference

`rmt_observation` excludes constant columns, standardizes the remaining columns
by population deviation, computes covariance eigenvalues, and reports theoretical
Marchenko--Pastur reference edges:

```python
import numpy as np
from invarlock.diagnostics import rmt_observation

samples = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 5.0],
        [3.0, 5.0, 8.0],
    ]
)
observation = rmt_observation(samples)
assert observation["status"] == "observation"
```

The theoretical edges rely on idealized independent-sample assumptions. They
are reference values, not anomaly thresholds. Correlated activations, chosen
layers, preprocessing, and low sample counts can all dominate the result.

### Bound the covariance allocation

The default `method="covariance"` retains the original feature-covariance
calculation and result fields. Both methods check `max_gram_bytes` before
allocating the square Gram matrix. The default budget is 128 MiB; callers can
supply a positive integer byte budget. This bounds that matrix alone, not peak
process memory or decomposition workspace.

For wide inputs, explicitly select the smaller Gram calculation:

```python
import numpy as np
from invarlock.diagnostics import rmt_observation

samples = np.tile([[-1.0], [1.0]], (8, 64))
observation = rmt_observation(samples, method="smaller_gram")
assert observation["gram_dimension"] == 16
assert observation["gram_matrix_bytes"] == 2048
assert observation["empirical_eigenvalue_min"] == 0.0
```

For `n` samples and `p` varying features, this method decomposes an
`n` by `n` matrix when `p > n`; otherwise it uses the feature covariance.
The full covariance's omitted zero eigenvalues still determine its minimum,
and the fraction above the edge still uses all `p` varying features.
The result identifies `column_standardized_smaller_gram_eigh` and adds
`gram_dimension`, `gram_matrix_bytes` and `minimum_upper_edge_distance`.

The two calculations have the same nonzero eigenvalues in exact arithmetic.
Floating-point differences can change the strict count above a reference edge.
The distance field reports the smallest absolute distance to the upper edge,
including omitted zeros. A small gap helps identify numerical sensitivity; it
is neither an error bound nor a significance level. No tolerance is applied to
the count. Keep the method and numerical environment with the result; do not
silently replace a historical observation with a newly calculated one.

### Compare aligned arrays directly

Separate summaries can miss changes in matrix action. Retain both input
observations and also summarize `subject - baseline`:

```python
import numpy as np
from invarlock.diagnostics import spectral_observation

baseline = np.array([[0.0, 2.0], [0.0, 0.0]])
subject = np.array([[2.0, 0.0], [0.0, 0.0]])
# The caller must establish identical row and column meanings on both sides.
assert baseline.shape == subject.shape
left = spectral_observation(baseline)
right = spectral_observation(subject)
change = spectral_observation(subject - baseline)
relative_change = (
    change["singular_value_max"] / left["singular_value_max"]
    if left["singular_value_max"] > 0 else None
)
assert left["singular_value_max"] == right["singular_value_max"] == 2.0
assert change["singular_value_max"] > 2.8
```

Equal shapes do not establish alignment. Retain the row/column identities,
selection, ordering and preprocessing separately. A basis change or permutation
can create a large difference while preserving model behavior. A zero baseline
has no defined relative change; report `null`, not an infinite value. Use finite
float arrays before subtraction so integer arithmetic cannot wrap; the diagnostic
rejects non-finite differences. Neither norm is a quality verdict.

Covariance eigenvalues also omit feature orientation. For a small, fixed set of
aligned features, compare the covariance matrices themselves. The following
correlation recipe requires every selected feature to vary on both sides:

```python
import numpy as np
from invarlock.diagnostics import spectral_observation

baseline = np.array([[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]])
subject = baseline.copy()
subject[:, 1] += 0.5 * subject[:, 0]
assert baseline.shape == subject.shape
assert baseline.shape[1] <= 256  # Fixed feature selection bounds this recipe.


def correlation(values):
    centered = values - values.mean(axis=0)
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("selected features must have finite, nonzero deviation")
    standardized = centered / scale
    return standardized.T @ standardized / len(values)


change = spectral_observation(correlation(subject) - correlation(baseline))
assert change["frobenius_norm"] > 0.0
```

For raw population covariance, omit division by `scale` and use centered values
in the matrix product. Raw covariance retains scale changes; correlation removes
each feature's scale. Declare this choice and the selected feature identities
before comparison. Never drop a different set of columns on each side. This
small-matrix recipe has quadratic feature cost; it does not inherit the smaller
Gram method's memory saving. Report its input digests and preparation alongside
the change observation.

### Scalar variance summary

`variance_observation` reports population and sample variance for a finite
sequence:

```python
from invarlock.diagnostics import variance_observation

observation = variance_observation([0.9, 1.0, 1.1, 1.2])
assert observation["status"] == "observation"
```

Choose the population or sample interpretation before comparing outputs. A
variance change alone does not identify its cause and is not an InvarLock
policy failure.

## Validate an observation

Before interpreting or retaining an observation, confirm that:

- the operation returned `status: observation`, not a policy verdict;
- both sides used the same array-selection, reshape, projection, and dtype
  rules;
- all reported values are finite and the documented population or sample
  convention matches the investigation;
- the exact input or its content digest can be recovered; and
- rerunning the operation over the same bytes produces the same JSON values.

If any check fails, correct the input preparation or investigation record and
recompute the observation. Do not compensate by changing evidence, policy, or
a signed receipt.

## Attach an authenticated observation

Create the diagnostic result, write its canonical JSON, and name it in a native
v1 evaluation request. Captured v2 requests do not accept this `observations`
field:

```python
from pathlib import Path

from invarlock.diagnostics import (
    canonical_observation_bytes,
    spectral_observation,
)

observation = spectral_observation([[3.0, 0.0], [0.0, 1.0]])
Path("observations").mkdir(exist_ok=True)
Path("observations/subject-spectral.json").write_bytes(
    canonical_observation_bytes(observation)
)
```

```yaml
observations:
  - id: subject-spectral
    kind: spectral
    scope: subject
    path: observations/subject-spectral.json
```

For native pack-v1 evidence, `invarlock evaluate` binds the observation to the
comparison, schedule, policy, and both artifact identities, includes its digest
in the signed manifest, and places it under `observations/`. Strict verification
rejects malformed, non-canonical, unbound, or tampered observations. The
verification JSON lists the authenticated observation ID, kind, scope, and digest.
Reports place the payload under **Authenticated observations** and state that it is outside
the acceptance calculation.

Native judge requests retain these observation payloads inside
`native_capture.json` under their separate signed judge envelope. Use the
[judge evidence contract](../reference/judge-measurements.md) for that layout
and its recipient verification.

Absence is valid. Adding a JSON file after publication still violates the
bundle's closed inventory and fails strict verification.

For each observation, retain the following provenance with the surrounding
investigation record:

- the exact input array or a content digest and stable locator;
- how the array was selected, reshaped, standardized, or projected;
- package and NumPy versions;
- the observation JSON; and
- the interpretation, clearly separated from the verified decision.

When diagnostic work leads to a new acceptance rule, define and calibrate that
rule outside the current bundle, review it, then create a new versioned policy
and a new evidence transaction. Do not reinterpret an existing signed receipt.

## Next step and upstream references

- [NumPy singular-value decomposition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
  documents the underlying exact SVD operation.
- [NumPy variance](https://numpy.org/doc/stable/reference/generated/numpy.var.html)
  documents population and degrees-of-freedom conventions.

Continue with [Schedule and policy](schedule-and-policy.md) for the canonical
acceptance decision and
[Evidence and verification](evidence-and-verification.md) for immutable output
handling.
