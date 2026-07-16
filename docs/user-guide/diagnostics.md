# Optional diagnostics

`invarlock-diagnostics` is a standalone first-party package for descriptive
numeric observations. It is useful when a paired decision needs supporting
engineering context. Its canonical JSON can travel inside the authenticated
`evaluate -> verify -> report` evidence transaction while remaining outside
the acceptance calculation.

!!! tip "User guide"

    **Outcome:** Produce deterministic numerical observations that support an
    investigation without changing an InvarLock acceptance decision.

    **Audience:** Engineers comparing numerical artifacts after conversion,
    quantization, or another externally managed transformation.

    **Prerequisites:** Fixed baseline and subject arrays, retained input
    provenance, and a completed or planned paired evidence transaction whose
    policy remains authoritative.

## Decide whether you need it

| Question | Use the diagnostics package? |
| --- | --- |
| Does the subject pass the paired metric under the caller-approved policy? | No. Use `invarlock verify`; diagnostics have no acceptance authority. |
| Did a model conversion produce an unexpected numerical shape? | Possibly. Compare observations from fixed baseline and subject arrays. |
| Do you need a calibrated safety, quality, or anomaly verdict? | No. The package produces observations, not calibrated verdicts. |
| Do you need a small deterministic summary for an investigation record? | Yes, provided the input array and its provenance are retained separately. |

Install it independently of the core runtime providers:

```bash
python -m pip install invarlock-diagnostics
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
| `rmt_observation` | Exactly two dimensions | At least two rows and one column; every standardized column must have nonzero sample deviation | `RmtObservation` |
| `variance_observation` | One or more dimensions | At least one value; scalar/zero-dimensional input is rejected; singleton input reports population variance `0.0` and sample variance `null` | `VarianceObservation` |

Catch `DiagnosticInputError` when a caller needs to distinguish an invalid
investigation input from an unexpected implementation failure:

```python
from invarlock_addins.diagnostics import DiagnosticInputError, rmt_observation

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
from invarlock_addins.diagnostics import spectral_observation

matrix = np.diag([3.0, 1.0])
observation = spectral_observation(matrix)

assert observation["status"] == "observation"
print(json.dumps(observation, indent=2, sort_keys=True))
```

Use it to describe the supplied matrix, not to infer model quality. Exact SVD
can be expensive for large matrices; take a fixed slice or projection when
the full calculation is not operationally justified, and record that choice.

### Covariance and Marchenko--Pastur reference

`rmt_observation` standardizes columns, computes covariance eigenvalues, and
reports theoretical Marchenko--Pastur reference edges:

```python
import numpy as np
from invarlock_addins.diagnostics import rmt_observation

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

### Scalar variance summary

`variance_observation` reports population and sample variance for a finite
sequence:

```python
from invarlock_addins.diagnostics import variance_observation

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

Create the diagnostic result, write its canonical JSON, and name it in the
evaluation request:

```python
from pathlib import Path

from invarlock_addins.diagnostics import canonical_observation_bytes

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

`invarlock evaluate` binds the observation to the comparison, schedule, policy,
and both artifact identities, includes its digest in the signed manifest, and
places it under `observations/`. Strict verification rejects malformed,
non-canonical, unbound, or tampered observations. The verification JSON lists
the authenticated observation ID, kind, scope, and digest. Human reports place
the payload under **Authenticated observations** and state that it is outside
the acceptance calculation.

Absence is valid. Adding a JSON file after publication still violates the
bundle's closed inventory and fails strict verification.

For each observation, retain the following provenance with the surrounding
investigation record:

- the exact input array or a content digest and stable locator;
- how the array was selected, reshaped, standardized, or projected;
- package and NumPy versions;
- the observation JSON; and
- the human interpretation, clearly separated from the verified decision.

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
