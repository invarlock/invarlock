# InvarLock diagnostics

`invarlock-diagnostics` is a standalone, first-party optional package for
descriptive numeric observations. It accepts caller-supplied arrays or
tensor-like values and computes three small summaries:

```bash
python -m pip install invarlock-diagnostics
```

- exact singular-value statistics (`spectral_observation`);
- standardized covariance and Marchenko–Pastur reference edges
  (`rmt_observation`); and
- scalar population/sample variance statistics (`variance_observation`).

```python
from pathlib import Path

import numpy as np

from invarlock_addins.diagnostics import (
    canonical_observation_bytes,
    spectral_observation,
)

observation = spectral_observation(np.diag([3.0, 1.0]))
assert observation["status"] == "observation"
Path("subject-spectral.json").write_bytes(
    canonical_observation_bytes(observation)
)
```

The package is deliberately separate from provider execution, policy
thresholds, and the acceptance calculation. Its outputs are descriptive
observations. An evaluation request can attach their canonical JSON through its
root `observations` array so the signed bundle authenticates and reports them.
The paired metric and policy remain the complete acceptance calculation for
`invarlock verify`.

All three functions accept real numeric NumPy arrays, Python sequences, or
torch-like values exposing `detach()`, `cpu()`, and `numpy()` without importing
PyTorch. Inputs must be nonempty, finite, representable as `float64`, and no
larger than 5,000,000 elements. Spectral and RMT observations require a
two-dimensional matrix; RMT also requires at least two sample rows and one
varying feature. Variance requires at least one dimension and reports
`sample_variance: null` for a singleton. Invalid input raises the public
`DiagnosticInputError` exception.

Every result is a closed typed mapping with format
`invarlock/diagnostic-observation-v1`, `status: observation`, a method name,
the input shape/count, and method-specific finite summaries. The exact public
result types are `SpectralObservation`, `RmtObservation`, and
`VarianceObservation`.

The RMT summary reports theoretical reference edges for column-standardized
samples. Those edges rely on idealized independent, identically distributed
assumptions. Model-quality and safety decisions require separate evidence.
Exact SVD is deterministic for fixed finite inputs but can be expensive for
large matrices.
