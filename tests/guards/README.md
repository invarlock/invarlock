# Guards Tests

This directory contains tests for InvarLock's guard mechanisms: invariants,
spectral, RMT, and variance. Guards are a critical release-gate surface, so the
suite favors behavior-local files and ratcheted coverage over a minimal file
count.

## Organization

| Area | Directory | Notes |
| --- | --- | --- |
| Shared contracts | `contracts/` | Assurance shape, import safety, GPU-only contracts, and cross-guard regression matrices. |
| Invariants | `invariants/` | Structural checks, profile behavior, CLI/docs invariants, and summary counts. |
| Spectral | `spectral/` | Prepare/after-edit flow, measurement, policies, multiple-testing controls, and runtime paths. |
| RMT | `rmt/` | Detection, correction, activation helpers, runtime finalize paths, and edge branches. |
| Variance | `variance/` | Calibration, predictive A/B gate, scale computation, target resolution, and finalize paths. |
| Policy | `policy/` | Guard fallback policy, tier config, and policy branch coverage. |
| Property | `property/` | Hypothesis/property checks for monotonicity and selection behavior. |
| Differential | `differential/` | Parity checks between reference and optimized guard decisions. |

## Running Tests

```bash
PYTHONPATH=src pytest -q tests/guards
PYTHONPATH=src pytest -q tests/guards/variance
PYTHONPATH=src pytest -q tests/guards/spectral
PYTHONPATH=src pytest -q tests/guards/rmt
PYTHONPATH=src pytest -q tests/guards/invariants
PYTHONPATH=src pytest -q tests/guards/property
PYTHONPATH=src pytest -q tests/guards/differential
```

Use the repository-wide pytest markers from `pyproject.toml` (`integration`,
`slow`, `gpu`, `manual`, and related optional-dependency markers) only when a
guard test genuinely needs that behavior.

## Coverage

Guard modules are covered by `make coverage-enforce`. Critical guard files
listed in `scripts/coverage/check_coverage_thresholds.py` are ratcheted to 100%
per-file coverage; remaining guard modules are still covered by the core package
floor. New guard behavior should add meaningful tests with the same change
rather than lowering thresholds or relying on broad smoke coverage.
