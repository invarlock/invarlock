# Guards Tests

This directory contains tests for InvarLock's guard mechanisms—the safety systems
that detect and validate model edits.

## Organization

### Current Structure (Legacy)

The current test files are organized by specific functionality, often with one file
per test case or small feature. This structure is being consolidated.

### Target Structure (Consolidated)

| File | Description |
|------|-------------|
| `test_variance_core.py` | Core guard lifecycle and utilities |
| `test_variance_ab_gate.py` | A/B testing gate logic |
| `test_variance_calibration.py` | Calibration and PPL computation |
| `test_variance_scales.py` | Scale computation and filtering |
| `test_variance_equalise.py` | Variance equalization functions |
| `test_variance_finalize.py` | Finalization and edge cases |
| `test_variance_hooks.py` | Hook management (enable/disable) |
| `test_variance_edge_cases.py` | Edge cases and error paths |
| `test_spectral_guard.py` | Core spectral guard |
| `test_spectral_filters.py` | Scope and path filtering |
| `test_spectral_utils.py` | Utilities and helpers |
| `test_rmt_guard.py` | Core RMT guard |
| `test_rmt_corrections.py` | Correction algorithms |
| `test_rmt_utils.py` | Utilities |
| `test_invariants_guard.py` | Structural invariants |
| `test_policies.py` | Guard policy configuration |

## Guard Types

### Variance Guard (`variance.py`)
Data-driven variance equalization (DD-VE) for transformer blocks.
Measures and scales projection weights to maintain stable residual stream dynamics.

### Spectral Guard (`spectral.py`)
Analyzes weight matrix spectra for anomalous changes indicating corruption or
unexpected edits.

### RMT Guard (`rmt.py`)
Random Matrix Theory-based detection of meaningful vs. noise in weight changes.

### Invariants Guard (`invariants.py`)
Validates structural invariants like weight tying and architecture consistency.

## Running Tests

```bash
# Fast variance tests only
PYTHONPATH=src pytest tests/guards/test_variance_*.py -v

# Full guard suite
PYTHONPATH=src pytest tests/guards/ -v

# Specific guard
PYTHONPATH=src pytest tests/guards/test_spectral_*.py -v

# Differential tests (guard implementation parity)
PYTHONPATH=src pytest tests/guards_differential/ -v
```

## Markers

- `unit`: Focused unit tests (default)
- `slow`: Long-running tests
- `gpu`: Requires CUDA/MPS

## Coverage Targets

Per CONTRIBUTING.md, guards are part of the critical surface:
- **Target: ≥90% branch coverage** for all files in `src/invarlock/guards/`

## Related Test Directories

- `tests/guards_differential/` - Tests verifying parity between reference and optimized implementations
- `tests/guards_property/` - Property-based tests using Hypothesis
