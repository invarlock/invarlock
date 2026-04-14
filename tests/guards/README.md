# Guards Tests

This directory contains tests for InvarLock's guard mechanisms—the safety systems
that detect and validate model edits.

## Organization

This structure (many small, focused test modules) is intentional. Guards are a
critical surface, and the test suite is optimized for targeted coverage and fast
local iteration rather than a minimal file count.

### Naming conventions

| Area | Pattern | Notes |
|------|---------|-------|
| Variance guard | `test_variance_*.py` | Calibration, gates, scale computation, finalize paths, branch coverage |
| Spectral guard | `test_spectral_*.py` | Prepare/after_edit flows, scope filters, multiple testing enforcement |
| RMT guard | `test_rmt_*.py` | Detection/correction algorithms, helpers, verbose/edge branches |
| Invariants guard | `test_invariants_*.py` | Structural checks, API/CLI/docs invariants |
| Policies/tier config | `test_guard_policies.py`, `test_tier_config.py`, etc. | Runtime policy parsing and validation |

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
PYTHONPATH=src pytest tests/guards/differential/ -v

# Property-based guard tests
PYTHONPATH=src pytest tests/guards/property/ -v
```

## Markers

- `unit`: Focused unit tests (default)
- `slow`: Long-running tests
- `gpu`: Requires CUDA/MPS

## Coverage Targets

Per CONTRIBUTING.md, guards are part of the critical surface:
- **Target: ≥90% branch coverage** for all files in `src/invarlock/guards/`

## Related Test Directories

- `tests/guards/differential/` - Tests verifying parity between reference and optimized implementations
- `tests/guards/property/` - Property-based tests using Hypothesis
