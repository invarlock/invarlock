"""
InvarLock: auditable strict verification for edited model checkpoints
=====================================================================

Core runtime package — torch-independent utilities, configuration, and interfaces.

This package provides the foundation for the InvarLock GuardChain without heavy dependencies.
For torch-dependent functionality, see subpackages under `invarlock.*`:
- `invarlock.adapters`: Model adapters (HF causal/MLM/seq2seq + auto)
- `invarlock.guards`: Guard mechanisms (invariants, spectral, RMT, variance)
- `invarlock.edits`: Built-in quantization and edit interfaces
- `invarlock.eval`: Metrics, guard-overhead checks, and evaluation reporting
"""

__version__ = "0.12.1"

from dataclasses import dataclass

# Core exports - torch-independent


@dataclass
class Defaults:
    """Global default parameters for InvarLock framework helpers."""

    fft_energy_keep: float = 0.95
    mi_info_keep: float = 0.90
    koopman_margin: float = 1.05
    mp_alpha: float = 1.5
    target_param_keep: float = 0.70
    seed: int = 42
    ve_min_gain: float = 0.30


CFG = Defaults()


def get_default_config() -> Defaults:
    """Return a fresh default configuration value object."""
    return Defaults()


__all__ = ["__version__", "get_default_config", "Defaults", "CFG"]
