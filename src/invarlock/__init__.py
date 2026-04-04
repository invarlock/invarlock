"""
InvarLock: Edit‑agnostic evaluation reports for weight edits
=============================================================

Core runtime package — torch-independent utilities, configuration, and interfaces.

This package provides the foundation for the InvarLock GuardChain without heavy dependencies.
For torch-dependent functionality, see subpackages under `invarlock.*`:
- `invarlock.adapters`: Model adapters (HF causal/MLM/seq2seq + auto)
- `invarlock.guards`: Safety mechanisms (invariants, spectral, RMT, variance)
- `invarlock.edits`: Built-in quantization and edit interfaces
- `invarlock.eval`: Metrics, guard-overhead checks, and evaluation reporting
"""

__version__ = "0.6.0"

# Core exports - torch-independent
from .config import CFG, Defaults, get_default_config

__all__ = ["__version__", "get_default_config", "Defaults", "CFG"]
