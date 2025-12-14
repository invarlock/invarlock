"""
InvarLock: Edit‑agnostic robustness certificates for weight edits
=============================================================

Core runtime package — torch-independent utilities, configuration, and interfaces.

This package provides the foundation for the InvarLock GuardChain without heavy dependencies.
For torch-dependent functionality, see subpackages under `invarlock.*`:
- `invarlock.adapters`: Model adapters (HF GPT-2/BERT/LLaMA, auto)
- `invarlock.guards`: Safety mechanisms (invariants, spectral, RMT, variance)
- `invarlock.edits`: Built-in quantization and edit interfaces
- `invarlock.eval`: Metrics, guard-overhead checks, and certification
"""

__version__ = "0.3.2"

# Core exports - torch-independent
from .config import CFG, Defaults, get_default_config

__all__ = ["__version__", "get_default_config", "Defaults", "CFG"]


def __getattr__(name: str):  # pragma: no cover - thin lazy loader
    """Lazily expose selected subpackages without importing heavy deps at init.

    This keeps `import invarlock` torch-free while allowing patterns like
    monkeypatching `invarlock.eval.*` in tests.
    """
    if name == "eval":
        import importlib

        return importlib.import_module(".eval", __name__)
    raise AttributeError(name)
