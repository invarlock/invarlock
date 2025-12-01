"""Reference decision kernels for guards (pure, side-effect-free).

Exposes small math-first helpers used by property/differential tests.
"""

from .rmt_ref import rmt_decide
from .spectral_ref import bh_select, spectral_decide
from .variance_ref import variance_decide

__all__ = [
    "bh_select",
    "spectral_decide",
    "rmt_decide",
    "variance_decide",
]
