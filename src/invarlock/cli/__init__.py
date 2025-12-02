"""CLI namespace wrapper for unified import path (`invarlock.cli`)."""

from __future__ import annotations

# Re-export common entry points for convenience
from .app import app

__all__ = ["app"]
