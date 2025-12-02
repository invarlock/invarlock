"""Evaluation utilities (`invarlock.eval`).

This package now focuses on evaluation helpers (metrics, data, probes).
Reporting (report building, certificate, schema validation) has moved to
`invarlock.reporting`.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = []


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy loader
    # Provide a minimal lazy import hook so tests that reference
    # `invarlock.eval.primary_metric` via string paths (e.g., monkeypatch)
    # can resolve the module without importing it eagerly at package import.
    if name == "primary_metric":
        import importlib

        return importlib.import_module(".primary_metric", __name__)
    raise AttributeError(name)
