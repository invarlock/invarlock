"""Snapshot-reuse runtime contracts for config-driven execution."""

from __future__ import annotations

from typing import Any


class SnapshotRestoreFailed(RuntimeError):
    """Signal that a requested snapshot could not be restored safely."""


def require_snapshot_reuse_model(*, model: Any, phase: str) -> Any:
    """Return the reusable model or fail when snapshot reuse has no model."""

    if model is None:
        raise SnapshotRestoreFailed(
            f"Snapshot reuse requested for {phase} without a live model instance."
        )
    return model


__all__ = ["SnapshotRestoreFailed", "require_snapshot_reuse_model"]
