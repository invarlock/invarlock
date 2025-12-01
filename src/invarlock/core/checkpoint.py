"""
InvarLock Core Checkpoint System
===========================

Checkpoint and rollback functionality for safe model editing.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from typing import Any

from .types import GuardOutcome


def _use_chunked_snapshot() -> bool:
    """Return True when chunked snapshot mode is enabled."""
    return os.environ.get("INVARLOCK_SNAPSHOT_MODE", "bytes").lower() == "chunked"


class PolicyCheckpoint:
    """
    Checkpoint manager for policy-based rollback decisions.
    """

    def __init__(self, model: Any, adapter: Any, policy: Any):
        """
        Initialize checkpoint.

        Args:
            model: Model to checkpoint
            adapter: ModelAdapter for model operations
            policy: Policy configuration
        """
        self.model = model
        self.adapter = adapter
        self.policy = policy
        self.checkpoint_data: dict[str, Any] | None = None
        self.rollback_performed = False

    def create_checkpoint(self) -> None:
        """Create a checkpoint of the current model state."""
        if _use_chunked_snapshot() and hasattr(self.adapter, "snapshot_chunked"):
            snapshot_path = self.adapter.snapshot_chunked(self.model)
            self.checkpoint_data = {"mode": "chunked", "path": snapshot_path}
        else:
            self.checkpoint_data = {
                "mode": "bytes",
                "blob": self.adapter.snapshot(self.model),
            }

    def should_rollback(self, outcomes: list[GuardOutcome]) -> tuple[bool, str]:
        """
        Determine if rollback should be performed based on guard outcomes.

        Args:
            outcomes: List of guard outcomes

        Returns:
            (should_rollback, reason) tuple
        """
        # Check for abort actions
        for outcome in outcomes:
            if hasattr(outcome, "action") and outcome.action == "abort":
                return True, "guard_abort"

        # Check for rollback actions
        for outcome in outcomes:
            if hasattr(outcome, "action") and outcome.action == "rollback":
                return True, "guard_rollback"

        # Check policy configuration
        if (
            hasattr(self.policy, "enable_auto_rollback")
            and self.policy.enable_auto_rollback
        ):
            # Check if any guards failed
            for outcome in outcomes:
                if hasattr(outcome, "passed") and not outcome.passed:
                    return True, "auto_rollback"

        return False, ""

    def rollback(self, reason: str) -> bool:
        """
        Perform rollback to checkpoint.

        Args:
            reason: Reason for rollback

        Returns:
            True if rollback was successful
        """
        if self.checkpoint_data is None:
            return False
        try:
            mode = self.checkpoint_data.get("mode", "bytes")
            if mode == "chunked":
                path = self.checkpoint_data.get("path")
                if not path or not hasattr(self.adapter, "restore_chunked"):
                    return False
                self.adapter.restore_chunked(self.model, path)
            else:
                blob = self.checkpoint_data.get("blob")
                self.adapter.restore(self.model, blob)
            self.rollback_performed = True
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        """Clean up checkpoint resources."""
        if self.checkpoint_data and self.checkpoint_data.get("mode") == "chunked":
            path = self.checkpoint_data.get("path")
            if path and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        self.checkpoint_data = None


@contextmanager
def create_policy_checkpoint(model: Any, adapter: Any, policy: Any):
    """
    Context manager for policy-based checkpointing.

    Args:
        model: Model to checkpoint
        adapter: ModelAdapter for operations
        policy: Policy configuration

    Yields:
        PolicyCheckpoint instance
    """
    checkpoint = PolicyCheckpoint(model, adapter, policy)
    checkpoint.create_checkpoint()

    try:
        yield checkpoint
    finally:
        checkpoint.cleanup()


class CheckpointManager:
    """
    Manager for model checkpoints during pipeline execution.
    """

    def __init__(self):
        """Initialize checkpoint manager."""
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.next_id = 1

    def create_checkpoint(self, model: Any, adapter: Any) -> str:
        """
        Create a checkpoint of the model.

        Args:
            model: Model to checkpoint
            adapter: ModelAdapter for serialization

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"checkpoint_{self.next_id}"
        self.next_id += 1

        try:
            if _use_chunked_snapshot() and hasattr(adapter, "snapshot_chunked"):
                snapshot_path = adapter.snapshot_chunked(model)
                checkpoint_data = {"mode": "chunked", "path": snapshot_path}
            else:
                checkpoint_data = {
                    "mode": "bytes",
                    "blob": adapter.snapshot(model),
                }
            self.checkpoints[checkpoint_id] = checkpoint_data
            return checkpoint_id
        except Exception as e:
            raise RuntimeError(f"Failed to create checkpoint: {e}") from e

    def restore_checkpoint(self, model: Any, adapter: Any, checkpoint_id: str) -> bool:
        """
        Restore model from checkpoint.

        Args:
            model: Model to restore
            adapter: ModelAdapter for deserialization
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restoration was successful
        """
        if checkpoint_id not in self.checkpoints:
            return False

        try:
            checkpoint_data = self.checkpoints[checkpoint_id]
            mode = checkpoint_data.get("mode", "bytes")
            if mode == "chunked":
                if not hasattr(adapter, "restore_chunked"):
                    return False
                adapter.restore_chunked(model, checkpoint_data.get("path"))
            else:
                adapter.restore(model, checkpoint_data.get("blob"))
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        """Clean up all checkpoints."""
        for data in self.checkpoints.values():
            if data.get("mode") == "chunked":
                path = data.get("path")
                if path and os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
        self.checkpoints.clear()
        self.next_id = 1


__all__ = ["PolicyCheckpoint", "create_policy_checkpoint", "CheckpointManager"]
