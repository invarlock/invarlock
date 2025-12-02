"""
InvarLock Core API
==============

Core interfaces and data structures - torch-free abstractions.

This module defines the fundamental abstractions used throughout InvarLock:
- ModelAdapter: Model-specific operations interface
- ModelEdit: Edit operation interface
- Guard: Safety validation interface
- RunConfig: Pipeline configuration
- RunReport: Execution results
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ModelAdapter(ABC):
    """
    Abstract interface for model-specific operations.

    Adapters provide model-agnostic access to different architectures
    (HuggingFace, custom models, etc.) without requiring torch imports
    in the core API.
    """

    name: str

    @abstractmethod
    def can_handle(self, model: Any) -> bool:
        """Check if this adapter can handle the given model."""
        pass

    @abstractmethod
    def describe(self, model: Any) -> dict[str, Any]:
        """
        Get model structure description.

        Returns:
            Dict with keys: n_layer, heads_per_layer, mlp_dims, tying
        """
        pass

    @abstractmethod
    def snapshot(self, model: Any) -> bytes:
        """Create serialized snapshot of model state."""
        pass

    @abstractmethod
    def restore(self, model: Any, blob: bytes) -> None:
        """Restore model state from snapshot."""
        pass


class ModelEdit(ABC):
    """
    Abstract interface for model editing operations.

    Edits modify model parameters/structure while maintaining
    the model's interface and basic functionality.
    """

    name: str

    @abstractmethod
    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        """Check if this edit can be applied to the described model."""
        pass

    @abstractmethod
    def apply(self, model: Any, adapter: ModelAdapter, **kwargs) -> dict[str, Any]:
        """
        Apply the edit to the model.

        Args:
            model: The model to edit
            adapter: Adapter for model-specific operations
            **kwargs: Edit-specific parameters

        Returns:
            Dict with edit metadata and statistics
        """
        pass


class Guard(ABC):
    """
    Abstract interface for safety guards.

    Guards validate model state and behavior to ensure
    edits don't cause unacceptable degradation.
    """

    name: str

    @abstractmethod
    def validate(
        self, model: Any, adapter: ModelAdapter, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate model state/behavior.

        Args:
            model: The model to validate
            adapter: Adapter for model operations
            context: Validation context (baseline metrics, etc.)

        Returns:
            Dict with validation results and status
        """
        pass


class GuardChain:
    """
    Manages a chain of guards with policy-based execution.

    Provides lifecycle management (prepare, before_edit, after_edit, finalize)
    and aggregates results across multiple guards.
    """

    def __init__(self, guards: list[Guard], policy: Any | None = None):
        """
        Initialize guard chain.

        Args:
            guards: List of guard instances
            policy: Policy configuration for guard execution
        """
        self.guards = guards
        self.policy = policy

    def prepare_all(
        self,
        model: Any,
        adapter: ModelAdapter,
        calib: Any,
        policy_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare all guards."""
        results = {}
        for guard in self.guards:
            if hasattr(guard, "prepare"):
                results[guard.name] = guard.prepare(
                    model, adapter, calib, policy_config
                )
            else:
                results[guard.name] = {"ready": True}
        return results

    def before_edit_all(self, model: Any) -> list[Any]:
        """Execute before_edit on all guards."""
        results = []
        for guard in self.guards:
            if hasattr(guard, "before_edit"):
                result = guard.before_edit(model)
                if result is not None:
                    results.append(result)
        return results

    def after_edit_all(self, model: Any) -> list[Any]:
        """Execute after_edit on all guards."""
        results = []
        for guard in self.guards:
            if hasattr(guard, "after_edit"):
                result = guard.after_edit(model)
                if result is not None:
                    results.append(result)
        return results

    def finalize_all(self, model: Any) -> list[Any]:
        """Finalize all guards and return outcomes."""
        results = []
        for guard in self.guards:
            if hasattr(guard, "finalize"):
                result = guard.finalize(model)
                results.append(result)
        return results

    def all_passed(self, outcomes: list[Any]) -> bool:
        """Check if all guard outcomes passed."""
        for outcome in outcomes:
            if hasattr(outcome, "passed") and not outcome.passed:
                return False
        return True

    def get_worst_action(self, outcomes: list[Any]) -> str:
        """Get the worst action from all outcomes."""
        actions = []
        for outcome in outcomes:
            if hasattr(outcome, "action"):
                actions.append(outcome.action)

        # Import from types to avoid circular dependency
        from .types import get_worst_action

        return get_worst_action(actions)


# Type alias for calibration data
CalibrationData = Any


@dataclass
class RunConfig:
    """
    Configuration for a InvarLock pipeline run.

    Contains all parameters needed to execute the full pipeline:
    prepare → edit → guards → eval → finalize/rollback.
    """

    # Device configuration
    device: str = "auto"

    # Safety thresholds
    max_pm_ratio: float = 1.5
    spike_threshold: float = 2.0  # Catastrophic spike threshold for immediate rollback

    # Output configuration
    event_path: Path | None = None
    checkpoint_interval: int = 0  # 0 = disabled

    # Execution flags
    dry_run: bool = False
    verbose: bool = False

    # Context for guards/eval
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunReport:
    """
    Results from a InvarLock pipeline execution.

    Contains comprehensive information about what was executed
    and the outcomes, suitable for analysis and certification.
    """

    # Execution metadata
    meta: dict[str, Any] = field(default_factory=dict)

    # Edit information
    edit: dict[str, Any] = field(default_factory=dict)

    # Guard validation results
    guards: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Evaluation metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Captured evaluation windows and auxiliary data
    evaluation_windows: dict[str, Any] = field(default_factory=dict)

    # Execution status
    status: str = "pending"  # pending, success, failed, rollback

    # Error information (if any)
    error: str | None = None

    # Additional context
    context: dict[str, Any] = field(default_factory=dict)


# Type aliases for common patterns
ModelType = Any  # Actual model type (torch.nn.Module, etc.)
DeviceType = str | Any  # Device specification
MetricsDict = dict[str, float | int | str | bool]
