"""
InvarLock Core Types
================

Core type definitions and enums used throughout InvarLock.
Torch-independent type system for cross-module compatibility.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

_ACTION_PRIORITY = {"none": 0, "warn": 1, "rollback": 2, "abort": 3}
_DECISION_PRIORITY = {"allow": 0, "monitor": 1, "rollback": 2, "block": 3}
_ACTION_TO_DECISION = {
    "none": "allow",
    "warn": "monitor",
    "rollback": "rollback",
    "abort": "block",
    "reject": "block",
}
_DECISION_TO_ACTION = {
    "allow": "none",
    "monitor": "warn",
    "rollback": "rollback",
    "block": "abort",
}


class EditType(Enum):
    """Types of model edits supported by InvarLock."""

    QUANTIZATION = "quantization"
    SPARSITY = "sparsity"
    MIXED = "mixed"


class GuardType(Enum):
    """Types of safety guards available."""

    INVARIANTS = "invariants"
    SPECTRAL = "spectral"
    VARIANCE = "variance"
    RMT = "rmt"
    NOOP = "noop"


class RunStatus(Enum):
    """Execution status for pipeline runs."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"
    CANCELLED = "cancelled"


class LogLevel(Enum):
    """Logging levels for events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelInfo:
    """Basic model information."""

    model_id: str
    architecture: str
    parameters: int
    device: str
    precision: str = "float32"


@dataclass
class EditInfo:
    """Information about an applied edit."""

    name: str
    type: EditType
    parameters: dict[str, Any]
    compression_ratio: float | None = None
    target_metrics: dict[str, float] | None = None


@dataclass
class GuardResult:
    """Result from a guard validation."""

    guard_name: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class GuardDiagnostic:
    """Neutral diagnostic emitted by reusable guard code."""

    kind: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class GuardValidationResult(dict[str, Any]):
    """Typed validation contract emitted by public guard interfaces."""

    def __init__(
        self,
        *,
        passed: bool,
        decision: str,
        metrics: dict[str, Any] | None = None,
        diagnostics: tuple[GuardDiagnostic, ...] | list[GuardDiagnostic] = (),
        policy: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        violations: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        extras: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "passed": bool(passed),
            "decision": str(decision),
            "action": decision_to_action(str(decision)),
            "metrics": dict(metrics or {}),
            "diagnostics": [
                {
                    "kind": diagnostic.kind,
                    "severity": diagnostic.severity,
                    "message": diagnostic.message,
                    "details": dict(diagnostic.details),
                }
                for diagnostic in diagnostics
            ],
            "policy": dict(policy or {}),
            "details": dict(details or {}),
            "violations": [dict(item) for item in violations],
        }
        if extras:
            payload.update(dict(extras))
        super().__init__(payload)

    @property
    def passed(self) -> bool:
        return bool(self.get("passed", False))

    @property
    def decision(self) -> str:
        return str(self.get("decision", "allow"))

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.get("metrics", {}))

    @property
    def action(self) -> str:
        raw = self.get("action")
        if isinstance(raw, str) and raw:
            return raw
        return decision_to_action(self.decision)

    @property
    def diagnostics(self) -> tuple[GuardDiagnostic, ...]:
        diagnostics = self.get("diagnostics", [])
        records: list[GuardDiagnostic] = []
        if isinstance(diagnostics, list):
            for item in diagnostics:
                if not isinstance(item, dict):
                    continue
                records.append(
                    GuardDiagnostic(
                        kind=str(item.get("kind", "guard_diagnostic")),
                        severity=str(item.get("severity", "info")),
                        message=str(item.get("message", "")),
                        details=dict(item.get("details", {})),
                    )
                )
        return tuple(records)

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self.get("policy", {}))

    @property
    def details(self) -> dict[str, Any]:
        return dict(self.get("details", {}))

    @property
    def violations(self) -> tuple[dict[str, Any], ...]:
        raw = self.get("violations", [])
        if not isinstance(raw, list):
            return ()
        return tuple(dict(item) for item in raw if isinstance(item, dict))

    @property
    def extras(self) -> dict[str, Any]:
        extras = dict(self)
        for key in (
            "passed",
            "decision",
            "action",
            "metrics",
            "diagnostics",
            "policy",
            "details",
            "violations",
        ):
            extras.pop(key, None)
        return extras


class ValidationResult(NamedTuple):
    """Result from validation operations."""

    passed: bool
    score: float
    threshold: float
    message: str = ""


@dataclass
class GuardOutcome:
    """Result from a guard execution."""

    name: str
    passed: bool
    decision: str = ""
    action: str = ""
    violations: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.metrics is None:
            self.metrics = {}
        if not self.decision:
            self.decision = normalize_guard_decision(
                fallback_action=self.action or None,
                passed=self.passed,
            )
        if not self.action:
            self.action = decision_to_action(self.decision)


@dataclass
class PolicyConfig:
    """Configuration for guard policies."""

    on_violation: str = "warn"
    guard_overrides: dict[str, str] | None = None
    enable_auto_rollback: bool = False

    def __post_init__(self):
        if self.guard_overrides is None:
            self.guard_overrides = {}

    def get_decision_for_guard(
        self,
        guard_name: str,
        requested_decision: str,
    ) -> str:
        """Get the typed decision for a specific guard."""
        if self.guard_overrides and guard_name in self.guard_overrides:
            return normalize_guard_decision(
                self.guard_overrides[guard_name],
                fallback_action=self.guard_overrides[guard_name],
                passed=False,
            )

        normalized_requested = normalize_guard_decision(
            requested_decision,
            fallback_action=requested_decision,
            passed=False,
        )
        if normalized_requested != "allow":
            return normalized_requested

        return normalize_guard_decision(
            self.on_violation,
            fallback_action=self.on_violation,
            passed=False,
        )

    def get_action_for_guard(self, guard_name: str, requested_action: str) -> str:
        """Get the action for a specific guard."""
        decision = self.get_decision_for_guard(guard_name, requested_action)
        return decision_to_action(decision)


def get_worst_action(actions: list[str]) -> str:
    """Get the worst (most severe) action from a list."""
    if not actions:
        return "none"

    return max(actions, key=lambda action: _ACTION_PRIORITY.get(action, 0))


def normalize_guard_decision(
    decision: str | None = None,
    *,
    fallback_action: str | None = None,
    passed: bool | None = None,
) -> str:
    """Normalize mixed action/decision inputs onto the typed guard-decision model."""
    if isinstance(decision, str):
        normalized = decision.strip().lower()
        if normalized:
            if normalized in _DECISION_PRIORITY:
                return normalized
            mapped = _ACTION_TO_DECISION.get(normalized)
            if mapped is not None:
                return mapped

    if isinstance(fallback_action, str):
        normalized_action = fallback_action.strip().lower()
        if normalized_action:
            mapped = _ACTION_TO_DECISION.get(normalized_action)
            if mapped is not None:
                return mapped

    if passed is False:
        return "block"
    return "allow"


def decision_to_action(decision: str) -> str:
    """Convert a typed guard decision back to the legacy action vocabulary."""
    normalized = normalize_guard_decision(decision)
    return _DECISION_TO_ACTION.get(normalized, decision)


def get_worst_decision(decisions: list[str]) -> str:
    """Get the worst (most severe) typed decision from a list."""
    if not decisions:
        return "allow"
    return max(
        decisions,
        key=lambda decision: _DECISION_PRIORITY.get(
            normalize_guard_decision(decision, fallback_action=decision),
            0,
        ),
    )


# Type aliases for clarity
DeviceSpec = str | Any  # Device specification
ConfigDict = dict[str, Any]  # Configuration dictionary
MetricsDict = dict[str, float | int | str | bool]  # Metrics
LayerIndex = int  # Layer index
HeadIndex = int  # Attention head index
