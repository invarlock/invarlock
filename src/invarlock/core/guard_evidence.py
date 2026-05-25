"""Canonical report evidence shape for guard results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardEvidence:
    name: str
    passed: Any
    decision: str
    policy: Any
    metrics: Any
    diagnostics: Any
    violations: Any
    details: Any
    final_z_scores: Any = None
    module_family_map: Any = None
    supported: Any = None
    reason: Any = None
    assurance_blocking: Any = None
    status: Any = None

    @classmethod
    def from_result(cls, name: str, result: Any) -> GuardEvidence | None:
        if not isinstance(result, dict):
            return None
        decision = result.get("decision")
        if not isinstance(decision, str) or not decision:
            decision = "allow" if bool(result.get("passed", False)) else "block"
        return cls(
            name=name,
            passed=result.get("passed"),
            decision=decision,
            policy=result.get("policy", {}),
            metrics=result.get("metrics", {}),
            diagnostics=result.get("diagnostics", []),
            violations=result.get("violations", []),
            details=result.get("details", {}),
            final_z_scores=result.get("final_z_scores"),
            module_family_map=result.get("module_family_map"),
            supported=result.get("supported"),
            reason=result.get("reason"),
            assurance_blocking=result.get("assurance_blocking"),
            status=result.get("status"),
        )

    def as_report_entry(self) -> dict[str, Any]:
        entry = {
            "name": self.name,
            "passed": self.passed,
            "decision": self.decision,
            "policy": self.policy,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "violations": self.violations,
            "details": self.details,
        }
        for key in (
            "final_z_scores",
            "module_family_map",
            "supported",
            "reason",
            "assurance_blocking",
            "status",
        ):
            value = getattr(self, key)
            if value is not None:
                entry[key] = value
        return entry

    def strict_blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.supported is False and self.assurance_blocking is True:
            reason = self.reason or "unsupported"
            reasons.append(f"{self.name} unsupported for strict assurance: {reason}.")
        if str(self.status or "").strip().lower() in {"degraded", "monitor-only"}:
            reasons.append(
                f"{self.name} status {self.status} is not strict-assurance passing."
            )
        if self.decision == "block" or self.passed is False:
            reasons.append(f"{self.name} did not pass.")
        return tuple(reasons)


__all__ = ["GuardEvidence"]
