"""Canonical report evidence shape for guard results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_EVIDENCE_DUMP_EXCEPTIONS = (AttributeError, OSError, TypeError, ValueError)


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

    @classmethod
    def from_report_block(cls, name: str, block: Any) -> GuardEvidence | None:
        evidence = cls.from_result(name, block)
        if evidence is None:
            return None
        if (
            isinstance(block, dict)
            and "decision" not in block
            and "passed" not in block
        ):
            return replace(evidence, decision="")
        return evidence

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
        status = str(self.status or "").strip().lower()
        if self.supported is False and self.assurance_blocking is True:
            reason = self.reason or "unsupported"
            reasons.append(f"{self.name} unsupported for strict assurance: {reason}.")
        if status in {"degraded", "monitor-only", "monitor_only"}:
            reasons.append(
                f"{self.name} status {self.status} is not strict-assurance passing."
            )
        if (
            not self.decision
            and self.passed is None
            and status not in {"ok", "pass", "passed"}
            and self.supported is None
        ):
            reasons.append(f"{self.name} missing strict guard pass evidence.")
        if self.decision == "block" or self.passed is False:
            reasons.append(f"{self.name} did not pass.")
        return tuple(reasons)


def maybe_dump_guard_evidence(
    target_dir: str | Path, payload: dict[str, Any]
) -> Path | None:
    """Dump a small JSON blob of guard decision inputs when enabled."""

    if os.getenv("INVARLOCK_EVIDENCE_DEBUG", "0").strip() != "1":
        return None
    try:
        path = Path(target_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "guards_evidence.json"
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out
    except _EVIDENCE_DUMP_EXCEPTIONS:
        return None


def build_guard_evidence_payload(report: Any) -> dict[str, Any]:
    """Build the compact guard evidence payload persisted with report bundles."""
    try:
        guard_ctx = report.get("guards") or []
    except _NON_FATAL_EXCEPTIONS:
        guard_ctx = []

    if not isinstance(guard_ctx, list) or not guard_ctx:
        return {"guards_decisions": []}

    tiny: list[dict[str, object]] = []
    guard_items: list[Any] = list(guard_ctx)
    for guard in guard_items:
        if not isinstance(guard, dict):
            continue
        entry: dict[str, object] = {}
        policy = guard.get("policy") or {}
        if isinstance(policy, dict):
            for key in (
                "deadband",
                "min_effect_lognll",
                "max_caps",
                "sigma_quantile",
            ):
                if key in policy:
                    entry[key] = policy[key]
        if guard.get("name"):
            entry["name"] = guard.get("name")
        if entry:
            tiny.append(entry)
    return {"guards_decisions": tiny}


__all__ = [
    "GuardEvidence",
    "build_guard_evidence_payload",
    "maybe_dump_guard_evidence",
]
