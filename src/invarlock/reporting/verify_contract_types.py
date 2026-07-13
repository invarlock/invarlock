"""Typed inputs and outputs for report verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from invarlock.core.assurance_contract import normalize_verify_assurance_mode

_RECOVERABLE_VALUE_EXCEPTIONS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def resolve_tolerance(tolerance: float) -> float:
    try:
        resolved = float(tolerance)
    except _RECOVERABLE_VALUE_EXCEPTIONS as exc:
        raise ValueError(
            "tolerance must be a finite number between 0 and 1e-9"
        ) from exc
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1e-9:
        raise ValueError("tolerance must be a finite number between 0 and 1e-9")
    return resolved


def normalize_warning_policy(value: str | None) -> str:
    policy = str(value or "pass").strip().lower()
    if policy in {"pass", "fail"}:
        return policy
    raise ValueError("warning_policy must be one of: pass, fail.")


@dataclass(frozen=True)
class VerifyDiagnostic:
    level: str
    message: str


class VerifyOutcome(StrEnum):
    OK = "ok"
    POLICY_FAIL = "policy_fail"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class VerifyExecutionResult:
    outcome: VerifyOutcome
    payload: Any
    diagnostics: tuple[VerifyDiagnostic, ...]
    error: Exception | None = None
    include_resolution: bool = False


@dataclass(frozen=True)
class VerifyRequest:
    reports: tuple[Path, ...]
    baseline: Path | None = None
    policy_pack: Path | None = None
    tolerance: float = 1e-9
    profile: str | None = None
    allow_unverified_provenance: bool = False
    json_mode: bool = False
    assurance_mode: str = "report"
    warning_policy: str = "pass"
    expected_runtime_image_digest: str | None = None

    @classmethod
    def from_args(
        cls,
        reports: list[Path],
        *,
        baseline: Path | None = None,
        policy_pack: Path | None = None,
        tolerance: float = 1e-9,
        profile: str | None = None,
        allow_unverified_provenance: bool = False,
        json_mode: bool = False,
        assurance_mode: str = "report",
        warning_policy: str = "pass",
        expected_runtime_image_digest: str | None = None,
    ) -> VerifyRequest:
        return cls(
            reports=tuple(reports),
            baseline=baseline,
            policy_pack=policy_pack,
            tolerance=tolerance,
            profile=profile,
            allow_unverified_provenance=allow_unverified_provenance,
            json_mode=json_mode,
            assurance_mode=assurance_mode,
            warning_policy=warning_policy,
            expected_runtime_image_digest=expected_runtime_image_digest,
        )

    @property
    def normalized_tolerance(self) -> float:
        return resolve_tolerance(self.tolerance)

    @property
    def normalized_assurance_mode(self) -> str:
        return normalize_verify_assurance_mode(self.assurance_mode)

    @property
    def normalized_warning_policy(self) -> str:
        return normalize_warning_policy(self.warning_policy)


@dataclass(frozen=True)
class VerifyReportResult:
    report: dict[str, Any]
    errors: tuple[str, ...]
    malformed: bool
    diagnostics: tuple[VerifyDiagnostic, ...]
    verification: dict[str, Any]


__all__ = [
    "VerifyDiagnostic",
    "VerifyExecutionResult",
    "VerifyOutcome",
    "VerifyReportResult",
    "VerifyRequest",
    "normalize_warning_policy",
    "resolve_tolerance",
]
