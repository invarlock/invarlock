from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from invarlock.runtime_security import (
    RuntimeManifestLoadIssueCode,
    RuntimeManifestLoadResult,
    apply_runtime_allowances,
    build_runtime_security_policy,
    load_runtime_manifest,
    reset_runtime_allowances,
    unverified_provenance_allowed,
)
from invarlock.runtime_verify import verify_runtime_manifest


class RuntimeProvenanceIssueCode(StrEnum):
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_INVALID = "manifest_invalid"
    EXECUTION_MODE_INVALID = "execution_mode_invalid"
    VERIFIER_FAILED = "verifier_failed"


@dataclass(frozen=True)
class RuntimeProvenanceIssue:
    code: RuntimeProvenanceIssueCode
    message: str
    details: dict[str, str] | None = None


@dataclass(frozen=True)
class RuntimeProvenanceResult:
    verified: bool
    skipped: bool
    issues: tuple[RuntimeProvenanceIssue, ...] = ()


@dataclass(frozen=True)
class RuntimeProvenanceVerdict:
    declared_mode: str
    status: str
    verified: bool
    skipped: bool
    issues: tuple[RuntimeProvenanceIssue, ...] = ()

    @property
    def strict_blocking(self) -> bool:
        return not self.verified

    @classmethod
    def from_result(
        cls,
        result: RuntimeProvenanceResult,
        *,
        declared_mode: str = "unknown",
    ) -> RuntimeProvenanceVerdict:
        if result.verified:
            status = "verified"
        elif result.skipped:
            status = "skipped"
        else:
            status = "failed"
        return cls(
            declared_mode=declared_mode,
            status=status,
            verified=result.verified,
            skipped=result.skipped,
            issues=result.issues,
        )

    def as_verification_payload(self) -> dict[str, object]:
        issues = []
        for issue in self.issues:
            code = issue.code
            issues.append(
                {
                    "code": code.value,
                    "message": issue.message,
                    "details": issue.details or {},
                }
            )
        return {
            "runtime_provenance": {
                "declared_mode": self.declared_mode,
                "status": self.status,
                "verified": self.verified,
                "skipped": self.skipped,
                "strict_blocking": self.strict_blocking,
                "issues": issues,
            }
        }


def _runtime_verifier_failed_result(
    report: Path,
    *,
    messages: tuple[str, ...],
) -> RuntimeProvenanceResult:
    return RuntimeProvenanceResult(
        verified=False,
        skipped=False,
        issues=tuple(
            RuntimeProvenanceIssue(
                code=RuntimeProvenanceIssueCode.VERIFIER_FAILED,
                message=message,
                details={"report": report.name},
            )
            for message in messages
        ),
    )


def _verify_runtime_manifest(
    report: Path,
    manifest_path: Path,
) -> RuntimeProvenanceResult:
    result = verify_runtime_manifest(report, manifest_path)
    if result.ok:
        return RuntimeProvenanceResult(verified=True, skipped=False)
    messages = result.errors or (f"Runtime verifier failed for {report.name}.",)
    return _runtime_verifier_failed_result(report, messages=messages)


@contextmanager
def configure_runtime_security(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> Iterator[None]:
    policy = build_runtime_security_policy(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    token = apply_runtime_allowances(policy=policy)
    try:
        yield
    finally:
        reset_runtime_allowances(token)


def verify_runtime_provenance(
    report_path: str | Path,
    *,
    allow_unverified: bool = False,
) -> RuntimeProvenanceResult:
    if allow_unverified or unverified_provenance_allowed():
        return RuntimeProvenanceResult(verified=False, skipped=True)

    report = Path(report_path)
    load_result = load_runtime_manifest(report)
    manifest_path = load_result.path
    manifest = load_result.payload
    if manifest is None:
        if load_result.issue_code == RuntimeManifestLoadIssueCode.MISSING:
            return RuntimeProvenanceResult(
                verified=False,
                skipped=False,
                issues=(
                    RuntimeProvenanceIssue(
                        code=RuntimeProvenanceIssueCode.MANIFEST_MISSING,
                        message=f"{manifest_path.name} missing for {report.name}.",
                        details={
                            "report": report.name,
                            "manifest": manifest_path.name,
                        },
                    ),
                ),
            )
        detail = (
            load_result.issue_message.rstrip(".")
            if load_result.issue_message
            else f"{manifest_path.name} is unreadable"
        )
        return RuntimeProvenanceResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeProvenanceIssue(
                    code=RuntimeProvenanceIssueCode.MANIFEST_INVALID,
                    message=f"{manifest_path.name} is invalid for {report.name}: {detail}.",
                    details={
                        "report": report.name,
                        "manifest": manifest_path.name,
                        "issue_code": (
                            load_result.issue_code.value
                            if load_result.issue_code is not None
                            else "unknown"
                        ),
                    },
                ),
            ),
        )

    if manifest.get("execution_mode") != "container":
        return RuntimeProvenanceResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeProvenanceIssue(
                    code=RuntimeProvenanceIssueCode.EXECUTION_MODE_INVALID,
                    message=(
                        f"{manifest_path.name} marks {report.name} as "
                        f"{manifest.get('execution_mode')!r}."
                    ),
                    details={
                        "report": report.name,
                        "manifest": manifest_path.name,
                        "execution_mode": str(manifest.get("execution_mode")),
                    },
                ),
            ),
        )

    return _verify_runtime_manifest(report, manifest_path)


__all__ = [
    "RuntimeManifestLoadIssueCode",
    "RuntimeManifestLoadResult",
    "RuntimeProvenanceIssue",
    "RuntimeProvenanceIssueCode",
    "RuntimeProvenanceResult",
    "RuntimeProvenanceVerdict",
    "configure_runtime_security",
    "verify_runtime_provenance",
]
