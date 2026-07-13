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
from invarlock.runtime_verify import verify_runtime_manifest_snapshot


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
    binding_verified: bool = False
    expected_digest_matched: bool = False
    trust_status: str = "failed"
    declared_image_digest: str | None = None


@dataclass(frozen=True)
class RuntimeProvenanceVerdict:
    declared_mode: str
    status: str
    verified: bool
    skipped: bool
    issues: tuple[RuntimeProvenanceIssue, ...] = ()
    binding_verified: bool = False
    expected_digest_matched: bool = False
    trust_status: str = "failed"
    declared_image_digest: str | None = None

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
        expected_digest_matched = bool(
            result.expected_digest_matched or result.verified
        )
        trust_status = result.trust_status
        if expected_digest_matched and trust_status == "failed":
            trust_status = "expected_image_digest_matched"
        if expected_digest_matched:
            status = "expected_image_digest_matched"
        elif result.binding_verified:
            status = "manifest_bound"
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
            binding_verified=result.binding_verified,
            expected_digest_matched=expected_digest_matched,
            trust_status=trust_status,
            declared_image_digest=result.declared_image_digest,
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
                "binding_verified": self.binding_verified,
                "expected_digest_matched": self.expected_digest_matched,
                "trust_status": self.trust_status,
                "declared_image_digest": self.declared_image_digest,
                "skipped": self.skipped,
                "strict_blocking": self.strict_blocking,
                "issues": issues,
            }
        }


def _runtime_verifier_failed_result(
    report: Path,
    *,
    messages: tuple[str, ...],
    binding_verified: bool = False,
    declared_image_digest: str | None = None,
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
        binding_verified=binding_verified,
        expected_digest_matched=False,
        trust_status="failed",
        declared_image_digest=declared_image_digest,
    )


def _verify_runtime_manifest(
    report: Path,
    manifest_path: Path,
    *,
    report_bytes: bytes,
    manifest_payload: dict[str, object],
    expected_image_digest: str | None = None,
    require_strict_runtime: bool = False,
) -> RuntimeProvenanceResult:
    result = verify_runtime_manifest_snapshot(
        report_bytes,
        manifest_payload,
        report=report,
        manifest=manifest_path,
        expected_image_digest=expected_image_digest,
        require_strict_runtime=require_strict_runtime,
    )
    binding_verified = bool(getattr(result, "binding_verified", result.ok))
    expected_digest_matched = bool(getattr(result, "expected_digest_matched", False))
    trust_status = str(
        getattr(
            result,
            "trust_status",
            (
                "expected_image_digest_matched"
                if expected_digest_matched
                else "manifest_bound"
            ),
        )
    )
    declared_image_digest = getattr(result, "declared_image_digest", None)
    if result.ok:
        return RuntimeProvenanceResult(
            verified=expected_digest_matched,
            skipped=False,
            binding_verified=binding_verified,
            expected_digest_matched=expected_digest_matched,
            trust_status=trust_status,
            declared_image_digest=declared_image_digest,
        )
    messages = result.errors or (f"Runtime verifier failed for {report.name}.",)
    return _runtime_verifier_failed_result(
        report,
        messages=messages,
        binding_verified=binding_verified,
        declared_image_digest=declared_image_digest,
    )


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
    expected_image_digest: str | None = None,
    report_bytes: bytes | None = None,
    require_strict_runtime: bool = False,
) -> RuntimeProvenanceResult:
    if allow_unverified or unverified_provenance_allowed():
        return RuntimeProvenanceResult(
            verified=False,
            skipped=True,
            trust_status="skipped",
        )

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

    if report_bytes is None:
        try:
            report_bytes = report.read_bytes()
        except OSError as exc:
            return _runtime_verifier_failed_result(
                report,
                messages=(f"unable to read report: {exc}",),
            )

    return _verify_runtime_manifest(
        report,
        manifest_path,
        report_bytes=report_bytes,
        manifest_payload=manifest,
        expected_image_digest=expected_image_digest,
        require_strict_runtime=require_strict_runtime,
    )


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
