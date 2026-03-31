from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from invarlock.runtime_security import (
    RuntimeManifestLoadIssueCode,
    RuntimeManifestLoadResult,
    apply_runtime_allowances,
    build_runtime_security_policy,
    load_runtime_manifest,
    reset_runtime_allowances,
    runtime_verifier_binary,
    unattested_artifacts_allowed,
)
from invarlock.runtime_verify import verify_runtime_manifest

_RUNTIME_VERIFIER_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class _RuntimeVerifierResult:
    returncode: int
    stdout: str
    stderr: str


class RuntimeAttestationIssueCode(str, Enum):
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_INVALID = "manifest_invalid"
    EXECUTION_MODE_INVALID = "execution_mode_invalid"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    VERIFIER_FAILED = "verifier_failed"


@dataclass(frozen=True)
class RuntimeAttestationIssue:
    code: RuntimeAttestationIssueCode
    message: str
    details: dict[str, str] | None = None


@dataclass(frozen=True)
class RuntimeAttestationResult:
    verified: bool
    skipped: bool
    issues: tuple[RuntimeAttestationIssue, ...] = ()


def _run_runtime_verifier(
    report: Path,
    manifest_path: Path,
) -> _RuntimeVerifierResult:
    binary = runtime_verifier_binary()
    completed = subprocess.run(
        [
            binary,
            "--report",
            str(report),
            "--manifest",
            str(manifest_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=_RUNTIME_VERIFIER_TIMEOUT_SECONDS,
    )
    return _RuntimeVerifierResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _runtime_verifier_failed_result(
    report: Path,
    *,
    binary: str,
    messages: tuple[str, ...],
) -> RuntimeAttestationResult:
    return RuntimeAttestationResult(
        verified=False,
        skipped=False,
        issues=tuple(
            RuntimeAttestationIssue(
                code=RuntimeAttestationIssueCode.VERIFIER_FAILED,
                message=message,
                details={"report": report.name, "verifier": binary},
            )
            for message in messages
        ),
    )


def _verify_runtime_manifest_in_process(
    report: Path,
    manifest_path: Path,
    *,
    binary: str,
) -> RuntimeAttestationResult:
    result = verify_runtime_manifest(report, manifest_path)
    if result.ok:
        return RuntimeAttestationResult(verified=True, skipped=False)
    messages = result.errors or (f"Runtime verifier failed for {report.name}.",)
    return _runtime_verifier_failed_result(report, binary=binary, messages=messages)


@contextmanager
def configure_runtime_security(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unattested_artifacts: bool = False,
) -> Iterator[None]:
    policy = build_runtime_security_policy(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unattested_artifacts=allow_unattested_artifacts,
    )
    token = apply_runtime_allowances(policy=policy)
    try:
        yield
    finally:
        reset_runtime_allowances(token)


def verify_runtime_attestation(
    report_path: str | Path,
    *,
    allow_unattested: bool = False,
) -> RuntimeAttestationResult:
    if allow_unattested or unattested_artifacts_allowed():
        return RuntimeAttestationResult(verified=False, skipped=True)

    report = Path(report_path)
    load_result = load_runtime_manifest(report)
    manifest_path = load_result.path
    manifest = load_result.payload
    if manifest is None:
        if load_result.issue_code == RuntimeManifestLoadIssueCode.MISSING:
            return RuntimeAttestationResult(
                verified=False,
                skipped=False,
                issues=(
                    RuntimeAttestationIssue(
                        code=RuntimeAttestationIssueCode.MANIFEST_MISSING,
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
        return RuntimeAttestationResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeAttestationIssue(
                    code=RuntimeAttestationIssueCode.MANIFEST_INVALID,
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
        return RuntimeAttestationResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeAttestationIssue(
                    code=RuntimeAttestationIssueCode.EXECUTION_MODE_INVALID,
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

    binary = runtime_verifier_binary()
    if shutil.which(binary) is None:
        return _verify_runtime_manifest_in_process(
            report,
            manifest_path,
            binary=binary,
        )

    try:
        completed = _run_runtime_verifier(report, manifest_path)
    except subprocess.TimeoutExpired:
        return RuntimeAttestationResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeAttestationIssue(
                    code=RuntimeAttestationIssueCode.VERIFIER_FAILED,
                    message=f"Runtime verifier timed out for {report.name}.",
                    details={"report": report.name, "verifier": binary},
                ),
            ),
        )
    if completed.returncode == 0:
        return RuntimeAttestationResult(verified=True, skipped=False)

    message = (completed.stdout or completed.stderr or "").strip()
    if message:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            pass
        else:
            errors = payload.get("errors")
            if isinstance(errors, list) and errors:
                return _runtime_verifier_failed_result(
                    report,
                    binary=binary,
                    messages=tuple(str(item) for item in errors),
                )
    return _runtime_verifier_failed_result(
        report,
        binary=binary,
        messages=(message or f"Runtime verifier failed for {report.name}.",),
    )


__all__ = [
    "RuntimeManifestLoadIssueCode",
    "RuntimeManifestLoadResult",
    "RuntimeAttestationIssue",
    "RuntimeAttestationIssueCode",
    "RuntimeAttestationResult",
    "configure_runtime_security",
    "verify_runtime_attestation",
]
