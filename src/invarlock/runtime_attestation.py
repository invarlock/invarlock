from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from invarlock.runtime_security import (
    apply_runtime_allowances,
    load_runtime_manifest,
    runtime_verifier_binary,
    unattested_artifacts_allowed,
)

_RUNTIME_VERIFIER_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class _RuntimeVerifierResult:
    returncode: int
    stdout: str
    stderr: str


class RuntimeAttestationIssueCode(str, Enum):
    MANIFEST_MISSING = "manifest_missing"
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


def configure_runtime_security(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unattested_artifacts: bool = False,
) -> None:
    apply_runtime_allowances(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unattested_artifacts=allow_unattested_artifacts,
    )


def verify_runtime_attestation(
    report_path: str | Path,
    *,
    allow_unattested: bool = False,
) -> RuntimeAttestationResult:
    if allow_unattested or unattested_artifacts_allowed():
        return RuntimeAttestationResult(verified=False, skipped=True)

    report = Path(report_path)
    manifest_path, manifest = load_runtime_manifest(report)
    if manifest is None:
        return RuntimeAttestationResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeAttestationIssue(
                    code=RuntimeAttestationIssueCode.MANIFEST_MISSING,
                    message=(
                        f"{manifest_path.name} missing or unreadable for {report.name}."
                    ),
                    details={
                        "report": report.name,
                        "manifest": manifest_path.name,
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
        return RuntimeAttestationResult(
            verified=False,
            skipped=False,
            issues=(
                RuntimeAttestationIssue(
                    code=RuntimeAttestationIssueCode.VERIFIER_UNAVAILABLE,
                    message=(
                        f"Runtime verifier '{binary}' is not installed; "
                        f"cannot verify {report.name}."
                    ),
                    details={"report": report.name, "verifier": binary},
                ),
            ),
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
        except Exception:
            pass
        else:
            errors = payload.get("errors")
            if isinstance(errors, list) and errors:
                return RuntimeAttestationResult(
                    verified=False,
                    skipped=False,
                    issues=tuple(
                        RuntimeAttestationIssue(
                            code=RuntimeAttestationIssueCode.VERIFIER_FAILED,
                            message=str(item),
                            details={"report": report.name, "verifier": binary},
                        )
                        for item in errors
                    ),
                )
    return RuntimeAttestationResult(
        verified=False,
        skipped=False,
        issues=(
            RuntimeAttestationIssue(
                code=RuntimeAttestationIssueCode.VERIFIER_FAILED,
                message=message or f"Runtime verifier failed for {report.name}.",
                details={"report": report.name, "verifier": binary},
            ),
        ),
    )


__all__ = [
    "RuntimeAttestationIssue",
    "RuntimeAttestationIssueCode",
    "RuntimeAttestationResult",
    "configure_runtime_security",
    "verify_runtime_attestation",
]
