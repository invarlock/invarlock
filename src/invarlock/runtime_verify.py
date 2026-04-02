from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from invarlock.core.runtime_manifest_verify import (
    verify_report_manifest as _verify_report_manifest,
)


@dataclass(frozen=True)
class RuntimeVerifyResult:
    ok: bool
    errors: tuple[str, ...]
    report: str
    manifest: str


def verify_runtime_manifest(
    report: str | Path,
    manifest: str | Path,
) -> RuntimeVerifyResult:
    report_path = Path(report)
    manifest_path = Path(manifest)
    errors = tuple(_verify_report_manifest(report_path, manifest_path))
    return RuntimeVerifyResult(
        ok=not errors,
        errors=errors,
        report=str(report_path),
        manifest=str(manifest_path),
    )


__all__ = ["RuntimeVerifyResult", "verify_runtime_manifest"]
