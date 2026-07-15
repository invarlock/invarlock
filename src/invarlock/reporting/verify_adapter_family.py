"""Advisory diagnostics for cross-family adapter comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .verify_contract_types import VerifyDiagnostic

_RECOVERABLE_EXCEPTIONS = (
    AttributeError,
    FileNotFoundError,
    json.JSONDecodeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_SNAPSHOT_NOT_PROVIDED = object()


def warn_adapter_family_mismatch(
    report: dict[str, Any],
    *,
    trusted_baseline_path: Path | None = None,
    trusted_baseline_payload: dict[str, Any] | None | object = (_SNAPSHOT_NOT_PROVIDED),
) -> tuple[VerifyDiagnostic, ...]:
    """Build an advisory when baseline and edited adapter families differ."""

    try:
        plugins = report.get("plugins") or {}
        adapter_meta = plugins.get("adapter") if isinstance(plugins, dict) else None
        edited_family = None
        edited_lib = None
        edited_ver = None
        if isinstance(adapter_meta, dict):
            provenance = adapter_meta.get("provenance")
            if isinstance(provenance, dict):
                edited_family = str(provenance.get("family") or "").lower() or None
                edited_lib = provenance.get("library") or None
                edited_ver = provenance.get("version") or None

        baseline_provenance = (
            report.get("provenance")
            if isinstance(report.get("provenance"), dict)
            else {}
        )
        baseline_ref = (
            baseline_provenance.get("baseline")
            if isinstance(baseline_provenance, dict)
            else None
        )
        baseline_report_path = (
            baseline_ref.get("report_path") if isinstance(baseline_ref, dict) else None
        )

        baseline_family = None
        baseline_lib = None
        baseline_ver = None
        if isinstance(baseline_report_path, str) and baseline_report_path:
            candidate = Path(baseline_report_path)
            trusted = (
                trusted_baseline_path.resolve(strict=False)
                if isinstance(trusted_baseline_path, Path)
                else None
            )
            if trusted is not None and candidate.resolve(strict=False) == trusted:
                if trusted_baseline_payload is _SNAPSHOT_NOT_PROVIDED:
                    baseline_report = (
                        json.loads(candidate.read_bytes())
                        if candidate.is_file()
                        else None
                    )
                else:
                    baseline_report = trusted_baseline_payload
                meta = (
                    baseline_report.get("meta", {})
                    if isinstance(baseline_report, dict)
                    else {}
                )
                baseline_plugins = (
                    meta.get("plugins") if isinstance(meta, dict) else None
                )
                if isinstance(baseline_plugins, dict):
                    baseline_adapter = baseline_plugins.get("adapter")
                    if isinstance(baseline_adapter, dict):
                        provenance = baseline_adapter.get("provenance")
                        if isinstance(provenance, dict):
                            family = provenance.get("family")
                            if isinstance(family, str) and family:
                                baseline_family = family.lower()
                            baseline_lib = provenance.get("library") or None
                            baseline_ver = provenance.get("version") or None

        if edited_family and baseline_family and edited_family != baseline_family:
            baseline_backend = baseline_lib or "—"
            baseline_version = (
                f"=={baseline_ver}" if baseline_lib and baseline_ver else "—"
            )
            edited_backend = edited_lib or "—"
            edited_version = f"=={edited_ver}" if edited_lib and edited_ver else "—"
            return (
                VerifyDiagnostic(
                    level="warning",
                    message="Adapter family differs between baseline and edited runs:",
                ),
                VerifyDiagnostic(
                    level="warning",
                    message=(
                        f"baseline: family={baseline_family}, backend={baseline_backend} "
                        f"{baseline_version}"
                    ),
                ),
                VerifyDiagnostic(
                    level="warning",
                    message=(
                        f"edited  : family={edited_family}, backend={edited_backend} "
                        f"{edited_version}"
                    ),
                ),
                VerifyDiagnostic(
                    level="warning",
                    message=(
                        "Ensure this cross-family comparison is intentional "
                        "(Compare & Evaluate flows should normally match families)."
                    ),
                ),
            )
    except _RECOVERABLE_EXCEPTIONS:
        return ()
    return ()


__all__ = ["warn_adapter_family_mismatch"]
