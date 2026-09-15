"""Neutral rendering of the stored captured-evaluation report.

This module is deliberately a presentation boundary.  It authenticates the
small captured pack envelope and displays the recorded comparison; it does not
discover receipts, replay runs, or recompute scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.captured_contracts import (
    CAPTURED_PACK_FORMAT,
    DETECTOR_LIMIT,
    PAYLOADS,
    CapturedContractError,
    captured_snapshot,
    load_payloads,
    read_file,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
)
from invarlock.record_reporting import (
    _assemble_view,
)
from invarlock.report_presentation import (
    ReportView,
)


class CapturedReportError(ValueError):
    """Raised when a captured report cannot be rendered safely."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def is_captured_manifest(path: Path) -> bool:
    """Boundedly identify captured input before native report validation."""
    try:
        raw = read_file(path / "manifest.json", DETECTOR_LIMIT)
        value = parse_json_bytes(raw, label="evidence manifest")
    except FileNotFoundError:
        # Preserve native missing-pack diagnostics when no discriminator exists.
        return False
    except (OSError, StrictJsonError, CapturedContractError) as exc:
        raise CapturedReportError(
            "evidence manifest could not be identified safely"
        ) from exc
    if isinstance(value, dict):
        if value.get("format") == "invarlock/evidence-pack-v1":
            return False
        if (
            value.get("format") == CAPTURED_PACK_FORMAT
            and value.get("kind") == "captured"
        ):
            return True
    raise CapturedReportError("evidence manifest format or kind is unsupported")


def _load(
    pack: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str, dict[str, Any]]:
    try:
        with captured_snapshot(pack) as snapshot:
            manifest, payloads, signer = load_payloads(snapshot)
            payload_bytes = {name: snapshot.files[name] for name in PAYLOADS.values()}
    except (OSError, CapturedContractError) as exc:
        raise CapturedReportError(str(exc)) from exc
    return manifest, payloads, signer, payload_bytes


def _view(
    manifest: dict[str, Any], payloads: dict[str, dict[str, Any]], signer: str
) -> ReportView:
    comparison = payloads["report"]
    policy = payloads["policy"]
    policy_by_name = {
        item.get("name"): item
        for item in policy.get("metrics", [])
        if isinstance(item, dict)
    }
    expected = {
        (name, scope)
        for name in policy_by_name
        for scope in ("overall", *(item["name"] for item in policy["slices"]))
    }
    seen: set[tuple[str, str]] = set()
    for item in comparison.get("metrics", []):
        if not isinstance(item, dict):
            raise CapturedReportError("captured report metrics are invalid")
        name, decision = item.get("name"), item.get("decision")
        if not isinstance(name, str) or decision not in {
            "pass",
            "regression",
            "insufficient_evidence",
        }:
            raise CapturedReportError("captured report metric decision is invalid")
        scope = item.get("slice")
        if not isinstance(scope, str) or (name, scope) not in expected:
            raise CapturedReportError("captured report metric scope is not in policy")
        if (name, scope) in seen:
            raise CapturedReportError("captured report metric scope is duplicated")
        seen.add((name, scope))
        configured = policy_by_name[name]
        if any(
            item.get(key) != configured[key]
            for key in ("kind", "direction", "unit", "aggregation")
        ) or item.get("scoring_assurance") != (
            "recorded" if configured["kind"] == "recorded" else "recomputed"
        ):
            raise CapturedReportError("captured report metric contradicts policy")
        missing = item.get("missing_ids", [])
        if not isinstance(missing, list):
            raise CapturedReportError(
                "captured report missing result inventory is invalid"
            )
    if not seen:
        raise CapturedReportError("captured report contains no metrics")
    if seen != expected:
        raise CapturedReportError("captured report omits configured metric scopes")
    return _assemble_view(comparison, payloads, manifest, signer)


__all__ = [
    "CapturedReportError",
    "is_captured_manifest",
]
