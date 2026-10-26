"""Explicit native report facade; presentation work remains in Rust."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from invarlock.evidence_reporting import EvidenceReportV2


def render_evidence(evidence_path: Path, **options: Any) -> EvidenceReportV2:
    from invarlock.evidence_reporting import EvidenceReportError, EvidenceReportV2
    from invarlock.native_backend import (
        NativeBackendUnavailable,
        require_native,
        transact,
    )

    try:
        invarlock_native = require_native()
    except NativeBackendUnavailable as exc:
        raise EvidenceReportError(str(exc), exit_code=2) from exc

    try:
        outcome = transact("report", {"evidence_path": evidence_path, **options})
    except invarlock_native.NativeVerificationError as exc:
        raise EvidenceReportError(str(exc), exit_code=exc.exit_code) from exc
    transport = outcome["payload"]
    payload = transport["report"]
    if outcome["status"]:
        raise EvidenceReportError(
            "; ".join(payload.get("errors", ("report failed",))),
            exit_code=outcome["status"],
            payload=payload,
        )
    common = {
        "text": transport["text"],
        "kind": payload["kind"],
        "requested_outputs": payload["requested_outputs"],
        "written_outputs": payload["written_outputs"],
        "errors": tuple(payload["errors"]),
    }
    if payload["kind"] in {"judge", "evidence_set"}:
        from invarlock.evidence_sets.reporting import EvidenceSetReport
        from invarlock.judge_measurements.reporting import JudgeEvidenceReport

        report_type: type[JudgeEvidenceReport] | type[EvidenceSetReport]
        if payload["kind"] == "judge":
            report_type = JudgeEvidenceReport
            digest_key = "evidence_digest"
        else:
            report_type = EvidenceSetReport
            digest_key = "index_sha256"
        reserved = {
            "format_version",
            "kind",
            "ok",
            digest_key,
            "requested_outputs",
            "written_outputs",
            "errors",
        }
        return report_type(
            **common,
            pack_manifest_digest=payload[digest_key],
            facts={key: value for key, value in payload.items() if key not in reserved},
        )
    return EvidenceReportV2(
        **common,
        pack_manifest_digest=payload["pack_manifest_digest"],
        failed_output=payload.get("failed_output"),
    )
