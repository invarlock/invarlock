"""Provenance assembly helpers for evaluation report generation."""

from __future__ import annotations

from typing import Any


def build_provenance_block(
    report: dict[str, Any],
    baseline_raw: dict[str, Any] | None,
    baseline_ref: dict[str, Any],
    artifacts_payload: dict[str, Any],
    policy_provenance: dict[str, Any],
    schedule_digest: str | None,
    ppl_analysis: dict[str, Any],
    current_run_id: str,
    *,
    compute_report_digest_fn: Any,
    collect_backend_versions_fn: Any,
    compute_edit_digest_fn: Any,
) -> dict[str, Any]:
    """Assemble report provenance with policy/baseline/edit/run context."""

    baseline_artifacts = (
        baseline_raw.get("artifacts", {}) if isinstance(baseline_raw, dict) else {}
    ) or {}
    baseline_report_hash = compute_report_digest_fn(baseline_raw)
    edited_report_hash = compute_report_digest_fn(report)

    provenance: dict[str, Any] = {
        "policy": dict(policy_provenance),
        "baseline": {
            "run_id": baseline_ref.get("run_id"),
            "report_hash": baseline_report_hash,
            "report_path": baseline_artifacts.get("report_path")
            or baseline_artifacts.get("logs_path"),
        },
        "edited": {
            "run_id": current_run_id,
            "report_hash": edited_report_hash,
            "report_path": artifacts_payload.get("report_path"),
        },
        "env_flags": collect_backend_versions_fn(),
    }

    try:
        report_prov = (
            report.get("provenance", {})
            if isinstance(report.get("provenance"), dict)
            else {}
        )
        provider_digest = (
            report_prov.get("provider_digest")
            if isinstance(report_prov, dict)
            else None
        )
        if isinstance(provider_digest, dict) and provider_digest:
            provenance["provider_digest"] = dict(provider_digest)
        try:
            ds = report_prov.get("dataset_split")
            sf = report_prov.get("split_fallback")
            if ds:
                provenance["dataset_split"] = ds
            if isinstance(sf, bool):
                provenance["split_fallback"] = sf
        except Exception:  # pragma: no cover
            pass
    except Exception:  # pragma: no cover
        pass

    if isinstance(ppl_analysis, dict) and ppl_analysis.get("window_plan"):
        provenance["window_plan"] = ppl_analysis["window_plan"]

    if isinstance(schedule_digest, str) and schedule_digest:
        provenance["window_ids_digest"] = schedule_digest
        provenance.setdefault("window_plan_digest", schedule_digest)
        try:
            if not isinstance(provenance.get("provider_digest"), dict):
                provenance["provider_digest"] = {"ids_sha256": schedule_digest}
        except Exception:  # pragma: no cover
            pass

    try:
        if isinstance(report, dict):
            provenance["edit_digest"] = compute_edit_digest_fn(report)
    except Exception:  # pragma: no cover
        pass

    return provenance
