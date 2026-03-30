from __future__ import annotations

from types import SimpleNamespace

from invarlock.reporting.report_builder_support import (
    build_baseline_reference,
    extract_report_meta,
)
from invarlock.reporting.report_overhead import (
    compute_quality_overhead_from_guard,
    prepare_guard_overhead_section,
)
from invarlock.reporting.report_policy import (
    resolve_pm_acceptance_range_from_report,
    resolve_pm_drift_band_from_report,
    resolve_tiny_relax_from_report,
)
from invarlock.reporting.report_provenance import build_provenance_block
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.report_validation import compute_validation_flags


def test_reporting_owner_modules_expose_injection_points():
    result = SimpleNamespace(
        metrics={
            "overhead_ratio": 1.02,
            "overhead_percent": 2.0,
            "bare_ppl": 10.0,
            "guarded_ppl": 10.2,
        },
        messages=["ok"],
        warnings=[],
        errors=[],
        checks={"ratio": True},
        passed=True,
    )
    payload, passed = prepare_guard_overhead_section(
        {
            "bare_report": {"metrics": {}},
            "guarded_report": {"metrics": {}},
            "overhead_threshold": 0.05,
        },
        validate_guard_overhead_fn=lambda *_a, **_k: result,
    )
    assert payload["evaluated"] is True
    assert passed is True

    bare_report = {"kind": "bare"}
    guarded_report = {"kind": "guarded"}

    def _pm(report, *, kind):  # noqa: ARG001
        point = 10.0 if report is bare_report else 10.5
        return {"final": point}

    quality = compute_quality_overhead_from_guard(
        {"bare_report": bare_report, "guarded_report": guarded_report},
        pm_kind_hint="ppl_causal",
        compute_primary_metric_from_report_fn=_pm,
        get_metric_fn=lambda _kind: SimpleNamespace(direction="lower"),
    )
    assert quality == {"basis": "ratio", "value": 1.05, "kind": "ppl_causal"}

    prov = build_provenance_block(
        {"run_id": "edited", "provenance": {"dataset_split": "eval"}},
        {"artifacts": {"report_path": "baseline.json"}},
        {"run_id": "base"},
        {"report_path": "report.json"},
        {"source": "test"},
        "abc123",
        {"window_plan": {"profile": "dev"}},
        "run123",
        compute_report_digest_fn=lambda payload: str(payload.get("run_id", "missing"))
        if isinstance(payload, dict)
        else "missing",
        collect_backend_versions_fn=lambda: {"python": "3.12"},
        compute_edit_digest_fn=lambda _report: {"family": "cert_only"},
    )
    assert prov["policy"]["source"] == "test"
    assert prov["window_ids_digest"] == "abc123"

    assert resolve_pm_acceptance_range_from_report({}) == {}
    assert resolve_pm_drift_band_from_report({}) == {}
    assert (
        resolve_tiny_relax_from_report({"context": {"run": {"tiny_relax": True}}})
        is True
    )

    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={"status": "ok"},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=1.0,
        guard_overhead={},
        primary_metric={},
        moe={},
        dataset_capacity={"tokens_available": 200000},
        pm_acceptance_range={"min": 0.95, "max": 1.1},
        pm_drift_band={"min": 0.95, "max": 1.05},
        pm_tail={},
        tiny_relax=False,
        get_tier_policies_fn=lambda: {"balanced": {"metrics": {"pm_ratio": {}}}},
    )
    assert flags["primary_metric_acceptable"] is True


def test_report_make_support_helpers_expose_narrower_builder_boundaries():
    report = create_empty_report()
    report["meta"]["model_id"] = "model-x"
    report["meta"]["adapter"] = "hf_causal"
    report["meta"]["device"] = "cpu"
    report["meta"]["ts"] = "now"
    report["meta"]["commit"] = "abc"
    report["meta"]["seed"] = 7
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
    }

    meta = extract_report_meta(report, [])
    baseline_ref = build_baseline_reference(report, report, {"run_id": "base"})

    assert meta["seed"] == 7
    assert baseline_ref["primary_metric"]["final"] == 1.0
