from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.reporting.report_builder_support import (
    build_baseline_reference,
    extract_report_meta,
)
from invarlock.reporting.report_metric_impact import (
    compute_guard_metric_impact_from_guard,
    prepare_guard_metric_impact_section,
)
from invarlock.reporting.report_policy import (
    resolve_pm_acceptance_range_from_report,
    resolve_pm_drift_band_from_report,
    resolve_tiny_relax_from_report,
)
from invarlock.reporting.report_provenance import build_provenance_block
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.validation.report import compute_validation_flags
from tests.reporting._support_guard_metric_impact import ppl_arm_report


def test_reporting_owner_modules_expose_injection_points():
    result = SimpleNamespace(
        metrics={
            "metric_kind": "ppl_causal",
            "direction": "lower",
            "degradation_basis": "relative_increase",
            "degradation": 0.02,
            "display_value": 2.0,
            "display_unit": "percent",
            "bare_value": 10.0,
            "guarded_value": 10.2,
        },
        messages=["ok"],
        warnings=[],
        errors=[],
        checks={"guard_metric_impact": True},
        passed=True,
    )
    payload, passed = prepare_guard_metric_impact_section(
        {
            "bare_report": ppl_arm_report(10.0),
            "guarded_report": ppl_arm_report(10.2),
            "degradation_limit": 0.05,
        },
        validate_guard_metric_impact_fn=lambda *_a, **_k: result,
    )
    assert payload["evaluated"] is True
    assert passed is True

    bare_report = {"kind": "bare"}
    guarded_report = {"kind": "guarded"}

    def _pm(report, *, kind):  # noqa: ARG001
        point = 10.0 if report is bare_report else 10.5
        return {"final": point}

    quality = compute_guard_metric_impact_from_guard(
        {"bare_report": bare_report, "guarded_report": guarded_report},
        pm_kind_hint="ppl_causal",
        compute_primary_metric_from_report_fn=_pm,
        get_metric_fn=lambda _kind: SimpleNamespace(direction="lower"),
    )
    assert quality is not None
    assert quality["metric_kind"] == "ppl_causal"
    assert quality["direction"] == "lower"
    assert quality["bare_value"] == 10.0
    assert quality["guarded_value"] == 10.5
    assert quality["degradation_basis"] == "relative_increase"
    assert quality["degradation"] == pytest.approx(0.05)
    assert quality["display_value"] == pytest.approx(5.0)
    assert quality["display_unit"] == "percent"

    prov = build_provenance_block(
        {"run_id": "edited", "provenance": {"dataset_split": "eval"}},
        {"artifacts": {"report_path": "baseline.json"}},
        {"run_id": "base"},
        {"report_path": "report.json"},
        {"source": "test"},
        "abc123",
        {"window_plan": {"profile": "dev"}},
        "run123",
        compute_report_digest_fn=lambda payload: (
            str(payload.get("run_id", "missing"))
            if isinstance(payload, dict)
            else "missing"
        ),
        collect_backend_versions_fn=lambda: {"python": "3.12"},
        compute_edit_digest_fn=lambda _report: {"family": "report_only"},
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
        guard_metric_impact={},
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
