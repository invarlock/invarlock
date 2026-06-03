from __future__ import annotations

from typing import Any

from invarlock.reporting import primary_metric_utils as primary_metric_utils_mod
from invarlock.reporting import report_enrichment as report_enrichment_mod
from invarlock.reporting import report_make as report_make_mod
from invarlock.reporting import (
    report_primary_metric_policy as report_primary_metric_policy_mod,
)


def _stub_finalize_dependencies(monkeypatch) -> None:
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    for name in (
        "attach_quality_overhead",
        "attach_policy_digest",
        "attach_secondary_metrics",
        "attach_classification",
        "attach_system_overhead",
        "ensure_primary_metric_display_ci",
        "attach_telemetry_summary_line",
        "attach_confidence_label",
    ):
        monkeypatch.setattr(report_enrichment_mod, name, _noop, raising=True)
    monkeypatch.setattr(
        report_primary_metric_policy_mod,
        "propagate_pairing_stats",
        _noop,
        raising=True,
    )
    monkeypatch.setattr(
        report_primary_metric_policy_mod,
        "enforce_display_ci_alignment",
        _noop,
        raising=True,
    )
    monkeypatch.setattr(
        primary_metric_utils_mod,
        "attach_primary_metric",
        _noop,
        raising=True,
    )


def _finalize_report(evaluation_report: dict[str, Any], monkeypatch) -> list[dict]:
    _stub_finalize_dependencies(monkeypatch)
    diagnostics: list[dict] = []
    report_make_mod._finalize_evaluation_report(
        evaluation_report,
        report_map={},
        report={},
        baseline_raw_map={},
        baseline_normalized={},
        baseline_ref={},
        telemetry={},
        resolved_policy={},
        auto={},
        policy_provenance={},
        raw_guard_ctx=None,
        ppl_analysis={},
        window_plan_profile="ci",
        pm_drift_band={"ratio": [0.9, 1.1]},
        tiny_relax=True,
        current_run_id="run-1",
        build_diagnostics=[{"code": "test"}],
        record_blocking_diagnostic=diagnostics.append,
        non_fatal_exceptions=(Exception,),
    )
    return diagnostics


def test_build_evaluation_report_preserves_top_level_guard_outcomes() -> None:
    report = report_make_mod._build_evaluation_report(
        report_map={
            "guards": [
                {"name": "spectral", "passed": True, "decision": "allow"},
                {"name": "rmt", "passed": False, "decision": "block"},
                {"name": "variance", "passed": True, "decision": "allow"},
                {"name": "invariants", "passed": True, "decision": "allow"},
            ]
        },
        current_run_id="run-1",
        meta={},
        auto={},
        dataset_info={},
        edit_metadata={},
        telemetry={},
        baseline_ref={},
        invariants={},
        spectral={"summary": {"status": "stable"}},
        rmt={"status": "stable"},
        variance={"enabled": False},
        structure={},
        policies={},
        resolved_policy={},
        policy_provenance={},
        provenance={},
        plugin_provenance={},
        edit_name=None,
        artifacts_payload={},
        validation_filtered={},
        guard_overhead_section={},
        pm_tail_result={},
    )

    assert report["spectral"]["passed"] is True
    assert report["spectral"]["decision"] == "allow"
    assert report["rmt"]["passed"] is False
    assert report["rmt"]["decision"] == "block"
    assert report["variance"]["passed"] is True
    assert report["invariants"]["decision"] == "allow"


def test_guard_outcome_collection_handles_malformed_entries() -> None:
    assert report_make_mod._collect_guard_outcomes("not-a-list") == {}

    outcomes = report_make_mod._collect_guard_outcomes(
        [
            object(),
            {"name": "unknown", "passed": False},
            {
                "name": "rmt",
                "passed": True,
                "decision": "allow",
                "policy": {"source": "first"},
            },
            {"name": "rmt", "passed": False, "decision": "rollback"},
        ]
    )

    assert outcomes["rmt"]["passed"] is False
    assert outcomes["rmt"]["decision"] == "rollback"
    assert outcomes["rmt"]["policy"] == {"source": "first"}


def test_attach_top_level_guard_outcomes_skips_non_dict_sections() -> None:
    report = {"guards": [{"name": "rmt", "passed": True}], "rmt": "bad-section"}

    report_make_mod._attach_top_level_guard_outcomes(report)

    assert report["rmt"] == "bad-section"


def test_finalize_evaluation_report_handles_non_dict_tiny_relax_sections(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, Any] = {
        "auto": [],
        "provenance": [],
        "primary_metric": [],
        "meta": [],
        "context": {
            "profile": "ci",
            "assurance": {"mode": "strict"},
            "tier": "balanced",
            "runtime": {"execution_mode": "container"},
        },
        "guards": [
            {"name": "invariants"},
            {"name": "spectral"},
            {"name": "rmt"},
            {"name": "variance"},
            {"name": "invariants"},
        ],
        "spectral": {"supported": True},
        "rmt": {"supported": True},
        "variance": {"supported": True},
        "invariants": {"supported": True},
    }

    diagnostics = _finalize_report(evaluation_report, monkeypatch)

    assert diagnostics == []
    assert evaluation_report["assurance"]["runtime_provenance_verification_status"] == (
        "pending"
    )
    assert evaluation_report["assurance"]["verdict"] == "pending_verifier"


def test_finalize_evaluation_report_handles_existing_tiny_relax_flag(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, Any] = {
        "auto": {},
        "provenance": {"flags": ["tiny_relax"]},
        "primary_metric": {},
        "meta": {},
        "context": {"runtime": {"execution_mode": "host"}},
    }

    diagnostics = _finalize_report(evaluation_report, monkeypatch)

    assert diagnostics == []
    assert evaluation_report["provenance"]["flags"] == ["tiny_relax"]
    assert evaluation_report["assurance"]["runtime_provenance_declared"] == "host"
