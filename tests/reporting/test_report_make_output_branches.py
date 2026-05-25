from __future__ import annotations

from typing import Any

from invarlock.reporting import primary_metric_utils as primary_metric_utils_mod
from invarlock.reporting import report_enrichment as report_enrichment_mod
from invarlock.reporting import report_make_output as report_make_output_mod
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
    report_make_output_mod._finalize_evaluation_report(
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
