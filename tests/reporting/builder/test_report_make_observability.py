from __future__ import annotations

from copy import deepcopy

import pytest

from invarlock.reporting import report_make_assembly as report_make_assembly_mod
from invarlock.reporting import report_normalization as report_normalization_mod
from invarlock.reporting import report_primary_metric_analysis as report_pm_analysis_mod
from invarlock.reporting import report_schema as allowlist_mod
from invarlock.reporting.report_make import make_report
from tests.reporting.builder._support_full_context import _rich_run_report


def test_make_evaluation_report_marks_tiny_relax() -> None:
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["context"] = {"run": {"tiny_relax": True}}
    evaluation_report = make_report(report, baseline)
    assert evaluation_report["auto"]["tiny_relax"] is True
    stats = evaluation_report["dataset"]["windows"]["stats"]
    assert "coverage" in stats and "window_match_fraction" in stats
    metric_impact = evaluation_report["guard_metric_impact"]
    assert metric_impact["evaluated"] is True
    assert metric_impact["passed"] is True
    assert metric_impact["degradation"] == pytest.approx(0.01)


def test_make_evaluation_report_embeds_telemetry_summary(monkeypatch):
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    evaluation_report = make_report(report, baseline)
    assert evaluation_report["telemetry"]["summary_line"].startswith(
        "INVARLOCK_TELEMETRY"
    )


def test_make_evaluation_report_marks_broken_profile_provenance_unhealthy(
    monkeypatch,
) -> None:
    class _BrokenContext(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key == "profile":
                raise RuntimeError("context-profile-bad")
            return super().get(key, default)

    class _BrokenWindowPlan(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key == "profile":
                raise RuntimeError("window-plan-profile-bad")
            return super().get(key, default)

    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["context"] = _BrokenContext({"profile": "ci"})
    report["metrics"]["window_plan"] = _BrokenWindowPlan({"profile": "ci"})

    # Keep the test focused on report_make's provenance extraction rather than
    # the primary-metric helper that also inspects window-plan shape.
    def _stub_primary_metric_analysis(*_args, **_kwargs):
        return {}, "dev"

    monkeypatch.setattr(
        report_pm_analysis_mod,
        "build_primary_metric_analysis",
        _stub_primary_metric_analysis,
        raising=True,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "validated_run_report_view",
        lambda payload: payload,
        raising=True,
    )

    evaluation_report = make_report(report, baseline)

    diagnostics = evaluation_report.get("meta", {}).get("build_diagnostics", [])
    codes = {entry.get("code") for entry in diagnostics if isinstance(entry, dict)}

    assert "policy.profile_from_context_failed" in codes
    assert "policy.profile_from_window_plan_failed" in codes
    assert any(
        entry.get("severity") == "error"
        for entry in diagnostics
        if isinstance(entry, dict)
    )
    assert evaluation_report["validation"]["primary_metric_acceptable"] is False


def test_make_report_assurance_ignores_diagnostic_code_text_for_fallback_fields(
    monkeypatch,
) -> None:
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["metrics"]["primary_metric"]["display_ci"] = [
        0.9900498337491681,
        1.0202013400267558,
    ]

    class _BrokenMeta(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key == "env_flags":
                raise RuntimeError("env-flags-bad")
            return super().get(key, default)

    report["meta"] = _BrokenMeta(report["meta"])
    monkeypatch.setattr(
        report_normalization_mod,
        "validated_run_report_view",
        lambda payload: payload,
        raising=True,
    )

    evaluation_report = make_report(report, baseline)

    codes = {
        entry.get("code")
        for entry in evaluation_report.get("meta", {}).get("build_diagnostics", [])
        if isinstance(entry, dict)
    }
    assert "meta.env_flags_unavailable" in codes
    assert evaluation_report["report_build"] == {
        "synthesized_fields": [],
        "repaired_fields": [],
        "fallback_fields": [],
    }
    assert evaluation_report["assurance"]["fallback_fields_used"] is False


def test_make_report_assurance_flags_structured_primary_metric_events() -> None:
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["metrics"]["primary_metric"]["display_ci"] = [True, False]

    evaluation_report = make_report(report, baseline)

    events = evaluation_report["report_build"]["repaired_fields"]
    assert any(event["field"] == "primary_metric.display_ci" for event in events)
    assert evaluation_report["assurance"]["fallback_fields_used"] is True


def test_make_evaluation_report_marks_broken_env_flag_provenance_unhealthy(
    monkeypatch,
) -> None:
    class _BrokenMeta(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key == "env_flags":
                raise RuntimeError("env-flags-bad")
            return super().get(key, default)

    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["meta"] = _BrokenMeta(report["meta"])

    monkeypatch.setattr(
        report_pm_analysis_mod,
        "build_primary_metric_analysis",
        lambda *_args, **_kwargs: ({}, "dev"),
        raising=True,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "validated_run_report_view",
        lambda payload: payload,
        raising=True,
    )

    evaluation_report = make_report(report, baseline)
    diagnostics = evaluation_report.get("meta", {}).get("build_diagnostics", [])
    codes = {entry.get("code") for entry in diagnostics if isinstance(entry, dict)}

    assert "meta.env_flags_unavailable" in codes
    assert any(
        entry.get("code") == "meta.env_flags_unavailable"
        and entry.get("severity") == "error"
        for entry in diagnostics
        if isinstance(entry, dict)
    )
    assert evaluation_report["validation"]["primary_metric_acceptable"] is False


def test_make_evaluation_report_fails_closed_when_validation_contract_is_unavailable(
    monkeypatch,
) -> None:
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    monkeypatch.setattr(
        report_make_assembly_mod,
        "load_validation_allowlist",
        lambda: (_ for _ in ()).throw(
            allowlist_mod.ValidationAllowlistContractError("allowlist-missing")
        ),
        raising=True,
    )

    with pytest.raises(
        allowlist_mod.ValidationAllowlistContractError,
        match="allowlist-missing",
    ):
        make_report(report, baseline)
