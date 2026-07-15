from __future__ import annotations

from rich.console import Console

from invarlock.cli import run_shell_output as run_output_mod
from invarlock.core.run_policy import GUARD_METRIC_DEGRADATION_LIMIT
from invarlock.reporting.report_metric_impact import (
    normalize_guard_metric_impact_result,
)
from invarlock.reporting.run_report_metrics_contract import (
    format_debug_metric_diffs,
    merge_primary_metric_health,
)


def test_format_debug_metric_diffs_returns_empty_on_non_dict_inputs() -> None:
    assert format_debug_metric_diffs(None, {}, None) == ""
    assert format_debug_metric_diffs({}, None, None) == ""


def test_format_debug_metric_diffs_includes_ratio_vs_baseline_fallback() -> None:
    pm = {"final": 12.0, "preview": 11.0, "ratio_vs_baseline": 2.4}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}
    baseline = {"metrics": {"primary_metric": {"final": 5.0}}}

    out = format_debug_metric_diffs(pm, metrics, baseline)
    assert "final: recomputed-recorded = +2.000000000" in out
    assert "preview: recomputed-recorded = +2.000000000" in out
    assert "ratio_vs_baseline: recomputed-recorded = +0.400000000" in out


def test_format_debug_metric_diffs_skips_log_terms_on_domain_error() -> None:
    pm = {"final": -1.0, "preview": 11.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}

    out = format_debug_metric_diffs(pm, metrics, baseline_report_data=None)
    assert "final: recomputed-recorded = -11.000000000" in out
    assert "Δlog(final)" not in out


def test_format_debug_metric_diffs_handles_bad_numeric_inputs() -> None:
    pm = {"final": "bad", "preview": "bad", "ratio_vs_baseline": "bad"}
    metrics = {"primary_metric": {"final": "bad", "preview": "bad"}}
    baseline = {"metrics": {"primary_metric": {"final": "bad"}}}

    out = format_debug_metric_diffs(pm, metrics, baseline)
    assert out == ""


def test_format_debug_metric_diffs_skips_preview_log_terms_on_domain_error() -> None:
    pm = {"final": 11.0, "preview": -1.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}

    out = format_debug_metric_diffs(pm, metrics, baseline_report_data=None)
    assert "preview: recomputed-recorded = -10.000000000" in out
    assert "Δlog(preview)" not in out


def test_format_debug_metric_diffs_skips_baseline_ratio_for_non_positive_baseline() -> (
    None
):
    pm = {"final": 12.0, "preview": 11.0, "ratio_vs_baseline": 2.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}
    baseline = {"metrics": {"primary_metric": {"final": 0.0}}}

    out = format_debug_metric_diffs(pm, metrics, baseline)
    assert "final: recomputed-recorded = +2.000000000" in out
    assert "ratio_vs_baseline: recomputed-recorded" not in out


def test_merge_primary_metric_health_returns_empty_for_non_mapping() -> None:
    assert merge_primary_metric_health(None, {"invalid": True}) == {}


def test_merge_primary_metric_health_prefers_core_flags() -> None:
    primary_metric = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 2.0,
        "ratio_vs_baseline": 2.0,
        "invalid": False,
        "degraded": False,
    }
    core_primary_metric = {
        "preview": None,
        "final": None,
        "invalid": True,
        "degraded": True,
        "degraded_reason": "non_finite_pm",
    }

    merged = merge_primary_metric_health(primary_metric, core_primary_metric)

    assert merged["preview"] == primary_metric["preview"]
    assert merged["final"] == primary_metric["final"]
    assert merged["ratio_vs_baseline"] == primary_metric["ratio_vs_baseline"]
    assert merged["invalid"] is True
    assert merged["degraded"] is True
    assert merged["degraded_reason"] == "non_finite_pm"


def test_normalize_metric_impact_result_marks_missing_degradation_as_not_evaluated() -> (
    None
):
    out = normalize_guard_metric_impact_result(None)
    assert out["evaluated"] is False
    assert out["passed"] is False


def test_normalize_metric_impact_result_handles_float_coercion_failure() -> None:
    class BadInt(int):
        def __float__(self) -> float:
            raise TypeError("boom")

    out = normalize_guard_metric_impact_result({"degradation": BadInt(1)})
    assert out["evaluated"] is False
    assert out["passed"] is False


def test_print_guard_metric_impact_summary_not_evaluated_path() -> None:
    console = Console(record=True)
    degradation_limit = run_output_mod._print_guard_metric_impact_summary(
        console, {"evaluated": False}
    )
    assert degradation_limit == GUARD_METRIC_DEGRADATION_LIMIT
    assert "not evaluated" in console.export_text()


def test_print_guard_metric_impact_summary_formats_percent_and_limit() -> None:
    console = Console(record=True)
    degradation_limit = run_output_mod._print_guard_metric_impact_summary(
        console,
        {
            "evaluated": True,
            "passed": False,
            "display_value": 1.23,
            "display_unit": "percent",
            "degradation_limit": 0.02,
        },
    )
    assert degradation_limit == 0.02
    text = console.export_text()
    assert "FAIL" in text
    assert "+1.23%" in text
    assert "≤ +2.0%" in text


def test_print_guard_metric_impact_summary_formats_percentage_points() -> None:
    console = Console(record=True)
    degradation_limit = run_output_mod._print_guard_metric_impact_summary(
        console,
        {
            "evaluated": True,
            "passed": True,
            "display_value": 0.5,
            "display_unit": "percentage_points",
            "degradation_limit": "bad",
        },
    )
    assert degradation_limit == GUARD_METRIC_DEGRADATION_LIMIT
    text = console.export_text()
    assert "PASS" in text
    assert "+0.50 pp" in text
    assert "≤ +1.0 pp" in text


def test_print_guard_metric_impact_summary_handles_missing_display_fields() -> None:
    console = Console(record=True)
    run_output_mod._print_guard_metric_impact_summary(
        console,
        {
            "evaluated": True,
            "passed": True,
            "display_value": "bad",
            "display_unit": "percent",
        },
    )
    assert "display unavailable" in console.export_text()


def test_print_guard_metric_impact_summary_uses_fallback_for_bad_default_limit() -> (
    None
):
    console = Console(record=True)
    degradation_limit = run_output_mod._print_guard_metric_impact_summary(
        console,
        {"evaluated": False},
        default_limit="bad",
    )
    assert degradation_limit == GUARD_METRIC_DEGRADATION_LIMIT


def test_print_guard_metric_impact_summary_uses_fallback_for_non_finite_limit() -> None:
    console = Console(record=True)
    degradation_limit = run_output_mod._print_guard_metric_impact_summary(
        console,
        {"evaluated": False},
        default_limit=float("nan"),
    )
    assert degradation_limit == GUARD_METRIC_DEGRADATION_LIMIT


def test_print_retry_summary_prints_when_attempts_present() -> None:
    console = Console(record=True)

    class Retry:
        attempt_history = [object()]

        def get_attempt_summary(self):  # noqa: ANN001
            return {"total_attempts": 2, "elapsed_time": 1.2}

    run_output_mod._print_retry_summary(console, Retry())
    assert "Retry Summary" in console.export_text()


def test_print_retry_summary_no_attempts_silent() -> None:
    console = Console(record=True)
    run_output_mod._print_retry_summary(console, None)
    assert console.export_text() == ""


def test_print_retry_summary_swallows_summary_errors() -> None:
    console = Console(record=True)

    class Retry:
        attempt_history = [object()]

        def get_attempt_summary(self):  # noqa: ANN001
            raise RuntimeError("boom")

    run_output_mod._print_retry_summary(console, Retry())
    assert console.export_text() == ""
