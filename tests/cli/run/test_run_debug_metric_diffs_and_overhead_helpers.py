from __future__ import annotations

from rich.console import Console

from invarlock.cli import run_shell_output as run_output_mod
from invarlock.core.run_guard_overhead_policy import normalize_guard_overhead_result
from invarlock.core.run_policy import GUARD_OVERHEAD_THRESHOLD
from invarlock.reporting.run_metric_utils import (
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
    assert "final: v1-v1 = +2.000000000" in out
    assert "preview: v1-v1 = +2.000000000" in out
    assert "ratio_vs_baseline: v1-v1 = +0.400000000" in out


def test_format_debug_metric_diffs_skips_log_terms_on_domain_error() -> None:
    pm = {"final": -1.0, "preview": 11.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}

    out = format_debug_metric_diffs(pm, metrics, baseline_report_data=None)
    assert "final: v1-v1 = -11.000000000" in out
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
    assert "preview: v1-v1 = -10.000000000" in out
    assert "Δlog(preview)" not in out


def test_format_debug_metric_diffs_skips_baseline_ratio_for_non_positive_baseline() -> (
    None
):
    pm = {"final": 12.0, "preview": 11.0, "ratio_vs_baseline": 2.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}
    baseline = {"metrics": {"primary_metric": {"final": 0.0}}}

    out = format_debug_metric_diffs(pm, metrics, baseline)
    assert "final: v1-v1 = +2.000000000" in out
    assert "ratio_vs_baseline: v1-v1" not in out


def test_merge_primary_metric_health_returns_empty_for_non_mapping() -> None:
    assert merge_primary_metric_health(None, {"invalid": True}) == {}


def test_normalize_overhead_result_marks_missing_ratio_as_not_evaluated() -> None:
    out = normalize_guard_overhead_result(None)
    assert out["evaluated"] is False
    assert out["passed"] is True


def test_normalize_overhead_result_handles_float_coercion_failure() -> None:
    class BadInt(int):
        def __float__(self) -> float:
            raise TypeError("boom")

    out = normalize_guard_overhead_result({"overhead_ratio": BadInt(1)})
    assert out["evaluated"] is False
    assert out["passed"] is True


def test_print_guard_overhead_summary_not_evaluated_path() -> None:
    console = Console(record=True)
    threshold = run_output_mod._print_guard_overhead_summary(
        console, {"evaluated": False}
    )
    assert threshold == GUARD_OVERHEAD_THRESHOLD
    assert "not evaluated" in console.export_text()


def test_print_guard_overhead_summary_formats_percent_and_threshold() -> None:
    console = Console(record=True)
    threshold = run_output_mod._print_guard_overhead_summary(
        console,
        {
            "evaluated": True,
            "passed": False,
            "overhead_percent": 1.23,
            "overhead_threshold": 0.02,
        },
    )
    assert threshold == 0.02
    text = console.export_text()
    assert "FAIL" in text
    assert "+1.23%" in text
    assert "≤ +2.0%" in text


def test_print_guard_overhead_summary_falls_back_to_ratio_and_default_threshold() -> (
    None
):
    console = Console(record=True)
    threshold = run_output_mod._print_guard_overhead_summary(
        console,
        {
            "evaluated": True,
            "passed": True,
            "overhead_ratio": 1.005,
            "overhead_threshold": "bad",
        },
    )
    assert threshold == GUARD_OVERHEAD_THRESHOLD
    text = console.export_text()
    assert "PASS" in text
    assert "1.005x" in text


def test_print_guard_overhead_summary_handles_missing_ratio_and_percent() -> None:
    console = Console(record=True)
    run_output_mod._print_guard_overhead_summary(
        console,
        {
            "evaluated": True,
            "passed": True,
            "overhead_percent": "bad",
            "overhead_ratio": "bad",
        },
    )
    assert "not evaluated" in console.export_text()


def test_print_guard_overhead_summary_uses_fallback_for_bad_default_threshold() -> None:
    console = Console(record=True)
    threshold = run_output_mod._print_guard_overhead_summary(
        console,
        {"evaluated": False},
        default_threshold="bad",
    )
    assert threshold == GUARD_OVERHEAD_THRESHOLD


def test_print_guard_overhead_summary_uses_fallback_for_non_finite_threshold() -> None:
    console = Console(record=True)
    threshold = run_output_mod._print_guard_overhead_summary(
        console,
        {"evaluated": False},
        default_threshold=float("nan"),
    )
    assert threshold == GUARD_OVERHEAD_THRESHOLD


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
