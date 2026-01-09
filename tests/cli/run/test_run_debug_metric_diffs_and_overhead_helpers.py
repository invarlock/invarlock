from __future__ import annotations

from rich.console import Console

from invarlock.cli.commands import run as run_mod


def test_format_debug_metric_diffs_returns_empty_on_non_dict_inputs() -> None:
    assert run_mod._format_debug_metric_diffs(None, {}, None) == ""
    assert run_mod._format_debug_metric_diffs({}, None, None) == ""


def test_format_debug_metric_diffs_includes_ratio_vs_baseline_fallback() -> None:
    pm = {"final": 12.0, "preview": 11.0, "ratio_vs_baseline": 2.4}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}
    baseline = {"metrics": {"primary_metric": {"final": 5.0}}}

    out = run_mod._format_debug_metric_diffs(pm, metrics, baseline)
    assert "final: v1-v1 = +2.000000000" in out
    assert "preview: v1-v1 = +2.000000000" in out
    assert "ratio_vs_baseline: v1-v1 = +0.400000000" in out


def test_format_debug_metric_diffs_skips_log_terms_on_domain_error() -> None:
    pm = {"final": -1.0, "preview": 11.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}

    out = run_mod._format_debug_metric_diffs(pm, metrics, baseline_report_data=None)
    assert "final: v1-v1 = -11.000000000" in out
    assert "Δlog(final)" not in out


def test_normalize_overhead_result_marks_missing_ratio_as_not_evaluated() -> None:
    out = run_mod._normalize_overhead_result(None)
    assert out["evaluated"] is False
    assert out["passed"] is True


def test_print_guard_overhead_summary_not_evaluated_path() -> None:
    console = Console(record=True)
    threshold = run_mod._print_guard_overhead_summary(console, {"evaluated": False})
    assert threshold == run_mod.GUARD_OVERHEAD_THRESHOLD
    assert "not evaluated" in console.export_text()


def test_print_guard_overhead_summary_formats_percent_and_threshold() -> None:
    console = Console(record=True)
    threshold = run_mod._print_guard_overhead_summary(
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


def test_print_guard_overhead_summary_falls_back_to_ratio_and_default_threshold() -> None:
    console = Console(record=True)
    threshold = run_mod._print_guard_overhead_summary(
        console,
        {
            "evaluated": True,
            "passed": True,
            "overhead_ratio": 1.005,
            "overhead_threshold": "bad",
        },
    )
    assert threshold == run_mod.GUARD_OVERHEAD_THRESHOLD
    text = console.export_text()
    assert "PASS" in text
    assert "1.005x" in text


def test_print_guard_overhead_summary_handles_missing_ratio_and_percent() -> None:
    console = Console(record=True)
    run_mod._print_guard_overhead_summary(
        console,
        {
            "evaluated": True,
            "passed": True,
            "overhead_percent": "bad",
            "overhead_ratio": "bad",
        },
    )
    assert "not evaluated" in console.export_text()

