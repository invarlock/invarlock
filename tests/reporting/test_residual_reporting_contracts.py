from __future__ import annotations

import math

import pytest

from invarlock.reporting import (
    report_edit_summary,
    report_outline,
    report_primary_metric_analysis,
    report_primary_metric_counts,
    report_summary,
)


def test_edit_summary_repairs_malformed_inference_receipt_containers() -> None:
    config: dict = {"scope": "unknown", "group_size": 32, "frac": 0.05}
    inference = {"flags": [], "sources": [], "log": "bad"}
    diagnostics = report_edit_summary.extract_compression_diagnostics(
        "quant_rtn",
        config,
        {"params_changed": 0, "layers_modified": 1},
        {},
        inference,
    )

    assert isinstance(inference["flags"], dict)
    assert isinstance(inference["sources"], dict)
    assert isinstance(inference["log"], list)
    assert diagnostics["parameter_analysis"]["group_size"]["value"] == 32
    assert any("too small" in warning for warning in diagnostics["warnings"])


def test_edit_summary_extracts_savings_without_rank_map() -> None:
    ranks = report_edit_summary.extract_rank_information(
        {"frac": 0.5},
        {
            "savings": {
                "total_realized_params_saved": 10,
                "total_theoretical_params_saved": 20,
                "deploy_mode": "factorized",
            }
        },
    )
    assert ranks["savings_summary"]["mode"] == "realized"
    assert ranks["savings_summary"]["total_realized_params_saved"] == 10


@pytest.mark.parametrize(
    ("plan", "expected_scope"),
    [
        ({"head_budget": {"count": 2}}, "heads"),
        ({"mlp_budget": {"count": 4}}, "ffn"),
        (
            {"head_budget": {"count": 2}, "mlp_budget": {"count": 4}},
            "heads+ffn",
        ),
    ],
)
def test_edit_metadata_infers_scope_from_real_budget_owners(
    plan: dict,
    expected_scope: str,
) -> None:
    metadata = report_edit_summary.extract_edit_metadata(
        {
            "edit": {"name": "magnitude_prune", "plan": plan},
            "meta": {"seed": 9},
        },
        {},
    )
    assert metadata["scope"] == expected_scope
    assert metadata["seed"] == 9


def test_edit_metadata_rejects_absent_and_noncanonical_implementation() -> None:
    assert report_edit_summary.extract_edit_metadata({}, {}) == {}
    metadata = report_edit_summary.extract_edit_metadata(
        {
            "edit": {
                "name": "custom-name",
                "algorithm": "invented",
                "implementation": "StructuredLowRankFixture",
                "config": {"plan": {"target_sparsity": 0.25}},
            }
        },
        {},
    )
    assert metadata["algorithm"] == ""
    assert metadata["implementation"] == ""
    assert metadata["budgets"]["target_sparsity"] == 0.25


def test_outline_scalar_formatters_cover_empty_invalid_and_finite_values() -> None:
    assert report_outline._status_bool(None) == ("N/A", "info")
    assert report_outline._status_bool(None, default=False) == ("FAIL", "fail")
    assert report_outline._status_bool("yes") == ("N/A", "info")
    assert report_outline._format_percent_range([]) == "N/A"
    assert report_outline._format_percent_range([0.2]) == "20.0%"
    assert report_outline._format_ci({"ci": ["bad", 1]}, "accuracy") == "N/A"
    assert (
        report_outline._format_baseline_comparison(
            {"kind": "ppl_causal", "ratio_vs_baseline": float("nan")}
        )
        == "N/A"
    )
    assert (
        report_outline._format_baseline_comparison(
            {"kind": "score", "ratio_vs_baseline": 1.25}
        )
        == "1.250"
    )


def test_outline_baseline_and_benchmark_cover_mixed_scenarios() -> None:
    assert (
        report_outline._baseline_summary({"baseline_ref": {"model_id": "base"}})
        == "base"
    )
    run_summary = report_outline._baseline_summary(
        {"provenance": {"baseline": {"run_id": "1234567890abcdef"}}}
    )
    assert run_summary.startswith("run 123456")

    section = report_outline._build_benchmark_section(
        {
            "benchmark": {
                "profile": "release",
                "scenarios": [
                    "noise",
                    {"skip": True},
                    {
                        "pass": {"pm": True, "runtime": True},
                        "guard_primary_metric_impact": 0.01,
                        "guard_runtime_overhead": 0.02,
                        "guard_memory_overhead": 0.03,
                        "rmt_outliers_bare": 3,
                        "rmt_outliers_guarded": 1,
                    },
                ],
            }
        }
    )
    assert section is not None
    assert "3 total, 1 passed, 1 skipped" in section.facts[1].value


def test_primary_metric_analysis_bounds_and_snapshot_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert report_primary_metric_analysis._coerce_bounds(None) is None
    assert report_primary_metric_analysis._coerce_bounds(["bad", 1]) is None
    assert report_primary_metric_analysis._coerce_bounds([0.9, 1.1]) == (0.9, 1.1)
    assert report_primary_metric_analysis._build_drift_ci((2.0, 4.0), (3.0, 6.0)) == (
        0.75,
        3.0,
    )

    monkeypatch.setattr(
        report_primary_metric_analysis,
        "compute_primary_metric_from_report",
        lambda _report: {"preview": 2.0, "final": 3.0},
    )
    snapshot = report_primary_metric_analysis._resolve_primary_metric_snapshot(
        {}, {"primary_metric": {"final": 1.5}}
    )
    assert snapshot == (2.0, 3.0, 2.0, 1.5)


def test_primary_metric_analysis_rejects_deprecated_paired_summary() -> None:
    with pytest.raises(ValueError, match="paired_delta_summary is not supported"):
        report_primary_metric_analysis.build_primary_metric_analysis(
            {
                "metrics": {
                    "primary_metric": {"preview": 1.0, "final": 1.0},
                    "paired_delta_summary": {},
                }
            },
            {},
            {},
            {},
        )


def test_primary_metric_count_helpers_reject_fractional_and_count_real_ids() -> None:
    assert report_primary_metric_counts._as_count(1.5) is None
    assert report_primary_metric_counts._as_count("2") is None
    assert (
        report_primary_metric_counts._count_examples(
            {"records": ["noise"], "example_ids": ["a", "b"]}
        )
        == 2
    )
    assert report_primary_metric_counts._count_examples({"window_ids": [1]}) == 1
    assert (
        report_primary_metric_counts._classification_total(
            {"preview": {"total": 4}}, "preview"
        )
        == 4
    )


def test_primary_metric_counts_populate_fallback_coverage_paths() -> None:
    stats: dict = {}
    coverage = {"preview": {"used": 2}, "used": 3}
    paired = report_primary_metric_counts._populate_stats_with_counts_and_coverage(
        {"data": {"preview_n": 2, "final_n": 3}},
        {},
        coverage,
        {"stats": stats},
        paired_windows=2,
        paired_windows_explicit=True,
    )
    assert paired == 2
    assert stats["actual_preview"] == 2
    assert stats["actual_final"] == 3
    assert stats["paired_windows"] == 2
    assert stats["coverage_ok"] is True


class _BadFloat(float):
    def __float__(self):
        raise TypeError("bad float")


def test_report_summary_numeric_and_gate_status_fallbacks() -> None:
    assert math.isnan(report_summary._finite_float_or_nan(_BadFloat(1.0)))
    assert report_summary._coerce_finite_float(_BadFloat(1.0)) is None
    assert report_summary._pm_acceptance_range(
        {"meta": {"pm_acceptance_range": {"min": 0.9}}}
    ) == {"min": 0.9}
    assert report_summary._pm_acceptance_range(
        {"meta": {"pm_acceptance_range": {"max": 1.1}}}
    ) == {"max": 1.1}
    assert report_summary._format_gate_status(None, "gate") == "ℹ️ N/A"
    assert report_summary._format_gate_status({}, "gate", True) == "✅ PASS"
    assert report_summary._format_gate_status({"gate": "bad"}, "gate") == "ℹ️ N/A"


def test_report_summary_handles_accuracy_drift_and_tail_policy_edges() -> None:
    measured, threshold, basis = (
        report_summary._primary_metric_drift_measured_and_threshold(
            {"primary_metric": {"kind": "accuracy"}}
        )
    )
    assert measured == "N/A"
    assert threshold.startswith("≤ ±") and basis == "absolute-delta"

    quality = report_summary.build_quality_gates_summary(
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 2.0,
                "final": 2.2,
                "ratio_vs_baseline": 1.1,
                "drift_band": [0.8, 1.2],
            },
            "validation": {},
            "guard_metric_impact": {
                "evaluated": True,
                "degradation": 0.01,
                "display_value": 1.0,
                "display_unit": "percent",
                "degradation_limit": "bad",
            },
            "primary_metric_tail": {
                "evaluated": True,
                "passed": False,
                "mode": "fail",
                "policy": {
                    "quantile": "bad",
                    "quantile_max": 0.2,
                    "mass_max": 0.1,
                    "epsilon": 1e-4,
                },
                "stats": {"q95": 0.1, "tail_mass": 0.05},
            },
        }
    )
    by_label = {row.label: row for row in quality.rows}
    assert by_label["Guard Metric Impact Acceptable"].measured == "+1.00%"
    assert by_label["Guard Metric Impact Acceptable"].threshold == "N/A"
    assert by_label["Primary Metric Tail"].status == "❌ FAIL"
    assert "P95≤0.200" in by_label["Primary Metric Tail"].threshold


def test_report_manifest_summary_ignores_non_mapping_metric() -> None:
    summary = report_summary.build_report_manifest_summary(
        {"meta": "bad"},
        {"primary_metric": "bad", "validation": {}},
    )
    assert summary.run_model is None
    assert summary.primary_metric_ratio is None
