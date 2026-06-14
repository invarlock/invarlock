from __future__ import annotations

import json
from pathlib import Path

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_outline import build_evaluation_report_outline


def _mk_report() -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "2026-06-14T12:00:00Z",
        },
        "data": {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 16,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "noop",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "unit": "ppl",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            }
        },
        "evaluation_windows": {
            "preview": {"window_ids": [1, 2], "logloss": [2.3, 2.3]},
            "final": {"window_ids": [3, 4], "logloss": [2.3, 2.3]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def _section(outline, key: str):
    for section in outline.sections:
        if section.key == key:
            return section
    raise AssertionError(f"missing section {key!r}")


def test_report_outline_orders_modern_sections_before_appendix() -> None:
    cert = make_report(_mk_report(), _mk_report())
    cert["guard_warnings"] = {
        "present": True,
        "warning_count": 1,
        "warnings": [
            {
                "guard": "spectral",
                "kind": "new_capped_module",
                "severity": "warning",
                "module": "layers.0.mlp.up_proj",
                "policy_gate": "pass",
                "message": "Subject introduced a new capped module.",
            }
        ],
    }
    cert["validation"]["guard_warnings_present"] = True
    cert["validation"]["guard_warning_policy_acceptable"] = True
    cert["primary_metric_tail"] = {
        "evaluated": True,
        "passed": True,
        "warned": False,
        "stats": {"q95": 0.01, "tail_mass": 0.0},
    }
    cert["baseline_ref"] = {
        "model_id": "gpt2",
        "run_id": "baseline-run-1234567890",
    }
    cert["benchmark_comparison"] = {
        "schema_version": "bench-v1",
        "profile": "ci",
        "scenarios": [
            {
                "edit": "quant_rtn",
                "tier": "balanced",
                "skip": False,
                "primary_metric_overhead": 0.009,
                "guard_overhead_time": 0.13,
                "guard_overhead_mem": 0.09,
                "rmt_outliers_bare": 2,
                "rmt_outliers_guarded": 3,
                "pass": {
                    "spike": True,
                    "tying": True,
                    "rmt": True,
                    "quality": True,
                    "time": True,
                    "mem": True,
                },
            }
        ],
    }

    outline = build_evaluation_report_outline(cert)

    assert [section.key for section in outline.sections] == [
        "decision",
        "primary_metric",
        "policy_gates",
        "guard_signals",
        "benchmark_comparison",
        "evidence_provenance",
        "technical_appendix",
    ]
    assert _section(outline, "decision").facts_by_label["Guard Warnings"].value == "1"
    assert (
        _section(outline, "decision").facts_by_label["Baseline"].value
        == "gpt2 · run baseline…34567890"
    )
    assert (
        _section(outline, "primary_metric").facts_by_label["Tail Gate"].value == "PASS"
    )
    assert (
        _section(outline, "benchmark_comparison")
        .facts_by_label["Scenarios"]
        .value
        == "1 total, 1 passed, 0 skipped"
    )


def test_report_outline_summarizes_multimodal_accuracy_without_ppl_language() -> None:
    cert = make_report(_mk_report(), _mk_report())
    cert["meta"]["adapter"] = "hf_multimodal"
    cert["primary_metric"] = {
        "kind": "accuracy",
        "unit": "accuracy",
        "preview": 0.86,
        "final": 0.855,
        "ratio_vs_baseline": 0.0,
        "baseline_point": 0.855,
        "display_ci": [-0.01, 0.01],
    }

    outline = build_evaluation_report_outline(cert)
    primary = _section(outline, "primary_metric")

    assert primary.facts_by_label["Metric"].value == "accuracy"
    assert primary.facts_by_label["Baseline Comparison"].value == "+0.00 pp"
    assert "ppl" not in primary.summary.lower()


def test_report_outline_uses_policy_threshold_not_auto_target_ratio() -> None:
    cert = make_report(_mk_report(), _mk_report())
    cert.setdefault("auto", {})["tier"] = "balanced"
    cert["auto"]["target_pm_ratio"] = 1.0
    cert["primary_metric"]["ratio_vs_baseline"] = 1.48
    cert["validation"]["primary_metric_acceptable"] = False

    outline = build_evaluation_report_outline(cert)
    policy_gates = _section(outline, "policy_gates")

    assert (
        policy_gates.facts_by_label["Primary Metric Acceptable"].value
        == "FAIL | 1.480x vs ≤ 1.10x"
    )


def test_report_outline_omits_benchmark_section_when_absent() -> None:
    cert = make_report(_mk_report(), _mk_report())

    outline = build_evaluation_report_outline(cert)

    assert "benchmark_comparison" not in [section.key for section in outline.sections]


def test_report_outline_accepts_step14_benchmark_fixture() -> None:
    cert = make_report(_mk_report(), _mk_report())
    fixture = Path("tests/fixtures/benchmarks/guard_effect_golden.json")
    cert["benchmark_comparison"] = json.loads(fixture.read_text(encoding="utf-8"))

    outline = build_evaluation_report_outline(cert)
    benchmark = _section(outline, "benchmark_comparison")

    assert benchmark.facts_by_label["Profile"].value == "ci"
    assert benchmark.facts_by_label["Scenarios"].value == "2 total, 2 passed, 0 skipped"
    assert benchmark.facts_by_label["Primary Metric Overhead"].value == "0.8% to 0.9%"
    assert benchmark.facts_by_label["RMT Outliers"].value == "2->3"
