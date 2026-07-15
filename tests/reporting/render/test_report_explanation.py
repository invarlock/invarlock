from __future__ import annotations

import math
from types import SimpleNamespace

from invarlock.reporting import report_explanation as mod


def _no_quality_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "build_quality_gates_summary",
        lambda _report: SimpleNamespace(rows=[]),
    )


def test_report_explanation_helper_edges() -> None:
    assert mod._coerce_optional_float(math.nan) is None
    assert mod._coerce_optional_float("bad") is None
    assert mod._coerce_int("bad", default=7) == 7
    assert mod._dataset_split_line({"provenance": {}}) is None
    assert (
        mod._dataset_split_line(
            {"provenance": {"dataset_split": "validation", "split_fallback": True}}
        )
        == "Dataset split: validation (fallback)"
    )
    assert mod._drift_ratio("bad", 1.0) is None
    assert mod._drift_ratio(0.0, 1.0) is None


def test_report_explanation_formats_raw_policy_and_guard_edges(monkeypatch) -> None:
    _no_quality_rows(monkeypatch)
    report = {
        "auto": {"tier": "balanced"},
        "validation": {
            "primary_metric_acceptable": False,
            "preview_final_drift_acceptable": False,
            "hysteresis_applied": True,
            "spectral_stable": False,
            "rmt_stable": False,
            "guard_metric_impact_acceptable": False,
        },
        "resolved_policy": {
            "metrics": {
                "pm_ratio": {
                    "ratio_limit_base": 1.05,
                    "hysteresis_ratio": 0.01,
                    "min_tokens": 100,
                }
            }
        },
        "telemetry": {"preview_total_tokens": 40, "final_total_tokens": 20},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 2.0,
            "final": 3.0,
            "ratio_vs_baseline": 1.2,
            "display_ci": [1.1, 1.3],
        },
        "primary_metric_tail": {
            "evaluated": True,
            "passed": True,
            "mode": "warn",
            "policy": {
                "quantile": 0.9,
                "quantile_max": 0.4,
                "mass_max": 0.2,
                "epsilon": 1e-6,
            },
            "stats": {"q90": 0.3, "tail_mass": 0.1},
        },
        "spectral": {"caps_applied": "n/a"},
        "rmt": {"status": "warn"},
        "guard_warnings": {
            "warnings": [
                "bad-entry",
                {"guard": "spectral", "kind": "z_shift", "policy_gate": "pass"},
            ]
        },
        "guard_metric_impact": {
            "evaluated": True,
            "degradation_limit": 0.02,
            "degradation": 0.035,
            "display_value": 3.5,
            "display_unit": "percent",
        },
    }

    text = "\n".join(
        mod.render_evaluation_report_explanation_lines(
            report,
            report_payload={
                "provenance": {
                    "dataset_split": "validation",
                    "split_fallback": True,
                }
            },
            tier_policies_getter=lambda: {
                "balanced": {"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}}
            },
        )
    )

    assert "status: FAIL" in text
    assert "observed: 1.200x (CI 1.100-1.300)" in text
    assert "threshold: ≤ 1.05x (+hysteresis 0.010)" in text
    assert "tokens: below floor" in text
    assert "effective threshold = 1.060x" in text
    assert "status: PASS" in text
    assert "threshold: P90≤0.4000; mass≤0.2000; ε=1.0e-06" in text
    assert "Dataset split: validation (fallback)" in text
    assert "observed: 1.500" in text
    assert "observed: caps not recorded" in text
    assert "threshold: resolved tier max_caps" in text
    assert "observed: warn" in text
    assert "spectral.z_shift; policy: pass" in text
    assert "observed: +3.50%" in text
    assert "threshold: ≤ +2.0%" in text


def test_report_explanation_tail_fail_and_rmt_na(monkeypatch) -> None:
    _no_quality_rows(monkeypatch)
    report = {
        "auto": {"tier": "balanced"},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "rmt_stable": True,
        },
        "primary_metric": {"kind": "accuracy", "delta_vs_baseline_pp": 0.0},
        "primary_metric_tail": {
            "evaluated": True,
            "passed": False,
            "mode": "fail",
            "policy": {},
            "stats": {},
        },
        "rmt": {"edge_risk": 0.1},
        "guard_metric_impact": {
            "evaluated": True,
            "degradation_limit": 0.04,
            "degradation": 0.01,
            "display_value": 1.0,
            "display_unit": "percent",
        },
    }

    text = "\n".join(
        mod.render_evaluation_report_explanation_lines(
            report,
            tier_policies_getter=lambda: {"balanced": {"metrics": {}}},
        )
    )

    assert "status: FAIL" in text
    assert "observed: N/A" in text
    assert "threshold: ≤ +4.0%" in text


def test_report_explanation_numeric_guard_counts_and_plain_split(
    monkeypatch,
) -> None:
    _no_quality_rows(monkeypatch)
    report = {
        "auto": {"tier": "balanced"},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "provenance": {"dataset_split": "test"},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 0.0,
            "final": 1.0,
            "ratio_vs_baseline": 1.0,
        },
        "spectral": {"caps_applied": 2, "max_caps": 5},
        "rmt": {"epsilon_violations": [{"family": "ffn"}]},
    }

    text = "\n".join(
        mod.render_evaluation_report_explanation_lines(
            report,
            tier_policies_getter=lambda: {"balanced": {"metrics": {}}},
        )
    )

    assert "Dataset split: test" in text
    assert "observed: 2 caps applied" in text
    assert "threshold: <= 5 caps" in text
    assert "observed: 1 epsilon violations" in text
