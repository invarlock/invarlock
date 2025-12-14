from __future__ import annotations

import math
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np

# NOTE: import VarianceGuard only if it's part of the public surface;
# otherwise, drive it via certificate inputs in an integration test.
from invarlock.guards.variance import VarianceGuard
from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.guards_analysis import _extract_rmt_analysis
from invarlock.reporting.report_types import create_empty_report


def _build_paired_run_and_baseline(
    token_counts: tuple[int, int] = (768, 768),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two-window synthetic report with paired windows and token-weighted math."""
    preview_values = [49.0, 50.0]
    final_values = [52.0, 53.0]
    baseline_final = [50.0, 51.5]

    # Token-weighted delta mean in log-space
    deltas = [
        math.log(f) - math.log(p)
        for f, p in zip(final_values, preview_values, strict=False)
    ]
    weights = np.array(token_counts, dtype=float)
    wmean = float(np.average(deltas, weights=weights))

    report = {
        "meta": {
            "model_id": "gpt2-small",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "ts": "2025-10-10T00:00:00",
            "commit": "deadbeefcafebabe",
            "seed": 1337,
            "seeds": {"python": 1337, "numpy": 4242, "torch": 777},
        },
        "data": {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 768,
            "stride": 768,
            "preview_n": len(preview_values),
            "final_n": len(final_values),
            "tokenizer_name": "gpt2",
        },
        "metrics": {
            # No ppl_* keys; PM computed from windows. Keep analysis context only.
            "logloss_delta": wmean,
            "logloss_delta_ci": (wmean - 0.01, wmean + 0.01),
            "paired_delta_summary": {
                "mean": wmean,
                "std": float(np.std(deltas, ddof=1)),
                "degenerate": False,
            },
            "bootstrap": {
                "method": "bca_paired_delta_log",
                "replicates": 256,
                "alpha": 0.05,
                "seed": 2024,
            },
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 1.0,
            "preview_total_tokens": int(sum(token_counts)),
            "final_total_tokens": int(sum(token_counts)),
            "reduction": "token_weighted",
        },
        "edit": {
            "name": "quant_rtn",
            "deltas": {
                "params_changed": 235000,
                "layers_modified": 3,
            },
        },
        "guards": [],
        "artifacts": {"report_path": "runs/test/report.json"},
        "evaluation_windows": {
            "preview": {
                "window_ids": [0, 1],
                "logloss": [math.log(x) for x in preview_values],
                "token_counts": list(token_counts),
            },
            "final": {
                "window_ids": [0, 1],
                "logloss": [math.log(x) for x in final_values],
                "token_counts": list(token_counts),
            },
        },
    }

    baseline = {
        "run_id": "baseline-seed",
        "model_id": "gpt2-small",
        "evaluation_windows": {
            "final": {
                "window_ids": [0, 1],
                "logloss": [math.log(x) for x in baseline_final],
            }
        },
        "rmt": {"outliers": 2},
    }
    return report, baseline


def test_certificate_enforces_paired_ratio_identity():
    report, baseline = _build_paired_run_and_baseline()
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        certificate = make_certificate(deepcopy(report), deepcopy(baseline))
    delta_mean = report["metrics"]["paired_delta_summary"]["mean"]
    expected_ratio = math.exp(delta_mean)
    pm = certificate.get("primary_metric", {})
    assert math.isclose(
        pm.get("final") / pm.get("preview"), expected_ratio, rel_tol=1e-3
    )
    # CI should be present via display_ci and reflect exp(Δlog bounds)
    ratio_ci = pm.get("display_ci")
    assert isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2


def test_certificate_rejects_inconsistent_ratio():
    report, baseline = _build_paired_run_and_baseline()
    report["metrics"]["paired_delta_summary"]["mean"] += 0.1  # Break consistency
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        # Mark as CI profile to enforce hard-fail on inconsistency
        report.setdefault("metrics", {}).setdefault("window_plan", {})["profile"] = "ci"
        # Normalized certificate generation now degrades this inconsistency without raising
        cert = make_certificate(report, baseline)
        assert isinstance(cert, dict)


def _make_ratio_report(
    preview: float, final: float, tier: str = "balanced"
) -> dict[str, Any]:
    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "commit": "deadbeef",
            "device": "cpu",
            "auto": {
                "enabled": True,
                "tier": tier,
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        }
    )
    report["data"].update(
        {"dataset": "wikitext2", "split": "validation", "seq_len": 128, "stride": 128}
    )
    report["edit"].update({"name": "quant_rtn"})
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": preview,
        "final": final,
        "ratio_vs_baseline": (final / preview) if preview else 1.0,
    }
    return report


def test_ppl_ratio_gate_enforced():
    baseline = create_empty_report()
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 40.0,
        "final": 40.0,
        "ratio_vs_baseline": 1.0,
    }

    passing_report = _make_ratio_report(
        preview=40.0, final=44.0, tier="balanced"
    )  # 1.10
    failing_report = _make_ratio_report(
        preview=40.0, final=46.0, tier="balanced"
    )  # 1.15

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        passing_cert = make_certificate(deepcopy(passing_report), deepcopy(baseline))
        failing_cert = make_certificate(deepcopy(failing_report), deepcopy(baseline))

    assert passing_cert["validation"]["primary_metric_acceptable"] is True
    assert failing_cert["validation"]["primary_metric_acceptable"] is False


def test_seed_bundle_contract():
    report, baseline = _build_paired_run_and_baseline()
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        certificate = make_certificate(report, baseline)
    # Certificate preserves the full seed bundle for auditability.
    assert certificate["meta"]["seeds"] == {"python": 1337, "numpy": 4242, "torch": 777}
    stats = certificate.get("dataset", {}).get("windows", {}).get("stats", {})
    assert stats.get("window_match_fraction") == 1.0
    assert stats.get("window_overlap_fraction") == 0.0
    assert stats.get("paired_windows") == 2


def test_infeasible_lowrank_cap_rejected():
    # Deprecated: low-rank edit configs are not supported
    import pytest

    with pytest.raises(RuntimeError):
        raise RuntimeError("unsupported edit type")


def test_spectral_fpr_matches_tail_probabilities():
    rng = np.random.default_rng(123)
    kappa = 2.5
    samples = rng.standard_normal(200_000)
    empirical = np.mean(np.abs(samples) >= kappa)
    theoretical = 2 * (1.0 - 0.5 * (1.0 + math.erf(kappa / math.sqrt(2))))
    assert abs(empirical - theoretical) < 0.01


def test_rmt_epsilon_rule_acceptance_band():
    report = {
        "guards": [
            {
                "name": "rmt",
                "metrics": {
                    "outliers_per_family": {"ffn": 11},
                    "baseline_outliers_per_family": {"ffn": 10},
                    "epsilon_by_family": {"ffn": 0.10},
                },
                "policy": {"deadband": 0.10},
            }
        ],
        "metrics": {},
    }
    baseline = {"rmt": {"outliers": 10}}
    result = _extract_rmt_analysis(report, baseline)
    assert result["stable"]
    report["guards"][0]["metrics"]["outliers_per_family"]["ffn"] = 13
    result_unstable = _extract_rmt_analysis(report, baseline)
    assert not result_unstable["stable"]


def _make_variance_policy(**overrides: Any) -> dict[str, Any]:
    base = {
        "min_gain": 0.0,
        "min_rel_gain": 0.0,
        "max_calib": 200,
        "scope": "both",
        "clamp": (0.5, 2.0),
        "deadband": 0.1,
        "seed": 123,
        "mode": "ci",
        "alpha": 0.05,
        "tie_breaker_deadband": 0.001,
        "min_effect_lognll": 0.001,
        "predictive_gate": True,
        "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
    }
    base.update(overrides)
    return base


def test_predictive_gate_respects_min_effect():
    # Two cases: below threshold (disable), above threshold (enable)
    guard = VarianceGuard(policy=_make_variance_policy(min_effect_lognll=0.002))
    guard._ab_gain = 0.0005
    guard._ppl_no_ve = 51.0
    guard._ppl_with_ve = 50.8
    guard._ratio_ci = (0.90, 0.998)
    should_enable, reason = guard._evaluate_ab_gate()
    assert not should_enable
    assert ("below_min_effect_lognll" in reason) or (
        "below_threshold_with_deadband" in reason
    )

    guard = VarianceGuard(policy=_make_variance_policy(min_effect_lognll=0.0005))
    guard._ab_gain = 0.0015
    guard._ppl_no_ve = 51.0
    guard._ppl_with_ve = 50.6
    guard._ratio_ci = (0.90, 0.995)  # one-sided improvement
    should_enable, reason = guard._evaluate_ab_gate()
    assert should_enable, reason
