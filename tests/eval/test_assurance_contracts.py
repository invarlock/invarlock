from __future__ import annotations

import math
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

# NOTE: import VarianceGuard only if it's part of the public surface;
# otherwise, drive it via evaluation_report inputs in an integration test.
from invarlock.core.auto_tuning import get_tier_policies
from invarlock.core.runner_runtime.pairing import BOOTSTRAP_COVERAGE_REQUIREMENTS
from invarlock.guards.spectral import SpectralGuard
from invarlock.guards.spectral_control import apply_relative_spectral_cap
from invarlock.guards.variance import VarianceGuard
from invarlock.reporting.guards_spectral import _extract_spectral_analysis
from invarlock.reporting.report_make import _extract_rmt_analysis, make_report
from invarlock.reporting.report_types import create_empty_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


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
    preview_ppl = math.exp(
        float(np.average([math.log(x) for x in preview_values], weights=weights))
    )
    final_ppl = math.exp(
        float(np.average([math.log(x) for x in final_values], weights=weights))
    )
    baseline_ppl = math.exp(
        float(np.average([math.log(x) for x in baseline_final], weights=weights))
    )

    report = {
        "meta": {
            "model_id": "gpt2-small",
            "adapter": "hf_causal",
            "device": "cpu",
            "ts": "2025-10-10T00:00:00",
            "commit": "deadbeefcafebabe",
            "seed": 1337,
            "seeds": {"python": 1337, "numpy": 4242, "torch": 777},
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
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
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": preview_ppl,
                "final": final_ppl,
                "ratio_vs_baseline": final_ppl / baseline_ppl,
            },
            "logloss_delta": wmean,
            "logloss_delta_ci": (wmean - 0.01, wmean + 0.01),
            "preview_final_slice_delta_summary": {
                "mean": wmean,
                "ci": [wmean - 0.01, wmean + 0.01],
                "basis": "independent_disjoint_slices",
                "paired": False,
                "ci_method": "independent_percentile_delta_log",
                "ci_reason": None,
                "preview_windows": len(preview_values),
                "final_windows": len(final_values),
                "degenerate": False,
                "degenerate_reason": None,
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
        "meta": {
            "model_id": "gpt2-small",
            "adapter": "hf_causal",
            "seed": 1337,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 768,
            "stride": 768,
            "preview_n": len(preview_values),
            "final_n": len(final_values),
        },
        "edit": {"name": "noop"},
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": baseline_ppl,
                "final": baseline_ppl,
            }
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [0, 1],
                "logloss": [math.log(x) for x in baseline_final],
                "token_counts": list(token_counts),
            },
            "final": {
                "window_ids": [0, 1],
                "logloss": [math.log(x) for x in baseline_final],
                "token_counts": list(token_counts),
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    return report, baseline


def _make_canonical_evaluation_report(
    report: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Build from current RunReport fixtures with an exact policy receipt."""

    return make_report(canonical_run_report(report), canonical_baseline(baseline))


def test_bootstrap_coverage_floors_match_assurance_docs():
    assert BOOTSTRAP_COVERAGE_REQUIREMENTS == {
        "conservative": {"preview": 220, "final": 220, "replicates": 1500},
        "balanced": {"preview": 180, "final": 180, "replicates": 1200},
        "aggressive": {"preview": 140, "final": 140, "replicates": 800},
    }


def test_evaluation_report_enforces_paired_ratio_identity():
    report, baseline = _build_paired_run_and_baseline()
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        evaluation_report = _make_canonical_evaluation_report(
            deepcopy(report), deepcopy(baseline)
        )
    delta_mean = report["metrics"]["preview_final_slice_delta_summary"]["mean"]
    expected_ratio = math.exp(delta_mean)
    pm = evaluation_report.get("primary_metric", {})
    assert math.isclose(
        pm.get("final") / pm.get("preview"), expected_ratio, rel_tol=1e-3
    )
    # CI should be present via display_ci and reflect exp(Δlog bounds)
    ratio_ci = pm.get("display_ci")
    assert isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2


def test_evaluation_report_flags_inconsistent_ratio_gate_failure_in_dev_profile():
    report, baseline = _build_paired_run_and_baseline()
    report["metrics"]["preview_final_slice_delta_summary"]["mean"] += (
        0.1  # Break consistency
    )
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        # Use dev profile to avoid strict pairing enforcement; inconsistency should not raise
        report.setdefault("metrics", {}).setdefault("window_plan", {})["profile"] = (
            "dev"
        )
        cert = _make_canonical_evaluation_report(report, baseline)
    assert isinstance(cert, dict)
    assert cert["validation"]["primary_metric_acceptable"] is False
    assert cert["validation"]["preview_final_drift_acceptable"] is False
    assert cert["primary_metric"]["invalid"] is False


def _make_ratio_report(
    preview: float, final: float, tier: str = "balanced"
) -> dict[str, Any]:
    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
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
    report["context"] = {"profile": "dev"}
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
    baseline["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "auto": {"tier": "balanced"},
        }
    )
    baseline["context"] = {"profile": "dev"}
    baseline["data"].update(
        {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 128,
            "stride": 128,
        }
    )
    baseline["edit"]["name"] = "noop"
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

    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        passing_cert = _make_canonical_evaluation_report(
            deepcopy(passing_report), deepcopy(baseline)
        )
        failing_cert = _make_canonical_evaluation_report(
            deepcopy(failing_report), deepcopy(baseline)
        )

    assert passing_cert["validation"]["primary_metric_acceptable"] is True
    assert failing_cert["validation"]["primary_metric_acceptable"] is False


def test_seed_bundle_contract():
    report, baseline = _build_paired_run_and_baseline()
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        evaluation_report = _make_canonical_evaluation_report(report, baseline)
    # Evaluation Report preserves the full seed bundle for auditability.
    assert evaluation_report["meta"]["seeds"] == {
        "python": 1337,
        "numpy": 4242,
        "torch": 777,
    }
    stats = evaluation_report.get("dataset", {}).get("windows", {}).get("stats", {})
    assert stats.get("window_match_fraction") == 1.0
    assert stats.get("window_overlap_fraction") == 0.0


def test_evaluation_report_rejects_ci_runs_below_bootstrap_floor():
    report, baseline = _build_paired_run_and_baseline()
    report["data"]["preview_n"] = 180
    report["data"]["final_n"] = 180
    report["metrics"]["window_plan"] = {
        "profile": "ci",
        "preview_n": 180,
        "final_n": 180,
    }
    report["metrics"]["stats"] = {
        "requested_preview": 180,
        "requested_final": 180,
        "actual_preview": 10,
        "actual_final": 10,
    }
    report["metrics"]["bootstrap"]["coverage"] = {
        "preview": {"used": 10},
        "final": {"used": 10},
        "replicates": {"used": 100},
    }
    report["metrics"]["bootstrap"]["replicates"] = 100
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        with pytest.raises(ValueError):
            _make_canonical_evaluation_report(deepcopy(report), deepcopy(baseline))


def _apply_ci_pairing_requirements(report: dict[str, Any]) -> None:
    report["data"]["preview_n"] = 180
    report["data"]["final_n"] = 180
    report.setdefault("metrics", {}).setdefault("window_plan", {}).update(
        {"profile": "ci", "preview_n": 180, "final_n": 180}
    )
    report["metrics"]["stats"] = {
        "requested_preview": 180,
        "requested_final": 180,
        "actual_preview": 180,
        "actual_final": 180,
    }
    report["metrics"].setdefault("bootstrap", {}).update(
        {
            "replicates": 1200,
            "coverage": {
                "preview": {"used": 180},
                "final": {"used": 180},
                "replicates": {"used": 1200},
            },
        }
    )


def test_evaluation_report_rejects_ci_overlap():
    report, baseline = _build_paired_run_and_baseline()
    _apply_ci_pairing_requirements(report)
    report["metrics"]["window_overlap_fraction"] = 0.25
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        with pytest.raises(ValueError):
            _make_canonical_evaluation_report(deepcopy(report), deepcopy(baseline))


def test_evaluation_report_rejects_ci_pairing_mismatch():
    report, baseline = _build_paired_run_and_baseline()
    _apply_ci_pairing_requirements(report)
    report["metrics"]["window_match_fraction"] = 0.98
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        with pytest.raises(ValueError):
            _make_canonical_evaluation_report(deepcopy(report), deepcopy(baseline))


def test_spectral_cap_product_path_limits_weight_growth():
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.diag(torch.tensor([4.0, 1.0])))
    model = torch.nn.Sequential(module)

    result = apply_relative_spectral_cap(
        model,
        cap_ratio=1.1,
        baseline_sigmas={"0": 1.0},
        should_process_module_fn=lambda name, _module, _scope: name == "0",
    )

    assert result["applied"] is True
    assert result["capped_modules"][0]["module"] == "0"
    sigma_after = float(torch.linalg.svdvals(module.weight).max().item())
    assert sigma_after <= 1.1 + 1e-6


def test_spectral_fpr_matches_tail_probabilities():
    policies = get_tier_policies()
    balanced = policies["balanced"]["spectral"]
    conservative = policies["conservative"]["spectral"]
    expected_balanced_caps = {
        "ffn": 3.849,
        "attn": 3.018,
        "embed": 1.05,
        "other": 0.0,
    }
    expected_conservative_caps = {
        "ffn": 3.849,
        "attn": 2.6,
        "embed": 2.8,
        "other": 2.8,
    }

    assert balanced["multiple_testing"] == {"method": "bh", "alpha": 0.05, "m": 4}
    assert conservative["multiple_testing"] == {
        "method": "bonferroni",
        "alpha": 0.000625,
        "m": 4,
    }
    assert balanced["scope"] == "all"
    assert conservative["scope"] == "ffn"
    assert balanced["max_caps"] == 5
    assert conservative["max_caps"] == 3

    for family, expected in expected_balanced_caps.items():
        assert balanced["family_caps"][family]["kappa"] == pytest.approx(expected)
    for family, expected in expected_conservative_caps.items():
        assert conservative["family_caps"][family]["kappa"] == pytest.approx(expected)

    guard = SpectralGuard(**balanced)
    assert guard.multiple_testing == balanced["multiple_testing"]
    for family, expected in expected_balanced_caps.items():
        assert guard.family_caps[family]["kappa"] == pytest.approx(expected)

    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "spectral",
                "policy": balanced,
                "metrics": {
                    "caps_applied": 0,
                    "family_caps": balanced["family_caps"],
                    "families": {
                        family: {"kappa": cap["kappa"], "violations": 0}
                        for family, cap in balanced["family_caps"].items()
                    },
                    "multiple_testing": balanced["multiple_testing"],
                },
            }
        ],
        "metrics": {},
    }
    spectral = _extract_spectral_analysis(report, {})
    assert spectral["multiple_testing"]["method"] == "bh"
    assert spectral["multiple_testing"]["alpha"] == pytest.approx(0.05)
    for family, expected in expected_balanced_caps.items():
        assert spectral["family_caps"][family]["kappa"] == pytest.approx(expected)

    rng = np.random.default_rng(123)
    samples = rng.standard_normal(200_000)
    tail_by_balanced_family = {}
    kappas = {
        "reference": 2.5,
        **{
            f"balanced_{family}": cap["kappa"]
            for family, cap in balanced["family_caps"].items()
        },
        **{
            f"conservative_{family}": cap["kappa"]
            for family, cap in conservative["family_caps"].items()
        },
    }
    for label, kappa in kappas.items():
        empirical = np.mean(np.abs(samples) >= kappa)
        theoretical = 2 * (1.0 - 0.5 * (1.0 + math.erf(kappa / math.sqrt(2))))
        assert abs(empirical - theoretical) < 0.01
        if label.startswith("balanced_"):
            tail_by_balanced_family[label.removeprefix("balanced_")] = theoretical

    assert tail_by_balanced_family["ffn"] < balanced["multiple_testing"]["alpha"]
    assert tail_by_balanced_family["attn"] < balanced["multiple_testing"]["alpha"]
    assert tail_by_balanced_family["embed"] > balanced["multiple_testing"]["alpha"]
    assert tail_by_balanced_family["other"] == pytest.approx(1.0)


def test_rmt_epsilon_rule_acceptance_band():
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "rmt",
                "metrics": {
                    "edge_risk_by_family_base": {"ffn": 1.0},
                    "edge_risk_by_family": {"ffn": 1.09},
                    "epsilon_by_family": {"ffn": 0.10},
                },
                "policy": {"deadband": 0.10},
            }
        ],
        "metrics": {},
    }
    baseline = {"rmt": {}}
    result = _extract_rmt_analysis(report, baseline)
    assert result["stable"]
    report["guards"][0]["metrics"]["edge_risk_by_family"]["ffn"] = 1.21
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
    guard.set_ab_results(
        ppl_no_ve=51.0,
        ppl_with_ve=50.9745,
        windows_used=8,
        seed_used=123,
        ratio_ci=(0.90, 0.998),
    )
    should_enable, reason = guard._evaluate_ab_gate()
    assert not should_enable
    assert ("below_min_effect_lognll" in reason) or (
        "below_threshold_with_deadband" in reason
    )

    guard = VarianceGuard(policy=_make_variance_policy(min_effect_lognll=0.0005))
    guard.set_ab_results(
        ppl_no_ve=51.0,
        ppl_with_ve=50.9235,
        windows_used=8,
        seed_used=123,
        ratio_ci=(0.90, 0.995),
    )
    should_enable, reason = guard._evaluate_ab_gate()
    assert should_enable, reason
