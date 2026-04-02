from __future__ import annotations

from copy import deepcopy

import pytest

import invarlock.reporting.report_normalization as report_normalization_mod
import invarlock.reporting.report_validation as report_validation_mod
from invarlock.reporting.report_make import make_report


def _mk_base_report() -> dict:
    return {
        "meta": {
            "model_id": "moe-test",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "noop",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (1.0, 1.0),
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_evaluation_report_includes_moe_families_when_present():
    report = _mk_base_report()
    baseline = _mk_base_report()
    # Inject spectral guard metrics with MoE families
    report["guards"].append(
        {
            "name": "spectral",
            "metrics": {
                "family_stats": {
                    "router": {
                        "count": 2,
                        "mean": 1.0,
                        "std": 0.1,
                        "min": 0.8,
                        "max": 1.2,
                    },
                    "expert_ffn": {
                        "count": 8,
                        "mean": 0.9,
                        "std": 0.2,
                        "min": 0.5,
                        "max": 1.1,
                    },
                },
                "family_caps": {"router": {"kappa": 2.5}, "expert_ffn": {"kappa": 2.5}},
            },
            "policy": {
                "family_caps": {"router": {"kappa": 2.5}, "expert_ffn": {"kappa": 2.5}}
            },
        }
    )
    # Inject RMT guard metrics with per-family outliers including MoE
    report["guards"].append(
        {
            "name": "rmt",
            "metrics": {
                "outliers_per_family": {"router": 1, "expert_ffn": 2},
                "baseline_outliers_per_family": {"router": 0, "expert_ffn": 1},
                "epsilon_by_family": {"router": 0.1, "expert_ffn": 0.1},
            },
            "policy": {
                "epsilon": {"attn": 0.08, "ffn": 0.1, "router": 0.1, "expert_ffn": 0.1}
            },
        }
    )

    evaluation_report = make_report(report, baseline)
    # Spectral families include MoE
    spectral_families = set(
        evaluation_report.get("spectral", {}).get("families", {}).keys()
    )
    assert {"router", "expert_ffn"}.issubset(spectral_families)
    # RMT families include MoE
    rmt_families = set(evaluation_report.get("rmt", {}).get("families", {}).keys())
    assert {"router", "expert_ffn"}.issubset(rmt_families)


def test_evaluation_report_moe_section_uses_normalized_baseline(monkeypatch):
    report = _mk_base_report()
    baseline = deepcopy(report)
    report["metrics"]["moe"] = {
        "load_balance_loss": 0.12,
        "router_entropy": 1.5,
        "utilization": [0.8, 0.9],
    }
    baseline["metrics"].pop("moe", None)
    normalized_baseline = {
        "run_id": "baseline-norm",
        "moe": {
            "load_balance_loss": 0.1,
            "router_entropy": 1.2,
            "utilization": [0.6, 0.7],
        },
    }

    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_baseline",
        lambda value: normalized_baseline,
        raising=False,
    )

    captured: dict[str, dict] = {}

    def _capture_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier,
        _ppl_metrics=None,
        target_ratio=None,
        guard_overhead=None,
        primary_metric=None,
        moe=None,
        dataset_capacity=None,
        pm_acceptance_range=None,
        pm_drift_band=None,
        pm_tail=None,
        tiny_relax=False,
        **_kwargs,
    ):
        captured["moe"] = moe or {}
        return {"primary_metric_acceptable": True}

    monkeypatch.setattr(
        report_validation_mod, "compute_validation_flags", _capture_flags, raising=False
    )

    make_report(report, baseline)
    moe_section = captured["moe"]
    assert moe_section["delta_load_balance_loss"] == pytest.approx(0.02)
    assert moe_section["delta_router_entropy"] == pytest.approx(0.3)
    assert "delta_utilization_mean" in moe_section
