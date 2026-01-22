"""Integration coverage for variance guard toggle behaviour."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from invarlock.reporting.certificate import make_certificate

BASELINE_PPL_FINAL = 46.5
PPL_PREVIEW = 46.8
PPL_FINAL = 47.0


def _build_run_report(
    tier: str, ve_enabled: bool, params_changed: int
) -> dict[str, Any]:
    """Construct a minimal RunReport matching the schema validator."""
    ppl_ratio = PPL_FINAL / BASELINE_PPL_FINAL

    variance_metrics: dict[str, Any] = {
        "ve_enabled": ve_enabled,
        "tap": "transformer.h.*.mlp.c_proj",
        "target_modules": [
            "transformer.h.4.mlp.c_proj",
            "transformer.h.7.mlp.c_proj",
            "transformer.h.9.mlp.c_proj",
        ],
        "focus_modules": ["transformer.h.4.mlp.c_proj"],
        "proposed_scales": [0.97] if ve_enabled else [],
        "proposed_scales_pre_edit": [1.0, 1.0, 1.0],
        "proposed_scales_post_edit": [0.97, 1.0, 1.0],
        "predictive_gate": {
            "evaluated": True,
            "passed": ve_enabled,
            "reason": "ci_gain_met" if ve_enabled else "ci_contains_zero",
            "delta_ci": [-0.0012, 0.0008],
            "gain_ci": [-0.0008, 0.0012],
            "mean_delta": -0.0002,
        },
        "ab_seed_used": 1337,
        "ab_windows_used": 16,
        "ab_provenance": {"baseline": "baseline-run"},
        "ab_point_estimates": {"no_ve": 53.24, "with_ve": 53.11},
        "monitor_only": not ve_enabled,
        "scope": "ffn",
        "mode": "delta",
        "min_rel_gain": 0.0005,
        "alpha": 0.05,
    }

    report: dict[str, Any] = {
        "meta": {
            "model_id": "gpt2-small",
            "adapter": "hf_causal",
            "commit": "abcdef1234567890",
            "seed": 42,
            "device": "cpu",
            "ts": "2025-10-10T10:00:00",
            "auto": {
                "enabled": False,
                "tier": tier,
                "probes": 0,
                "target_pm_ratio": 1.10,
            },
        },
        "data": {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 768,
            "stride": 768,
            "preview_n": 200,
            "final_n": 200,
            "tokenizer_name": "gpt2",
            "tokenizer_hash": "tokhash",
            "vocab_size": 50257,
            "bos_token": "Ġ",
            "eos_token": "",
            "pad_token": "",
            "add_prefix_space": False,
        },
        "edit": {
            "name": "lowrank_svd",
            "plan": {},
            "deltas": {
                "params_changed": params_changed,
                "layers_modified": 3 if params_changed else 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
            },
        },
        "guards": [
            {
                "name": "variance",
                "policy": {
                    "deadband": 0.02,
                    "min_abs_adjust": 0.012,
                    "max_scale_step": 0.03,
                },
                "metrics": variance_metrics,
                "actions": [],
                "violations": [],
                "details": {},
            }
        ],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": PPL_PREVIEW,
                "final": PPL_FINAL,
                "ratio_vs_baseline": ppl_ratio,
            },
            "invariants": {"status": "pass"},
            "spectral": {},
            "rmt": {},
        },
        "artifacts": {
            "events_path": "/tmp/events.jsonl",
            "report_path": "/tmp/report.json",
        },
        "flags": {},
    }

    return report


def _build_baseline_report() -> dict[str, Any]:
    baseline = _build_run_report("balanced", False, params_changed=0)
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": PPL_PREVIEW,
        "final": BASELINE_PPL_FINAL,
        "ratio_vs_baseline": 1.0,
    }
    baseline["edit"]["deltas"]["params_changed"] = 0
    baseline["guards"][0]["metrics"]["proposed_scales"] = []
    baseline["guards"][0]["metrics"]["predictive_gate"]["passed"] = False
    baseline["guards"][0]["metrics"]["predictive_gate"]["reason"] = "baseline_monitor"
    return baseline


def test_variance_toggle_balanced_vs_conservative():
    baseline = _build_baseline_report()

    balanced_report = _build_run_report(
        "balanced", ve_enabled=True, params_changed=2360064
    )
    conservative_report = _build_run_report(
        "conservative", ve_enabled=False, params_changed=2360064
    )

    balanced_cert = make_certificate(deepcopy(balanced_report), deepcopy(baseline))
    conservative_cert = make_certificate(
        deepcopy(conservative_report), deepcopy(baseline)
    )

    assert balanced_cert["auto"]["tier"] == "balanced"
    assert balanced_cert["variance"]["enabled"] is True
    assert balanced_cert["variance"]["proposed_scales"]
    assert balanced_cert["variance"]["predictive_gate"]["evaluated"] is True
    assert balanced_cert["variance"]["predictive_gate"]["reason"] == "ci_gain_met"

    assert conservative_cert["auto"]["tier"] == "conservative"
    assert conservative_cert["variance"]["enabled"] is False
    assert not conservative_cert["variance"].get("proposed_scales")
    assert (
        conservative_cert["variance"]["predictive_gate"]["reason"] == "ci_contains_zero"
    )
