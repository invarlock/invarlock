from __future__ import annotations

import math
from copy import deepcopy


def _rich_run_report() -> tuple[dict, dict]:
    window_ids = list(range(1, 181))
    token_counts = [10] * len(window_ids)
    report = {
        "meta": {
            "model_id": "demo-model",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 7,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00Z",
            "auto": {
                "tier": "balanced",
                "probes_used": ["spectral"],
                "target_pm_ratio": 1.1,
            },
        },
        "data": {
            "dataset": "demo-ds",
            "split": "eval",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 180,
            "final_n": 180,
            "windows": {"preview": 180, "final": 180, "seed": 7},
        },
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "plan123",
            "config": {"scope": "ffn", "seed": 7, "frac": 0.5, "clamp_ratio": 0.2},
            "deltas": {
                "params_changed": 10,
                "layers_modified": 2,
                "bitwidth_map": {
                    "layer1": {"bitwidth": 4, "group_size": None, "params": 512},
                    "layer2": {"bitwidth": 8, "group_size": 32, "params": 256},
                },
                "rank_map": {
                    "layer1": {
                        "rank": 8,
                        "params_saved": 128,
                        "energy_retained": 0.95,
                        "deploy_mode": "recompose",
                        "savings_mode": "realized",
                        "realized_params_saved": 64,
                        "theoretical_params_saved": 80,
                        "realized_params": 900,
                        "theoretical_params": 920,
                        "skipped": False,
                    }
                },
                "savings": {"deploy_mode": "recompose"},
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 1.05,
                "ratio_vs_baseline": 1.04,
                "display_ci": (1.02, 1.06),
            },
            "logloss_preview": 0.0,
            "logloss_final": 0.05,
            "logloss_delta_ci": (-0.01, 0.02),
            "paired_delta_summary": {"mean": math.log(1.04), "degenerate": False},
            "window_plan": {"profile": "ci", "preview_n": 180, "final_n": 180},
            "bootstrap": {
                "replicates": 1200,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            },
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "spectral": {
                "caps_applied": 1,
                "max_caps": 3,
                "multiple_testing": {"alpha": 0.05},
                "summary": {"caps_exceeded": False},
                "caps_applied_by_family": {"attn": 1},
                "family_caps": {"attn": {"kappa": 0.8}},
                "family_z_quantiles": {
                    "attn": {"q95": 1.2, "q99": 2.3, "max": 2.5, "count": 5}
                },
                "policy": {"family_caps": {"attn": {"kappa": 0.75}}},
                "top_z_scores": {"attn": [{"module": "attn.0", "z": 2.0}]},
            },
            "rmt": {"families": {"mlp": {"epsilon": 0.2, "bare": 5, "guarded": 4}}},
            "variance": {"enabled": True, "summary": {"stable": True}, "policy": {}},
            "moe": {"top_k": 2, "capacity_factor": 1.2, "utilization": [0.7, 0.8]},
        },
        "artifacts": {
            "events_path": "",
            "logs_path": "",
            "generated_at": "2024-01-01T00:00:00Z",
        },
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "guard_overhead": {
            "bare_report": {"metrics": {"primary_metric": {"final": 10.0}}},
            "guarded_report": {"metrics": {"primary_metric": {"final": 10.1}}},
        },
        "structure": {"parameters_total": 2000, "compression_diagnostics": {}},
        "provenance": {"edits": {"name": "quant_rtn"}},
        "policies": {"tier": "balanced"},
        "policy_provenance": {"source": "auto"},
        "evaluation_windows": {
            "preview": {
                "window_ids": window_ids,
                "logloss": [0.1] * len(window_ids),
                "token_counts": token_counts,
            },
            "final": {
                "window_ids": window_ids,
                "logloss": [0.2] * len(window_ids),
                "token_counts": token_counts,
            },
        },
    }
    baseline = deepcopy(report)
    baseline["metrics"]["primary_metric"]["final"] = 1.0
    baseline["metrics"]["primary_metric"]["ratio_vs_baseline"] = 1.0
    baseline["metrics"]["paired_delta_summary"]["mean"] = 0.0
    baseline["guard_overhead"]["guarded_report"]["metrics"]["primary_metric"][
        "final"
    ] = 10.0
    baseline["metrics"]["window_plan"]["profile"] = "dev"
    baseline["evaluation_windows"]["final"]["logloss"] = [0.18, 0.2]
    return report, baseline
