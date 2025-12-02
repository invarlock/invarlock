from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _rep_with_subgroups_and_sys() -> tuple[dict, dict]:
    rep = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "ts": "2024-01-01T00:00:00",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 4.0,
                "final": 4.0,
                "ratio_vs_baseline": 1.0,
            },
            "classification": {
                "subgroups": {
                    "preview": {"group_counts": {"A": 10}, "correct_counts": {"A": 8}},
                    "final": {"group_counts": {"A": 12}, "correct_counts": {"A": 10}},
                }
            },
            "latency_ms_per_tok": 1.5,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    base = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 4.0},
            "latency_ms_per_tok": 1.6,
        },
    }
    return rep, base


def test_make_certificate_classification_and_system_overhead_blocks():
    rep, base = _rep_with_subgroups_and_sys()
    cert = make_certificate(rep, base)
    # Classification subgroup summary present
    cls = cert.get("classification", {})
    assert isinstance(cls, dict)
    # System overhead section present
    sys = cert.get("system_overhead", {})
    assert isinstance(sys, dict) and "latency_ms_p50" in sys
