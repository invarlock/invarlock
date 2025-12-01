from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def test_accuracy_hysteresis_applied_and_pm_ok() -> None:
    # Build an accuracy-style report; set delta slightly below delta_min_pp but within hysteresis
    # Tier balanced: delta_min_pp=-1.0, hysteresis_delta_pp=0.1 → accept >= -1.1
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
                "kind": "accuracy",
                "final": 0.79,
                "ratio_vs_baseline": -1.05,
            },
            "classification": {
                "final": {"correct_total": 210, "total": 210},
                "preview": {"correct_total": 200, "total": 200},
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    base = {"metrics": {"primary_metric": {"kind": "accuracy", "final": 0.80}}}
    cert = make_certificate(rep, base)
    val = cert.get("validation", {})
    # Acceptable due to hysteresis with sufficient n_final and mark hysteresis applied
    assert val.get("primary_metric_acceptable") is True
    assert val.get("hysteresis_applied") is True
