from __future__ import annotations

from invarlock.reporting.certificate import _compute_validation_flags, make_certificate


def _mk_pm_report(*, ratio: float, pm_final: float = 10.0) -> dict:
    return {
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
            "name": "quant_rtn",
            "plan_digest": "deadbeef",
            "deltas": {"params_changed": 1, "layers_modified": 1},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": pm_final,
                "final": pm_final * ratio,
                "ratio_vs_baseline": ratio,
            },
            "bootstrap": {"replicates": 10, "alpha": 0.05},
            "preview_total_tokens": 1000,
            "final_total_tokens": 1000,
            # Inject ratio_ci at ppl-level via stats pairing in certificate internals
            "stats": {"pairing": "run_metrics"},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_policy_digest_unchanged_same_tier_thresholds() -> None:
    # Subject and baseline share the same tier and thresholds → changed=False
    rep = _mk_pm_report(ratio=1.0)
    base = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    cert = make_certificate(rep, base)
    pd = cert.get("policy_digest", {})
    assert isinstance(pd, dict) and pd.get("changed") is False


def test_ci_upper_bound_gating_from_ratio_ci() -> None:
    # Build a ppl dict where point ratio passes the limit but CI upper bound exceeds it
    ppl = {
        "ratio_vs_baseline": 1.08,  # within balanced ratio_limit=1.10
        "ratio_ci": (1.02, 1.12),  # upper bound beyond limit → should gate to False
        "preview_final_ratio": 1.0,
    }
    # Minimal tokens to satisfy sample-size floors
    ppl_metrics = {"preview_total_tokens": 1000, "final_total_tokens": 1000}
    flags = _compute_validation_flags(
        ppl=ppl,
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics=ppl_metrics,
        target_ratio=None,
        guard_overhead={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 2000},
    )
    assert flags["primary_metric_acceptable"] is False
