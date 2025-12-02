from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_pm_report(*, ratio: float = 1.0, pm_final: float = 10.0) -> dict:
    return {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": pm_final,
                "final": pm_final * ratio,
                "ratio_vs_baseline": ratio,
                "display_ci": (1.0, 1.0),
            },
            "bootstrap": {"replicates": 10, "alpha": 0.05},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_policy_digest_changed_when_baseline_tier_differs() -> None:
    rep = _mk_pm_report(ratio=1.0)
    # Subject implicit tier is balanced (default); baseline sets explicit conservative tier
    base = {
        "meta": {"auto": {"tier": "conservative"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    cert = make_certificate(rep, base)
    pd = cert.get("policy_digest", {})
    assert isinstance(pd, dict)
    assert pd.get("changed") is True
