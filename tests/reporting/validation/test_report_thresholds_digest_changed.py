from __future__ import annotations

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _mk_pm_report(*, ratio: float = 1.0, pm_final: float = 10.0) -> dict:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
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
    base = _mk_pm_report(pm_final=10.0)
    base["meta"]["auto"]["tier"] = "conservative"
    cert = make_report(rep, base)
    pd = cert.get("policy_digest", {})
    assert isinstance(pd, dict)
    assert pd.get("changed") is True
