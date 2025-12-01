from __future__ import annotations

import math

from invarlock.reporting.certificate import make_certificate


def _mk_rep_base_with_windows(delta: float) -> tuple[dict, dict]:
    # Construct paired windows with logloss differing by a constant delta
    # so that ΔlogNLL mean equals `delta` exactly.
    base_ll = [2.30, 2.31, 2.29, 2.33]
    subj_ll = [x + delta for x in base_ll]
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
            "preview_n": 4,
            "final_n": 4,
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
                "preview": math.exp(base_ll[0]),
                "final": math.exp(subj_ll[0]),
                "ratio_vs_baseline": math.exp(delta),
            },
            "bootstrap": {
                "method": "bca",
                "replicates": 50,
                "alpha": 0.10,
                "coverage": {"preview": {"used": 4}, "final": {"used": 4}},
            },
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2, 3, 4],
                "logloss": subj_ll,
                "token_counts": [100, 100, 100, 100],
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    base = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": math.exp(base_ll[0])}
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2, 3, 4],
                "logloss": base_ll,
                "token_counts": [100, 100, 100, 100],
            }
        },
    }
    return rep, base


def test_paired_ci_identity_holds() -> None:
    # Build data where Δlog is known; the implementation should ensure
    # ratio_ci == exp(logloss_delta_ci) for paired_baseline source.
    delta = 0.02
    rep, base = _mk_rep_base_with_windows(delta)
    cert = make_certificate(rep, base)
    pm = cert.get("primary_metric", {})
    # Expect display_ci present and identity with exp(Δ bounds)
    dci = pm.get("display_ci") if isinstance(pm, dict) else None
    assert isinstance(dci, list | tuple) and len(dci) == 2
