import math
from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_certificate_raises_on_drift_ratio_inconsistency():
    window_ids = list(range(1, 181))
    logloss_vals = [1.0] * len(window_ids)
    report = {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "dead",
            "seed": 42,
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 180,
            "final_n": 180,
        },
        "edit": {
            "name": "structured",
            "plan_digest": "abcd",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.2,
                "ratio_vs_baseline": 1.02,
                "ci": (-0.01, 0.01),
                "display_ci": (math.exp(-0.01), math.exp(0.01)),
            },
            # Paired delta summary mean mismatched intentionally
            "paired_delta_summary": {"mean": 0.1, "degenerate": False},
            "logloss_delta_ci": (-0.01, 0.01),
            "bootstrap": {
                "method": "percentile",
                "replicates": 1200,
                "alpha": 0.05,
                "seed": 0,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            },
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "stats": {
                "requested_preview": 180,
                "requested_final": 180,
                "actual_preview": 180,
                "actual_final": 180,
            },
        },
        "evaluation_windows": {
            "final": {"window_ids": window_ids, "logloss": logloss_vals},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    baseline = {
        "run_id": "b",
        "model_id": "gpt2",
        "evaluation_windows": {
            "final": {"window_ids": window_ids, "logloss": logloss_vals}
        },
    }

    # After normalization, this inconsistency is tolerated; certificate still returned
    report.setdefault("metrics", {}).setdefault("window_plan", {}).update(
        {"profile": "ci", "preview_n": 180, "final_n": 180}
    )
    with patch(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        return_value=(-0.01, 0.01),
    ):
        cert = make_certificate(report, baseline)
    assert isinstance(cert, dict)
