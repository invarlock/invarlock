import math
from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_certificate_paired_ci_success_and_stats_passthrough():
    # Report with consistent preview/final and window data
    window_ids = list(range(1, 181))
    logloss_vals = [1.0] * len(window_ids)
    ratio_ci = (math.exp(-0.08), math.exp(-0.06))
    report = {
        "run_id": "r1",
        "meta": {
            "model_id": "m",
            "adapter": "a",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "deadbeef",
            "seed": 7,
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
            # Choose preview/final so drift ≈ exp(-0.07)
            "ppl_preview": 10.0,
            "ppl_final": 10.0 * math.exp(-0.07),
            "ppl_ratio": math.exp(-0.07),
            "ppl_preview_ci": (9.5, 10.5),
            "ppl_final_ci": (9.5, 10.5),
            "ppl_ratio_ci": ratio_ci,
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0 * math.exp(-0.07),
                "ratio_vs_baseline": math.exp(-0.07),
                "ci": (-0.08, -0.06),
                "display_ci": ratio_ci,
            },
            # Provide paired delta summary mean consistent with drift
            "paired_delta_summary": {"mean": -0.07, "degenerate": False},
            # Include stats passthrough keys
            "stats": {
                "requested_preview": 180,
                "requested_final": 180,
                "actual_preview": 180,
                "actual_final": 180,
                "coverage_ok": True,
            },
            # Include a window plan to propagate
            "window_plan": {"profile": "ci", "preview_n": 180, "final_n": 180},
            # Configure bootstrap so paired path is attempted
            "bootstrap": {
                "method": "bca",
                "replicates": 1200,
                "alpha": 0.1,
                "seed": 0,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            },
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
        },
        "evaluation_windows": {
            "final": {"window_ids": window_ids, "logloss": logloss_vals}
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {
            "final": {"window_ids": window_ids, "logloss": logloss_vals}
        },
    }

    # Return a tight ΔlogNLL CI around the mean so ratio_ci == exp(bounds)
    with patch(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        return_value=(-0.08, -0.06),
    ):
        with patch(
            "invarlock.reporting.certificate.validate_report", return_value=True
        ):
            cert = make_certificate(report, baseline)

    # Pairing stats live in dataset.windows.stats
    stats = (cert.get("dataset", {}).get("windows", {}) or {}).get("stats", {})
    assert stats.get("pairing") == "paired_baseline"

    # Optional passthrough keys and window plan may be omitted after normalization
    assert isinstance(stats, dict)
