import math
from unittest.mock import patch

import pytest

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_evaluation_report_raises_on_drift_ratio_inconsistency():
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
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "ci", "assurance": {"mode": "off"}},
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
            # Preview/final slice delta mean mismatched intentionally
            "preview_final_slice_delta_summary": {
                "mean": 0.1,
                "degenerate": False,
            },
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
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "ci", "assurance": {"mode": "off"}},
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 180,
            "final_n": 180,
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
        },
        "edit": {
            "name": "noop",
            "plan_digest": "baseline",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {
            "final": {"window_ids": window_ids, "logloss": logloss_vals}
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }

    # CI and release reports fail closed when the summary and ratio disagree.
    report.setdefault("metrics", {}).setdefault("window_plan", {}).update(
        {"profile": "ci", "preview_n": 180, "final_n": 180}
    )
    with (
        patch(
            "invarlock.core.bootstrap.compute_paired_delta_log_ci",
            return_value=(-0.01, 0.01),
        ),
        pytest.raises(
            ValueError,
            match="Preview/final ΔlogNLL mean is inconsistent",
        ),
    ):
        make_report(canonical_run_report(report), canonical_baseline(baseline))
