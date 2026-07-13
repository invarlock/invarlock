from __future__ import annotations

import copy
import math
from typing import Any

from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def mock_report_with_windows() -> dict[str, Any]:
    """Build the canonical paired-window subject used by report acceptance tests."""
    preview = {
        "window_ids": [1, 2],
        "logloss": [1.00, 1.06],
        "token_counts": [100, 200],
    }
    final = {
        "window_ids": [3, 4],
        "logloss": [1.05, 1.15],
        "token_counts": [100, 200],
    }
    ppl_prev = math.exp((1.00 * 100 + 1.06 * 200) / 300)
    ppl_fin_subj = math.exp((1.05 * 100 + 1.15 * 200) / 300)
    return canonical_run_report(
        {
            "meta": {
                "model_id": "stub",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 7,
                "seeds": {"python": 7, "numpy": 7, "torch": 7},
                "auto": {
                    "tier": "balanced",
                    "probes_used": 0,
                    "target_pm_ratio": None,
                },
            },
            "context": {"profile": "dev"},
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 2,
                "final_n": 2,
            },
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": ppl_prev,
                    "final": ppl_fin_subj,
                    "ratio_vs_baseline": 1.0,
                },
                "bootstrap": {
                    "replicates": 200,
                    "alpha": 0.05,
                    "method": "percentile",
                },
            },
            "evaluation_windows": {"preview": preview, "final": final},
            "edit": {
                "name": "structured",
                "plan_digest": "structured-test",
                "deltas": {"params_changed": 1, "layers_modified": 1},
            },
            "artifacts": {"events_path": "", "logs_path": ""},
            "guards": [],
            "flags": {"guard_recovered": False, "rollback_reason": None},
        }
    )


def mock_baseline(report: dict[str, Any]) -> dict[str, Any]:
    """Derive the canonical paired baseline for an acceptance subject."""
    preview = report["evaluation_windows"]["preview"]
    final = report["evaluation_windows"]["final"]
    ppl_fin_base = math.exp((1.00 * 100 + 1.10 * 200) / 300)
    baseline = copy.deepcopy(report)
    baseline["edit"]["name"] = "noop"
    baseline["edit"]["deltas"] = {"params_changed": 0, "layers_modified": 0}
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": ppl_fin_base,
        "final": ppl_fin_base,
    }
    baseline["metrics"]["bootstrap"] = {
        "replicates": 200,
        "alpha": 0.05,
        "method": "percentile",
    }
    baseline["evaluation_windows"] = {"preview": preview, "final": final}
    return canonical_baseline(baseline)
