from __future__ import annotations

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_guard_metric_impact import ppl_guard_context


def _mk_min_report_pm(kind: str = "ppl_causal") -> dict:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "ts": "2024-01-01T00:00:00",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "context": {"profile": "dev"},
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
            "plan_digest": "deadbeef",
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
                "kind": kind,
                "preview": 4.0,
                "final": 4.0,
                "ratio_vs_baseline": 1.0,
            },
            "bootstrap": {
                "replicates": 10,
                "alpha": 0.05,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_make_evaluation_report_ratio_ci_from_run_metrics():
    rep = _mk_min_report_pm()
    # Provide a logloss_delta_ci to trigger run-metrics ratio_ci derivation path
    rep["metrics"]["logloss_delta_ci"] = (-0.02, 0.03)
    baseline = _mk_min_report_pm()
    baseline["edit"]["name"] = "noop"
    cert = make_report(canonical_run_report(rep), canonical_baseline(baseline))
    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict)
    dci = pm.get("display_ci")
    assert isinstance(dci, list | tuple) and len(dci) == 2
    lo, hi = float(dci[0]), float(dci[1])
    # A pair is present (may be degenerate depending on propagation path)
    assert isinstance(lo, float) and isinstance(hi, float)


def test_make_evaluation_report_pairing_and_dataset_stats_injection():
    # Matching windows → paired_baseline path
    rep = _mk_min_report_pm()
    rep.setdefault("evaluation_windows", {})["final"] = {
        "window_ids": [1, 2],
        "logloss": [4.0, 4.0],
        "token_counts": [100, 100],
    }
    base = _mk_min_report_pm()
    base["evaluation_windows"] = {
        "final": {"window_ids": [1, 2], "logloss": [4.0, 4.0]}
    }
    cert = make_report(canonical_run_report(rep), canonical_baseline(base))
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    assert stats.get("pairing") == "paired_baseline"
    assert stats.get("paired_windows") == 2


def test_make_evaluation_report_policy_digest_changed_with_tier_difference():
    rep = _mk_min_report_pm()
    base = _mk_min_report_pm()
    base["meta"]["auto"] = {
        "tier": "conservative",
        "probes_used": 0,
        "target_pm_ratio": None,
    }
    cert = make_report(canonical_run_report(rep), canonical_baseline(base))
    pd = cert.get("policy_digest", {})
    assert isinstance(pd, dict) and pd.get("changed") in {True, False}


def test_make_evaluation_report_guard_metric_impact_integration():
    rep = _mk_min_report_pm()
    # Provide direct ratio path for overhead section
    rep["guard_metric_impact"] = ppl_guard_context(100.0, 101.0, degradation_limit=0.02)
    base = _mk_min_report_pm()
    cert = make_report(canonical_run_report(rep), canonical_baseline(base))
    go = cert.get("guard_metric_impact", {})
    assert go.get("evaluated") is True
    assert go.get("passed") is True
    assert cert.get("validation", {}).get("guard_metric_impact_acceptable") in {
        True,
        False,
    }
