from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def _minimal_pm_report():
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 7,
            "device": "cpu",
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "x",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (10.0, 10.0),
            },
            "bootstrap": {"replicates": 150, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {
            "final": {"window_ids": [1], "logloss": [1.0], "token_counts": [100]}
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def _baseline():
    payload = _minimal_pm_report()
    payload["metrics"]["primary_metric"].pop("ratio_vs_baseline", None)
    return canonical_baseline(payload)


def test_evaluation_report_has_no_ppl_keys():
    cert = make_report(canonical_run_report(_minimal_pm_report()), _baseline())
    assert "ppl" not in cert
