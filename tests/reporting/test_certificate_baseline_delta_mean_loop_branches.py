from __future__ import annotations

from invarlock.reporting.certificate import make_certificate, validate_certificate
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_report_with_bad_window_entries() -> RunReport:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["device"] = "cpu"
    r["meta"]["auto"] = {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None}  # type: ignore[assignment]
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    r["metrics"]["bootstrap"] = {
        "method": "percentile",
        "replicates": 10,
        "alpha": 0.05,
        "seed": 0,
        "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
    }
    r["metrics"]["paired_delta_summary"] = {"mean": 0.0}
    r["metrics"]["preview_total_tokens"] = 50
    r["metrics"]["final_total_tokens"] = 50
    r["metrics"]["logloss_delta"] = 0.0
    r["metrics"]["logloss_delta_ci"] = (-0.01, 0.01)
    r["evaluation_windows"] = {
        "final": {
            "window_ids": [1, 2, "bad", 3],
            "logloss": [2.30, 2.31, 2.00, 2.50],
            "token_counts": [100, -1, 50, 100],
        }
    }
    return r


def _mk_baseline() -> dict:
    return {
        "run_id": "base",
        "model_id": "m",
        "meta": {"seed": 0, "model_id": "m"},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            }
        },
        "data": {
            "seq_len": 8,
            "preview_n": 2,
            "final_n": 2,
            "dataset": "unit",
            "split": "validation",
            "stride": 8,
        },
        "edit": {
            "name": "none",
            "plan_digest": "0",
            "deltas": {
                "params_changed": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_make_certificate_handles_bad_window_entries_in_weighted_mean_loop(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        lambda *_a, **_k: (-0.01, 0.01),
    )
    report = _mk_report_with_bad_window_entries()
    baseline = _mk_baseline()
    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
