from __future__ import annotations

from invarlock.reporting.certificate import make_certificate, validate_certificate
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_minimal_report() -> RunReport:
    r = create_empty_report()
    # Fill meta/data/edit minimally
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["device"] = "cpu"
    r["meta"]["auto"] = {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None}  # type: ignore[assignment]
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    # Primary metric
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    # Provide bootstrap and coverage stats to exercise paths
    r["metrics"]["bootstrap"] = {
        "method": "percentile",
        "replicates": 50,
        "alpha": 0.05,
        "seed": 0,
        "coverage": {
            "preview": {"used": 2},
            "final": {"used": 2},
        },
    }
    # Provide paired delta summary to be copied through
    r["metrics"]["paired_delta_summary"] = {"mean": 0.0}
    # Provide tokens to consider token floor path
    r["metrics"]["preview_total_tokens"] = 50
    r["metrics"]["final_total_tokens"] = 50
    # Provide ΔlogNLL CI so ratio_ci can be derived from it
    r["metrics"]["logloss_delta"] = 0.0
    r["metrics"]["logloss_delta_ci"] = (-0.01, 0.01)
    # Provide evaluation windows for pairing
    r["evaluation_windows"] = {
        "final": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.31],
            "token_counts": [100, 100],
        }
    }
    return r


def _mk_minimal_baseline() -> dict:
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
        # Allow make_certificate to compute baseline primary_metric from windows
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


def test_make_certificate_minimal_paths() -> None:
    report = _mk_minimal_report()
    baseline = _mk_minimal_baseline()
    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
    # Core fields present
    assert cert["schema_version"] == "v1"
    assert isinstance(cert.get("primary_metric"), dict)
    # Confidence label computed
    assert isinstance(cert.get("confidence"), dict)


def test_make_certificate_tiny_relax_flag(monkeypatch) -> None:
    report = _mk_minimal_report()
    baseline = _mk_minimal_baseline()
    # Enable tiny-relax to exercise relaxed gating and provenance flags
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    cert = make_certificate(report, baseline)
    assert cert.get("auto", {}).get("tiny_relax") is True
