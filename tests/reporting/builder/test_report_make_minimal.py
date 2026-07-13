from __future__ import annotations

import math

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.report_types import RunReport, create_empty_report
from tests.reporting._support_auto_config import make_auto_config
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_primary_metric import independent_slice_summary


def _mk_minimal_report() -> RunReport:
    r = create_empty_report()
    # Fill meta/data/edit minimally
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["device"] = "cpu"
    r["meta"]["auto"] = make_auto_config()
    r["context"] = {"profile": "dev"}
    r["edit"]["name"] = "structured"
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
    r["metrics"]["preview_final_slice_delta_summary"] = independent_slice_summary(
        0.0,
        preview_windows=2,
        final_windows=2,
    )
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
            "logloss": [math.log(10.0), math.log(10.0)],
            "token_counts": [100, 100],
        }
    }
    return canonical_run_report(r)


def _mk_minimal_baseline() -> dict:
    return canonical_baseline(
        {
            "run_id": "base",
            "model_id": "m",
            "meta": {
                "seed": 0,
                "model_id": "m",
                "adapter": "hf",
                "auto": make_auto_config(),
            },
            "context": {"profile": "dev"},
            "evaluation_windows": {
                "final": {
                    "window_ids": [1, 2],
                    "logloss": [math.log(10.0), math.log(10.0)],
                    "token_counts": [100, 100],
                }
            },
            # Allow make_report to compute baseline primary_metric from windows
            "data": {
                "seq_len": 8,
                "preview_n": 2,
                "final_n": 2,
                "dataset": "unit",
                "split": "validation",
                "stride": 8,
            },
            "edit": {
                "name": "noop",
                "plan_digest": "0",
                "deltas": {
                    "params_changed": 0,
                    "layers_modified": 0,
                    "sparsity": None,
                    "bitwidth_map": None,
                },
            },
            "guards": [],
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 10.0,
                    "final": 10.0,
                }
            },
            "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
            "flags": {"guard_recovered": False, "rollback_reason": None},
        }
    )


def test_make_evaluation_report_minimal_paths() -> None:
    report = _mk_minimal_report()
    baseline = _mk_minimal_baseline()
    cert = make_report(report, baseline)
    assert validate_report(cert)
    # Core fields present
    assert cert["schema_version"] == "v1"
    assert isinstance(cert.get("primary_metric"), dict)
    # Confidence label computed
    assert isinstance(cert.get("confidence"), dict)


def test_make_evaluation_report_tiny_relax_flag() -> None:
    report = _mk_minimal_report()
    baseline = _mk_minimal_baseline()
    report["context"] = {"run": {"tiny_relax": True}}
    cert = make_report(report, baseline)
    assert cert.get("auto", {}).get("tiny_relax") is True


def test_make_evaluation_report_sets_measurement_contract_match_from_run_baseline() -> (
    None
):
    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    rmt_contract = {
        "kind": "activation_edge_risk",
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    report = _mk_minimal_report()
    report["guards"] = [
        {
            "name": "spectral",
            "passed": True,
            "policy": {"enabled": True},
            "metrics": {
                "measurement_contract": spectral_contract,
                "max_spectral_norm_final": 1.0,
                "mean_spectral_norm_final": 1.0,
                "caps_applied": 0,
            },
        },
        {
            "name": "rmt",
            "passed": True,
            "policy": {"enabled": True},
            "metrics": {
                "measurement_contract": rmt_contract,
                "edge_risk_by_family_base": {"attn": 1.0},
                "edge_risk_by_family": {"attn": 1.0},
                "epsilon_by_family": {"attn": 0.01},
                "stable": True,
            },
        },
    ]
    baseline = _mk_minimal_baseline()
    baseline["guards"] = [
        {
            "name": "spectral",
            "passed": True,
            "policy": {"enabled": True},
            "metrics": {
                "measurement_contract": spectral_contract,
                "max_spectral_norm_final": 1.0,
                "mean_spectral_norm_final": 1.0,
            },
        },
        {
            "name": "rmt",
            "passed": True,
            "policy": {"enabled": True},
            "metrics": {
                "measurement_contract": rmt_contract,
                "edge_risk_by_family": {"attn": 1.0},
            },
        },
    ]

    cert = make_report(canonical_run_report(report), canonical_baseline(baseline))

    assert cert["spectral"]["measurement_contract_match"] is True
    assert cert["rmt"]["measurement_contract_match"] is True
