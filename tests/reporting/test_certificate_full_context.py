from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from invarlock.reporting import certificate as cert


def _rich_run_report() -> tuple[dict, dict]:
    window_ids = list(range(1, 181))
    token_counts = [10] * len(window_ids)
    report = {
        "meta": {
            "model_id": "demo-model",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 7,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00Z",
            "auto": {
                "tier": "balanced",
                "probes_used": ["spectral"],
                "target_pm_ratio": 1.1,
            },
        },
        "data": {
            "dataset": "demo-ds",
            "split": "eval",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 180,
            "final_n": 180,
            "windows": {"preview": 180, "final": 180, "seed": 7},
        },
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "plan123",
            "config": {"scope": "ffn", "seed": 7, "frac": 0.5, "clamp_ratio": 0.2},
            "deltas": {
                "params_changed": 10,
                "layers_modified": 2,
                "bitwidth_map": {
                    "layer1": {"bitwidth": 4, "group_size": None, "params": 512},
                    "layer2": {"bitwidth": 8, "group_size": 32, "params": 256},
                },
                "rank_map": {
                    "layer1": {
                        "rank": 8,
                        "params_saved": 128,
                        "energy_retained": 0.95,
                        "deploy_mode": "recompose",
                        "savings_mode": "realized",
                        "realized_params_saved": 64,
                        "theoretical_params_saved": 80,
                        "realized_params": 900,
                        "theoretical_params": 920,
                        "skipped": False,
                    }
                },
                "savings": {"deploy_mode": "recompose"},
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 1.05,
                "ratio_vs_baseline": 1.04,
                "display_ci": (1.02, 1.06),
            },
            "logloss_preview": 0.0,
            "logloss_final": 0.05,
            "logloss_delta_ci": (-0.01, 0.02),
            "paired_delta_summary": {"mean": math.log(1.04), "degenerate": False},
            "window_plan": {"profile": "ci", "preview_n": 180, "final_n": 180},
            "bootstrap": {
                "replicates": 1200,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            },
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "spectral": {
                "caps_applied": 1,
                "max_caps": 3,
                "multiple_testing": {"alpha": 0.05},
                "summary": {"caps_exceeded": False},
                "caps_applied_by_family": {"attn": 1},
                "family_caps": {"attn": {"kappa": 0.8}},
                "family_z_quantiles": {
                    "attn": {"q95": 1.2, "q99": 2.3, "max": 2.5, "count": 5}
                },
                "policy": {"family_caps": {"attn": {"kappa": 0.75}}},
                "top_z_scores": {"attn": [{"module": "attn.0", "z": 2.0}]},
            },
            "rmt": {"families": {"mlp": {"epsilon": 0.2, "bare": 5, "guarded": 4}}},
            "variance": {"enabled": True, "summary": {"stable": True}, "policy": {}},
            "moe": {"top_k": 2, "capacity_factor": 1.2, "utilization": [0.7, 0.8]},
        },
        "artifacts": {
            "events_path": "",
            "logs_path": "",
            "generated_at": "2024-01-01T00:00:00Z",
        },
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "guard_overhead": {
            "bare_report": {"metrics": {"primary_metric": {"final": 10.0}}},
            "guarded_report": {"metrics": {"primary_metric": {"final": 10.1}}},
        },
        "structure": {"parameters_total": 2000, "compression_diagnostics": {}},
        "provenance": {"edits": {"name": "quant_rtn"}},
        "policies": {"tier": "balanced"},
        "policy_provenance": {"source": "auto"},
        "evaluation_windows": {
            "preview": {
                "window_ids": window_ids,
                "logloss": [0.1] * len(window_ids),
                "token_counts": token_counts,
            },
            "final": {
                "window_ids": window_ids,
                "logloss": [0.2] * len(window_ids),
                "token_counts": token_counts,
            },
        },
    }
    baseline = deepcopy(report)
    baseline["metrics"]["primary_metric"]["final"] = 1.0
    baseline["metrics"]["primary_metric"]["ratio_vs_baseline"] = 1.0
    baseline["metrics"]["paired_delta_summary"]["mean"] = 0.0
    baseline["guard_overhead"]["guarded_report"]["metrics"]["primary_metric"][
        "final"
    ] = 10.0
    baseline["metrics"]["window_plan"]["profile"] = "dev"
    baseline["evaluation_windows"]["final"]["logloss"] = [0.18, 0.2]
    return report, baseline


def test_make_certificate_rich_context_generates_diagnostics(monkeypatch):
    def fake_compute(report, *, kind, baseline=None):
        metrics = (report.get("metrics") or {}).get("primary_metric", {})
        return {"final": metrics.get("final", 1.0), "direction": "lower"}

    monkeypatch.setattr(cert, "compute_primary_metric_from_report", fake_compute)
    monkeypatch.setattr(
        cert, "get_metric", lambda *_: SimpleNamespace(direction="lower")
    )

    report, baseline = _rich_run_report()
    cert_obj = cert.make_certificate(report, baseline)
    assert cert_obj["quality_overhead"]["basis"] == "ratio"
    stats = cert_obj["dataset"]["windows"]["stats"]
    assert stats["paired_windows"] >= 1
    structure = cert_obj["structure"]["compression_diagnostics"]
    assert structure["execution_status"] in {"successful", "partial"}
    assert cert_obj["provenance"]["edit_digest"]["family"] == "quantization"


def test_make_certificate_surfaces_pairing_and_policy_digest():
    report, baseline = _rich_run_report()
    cert_obj = cert.make_certificate(report, baseline)
    stats = cert_obj["dataset"]["windows"]["stats"]
    assert "pairing" in stats and stats["paired_windows"] >= 1
    assert "coverage" in stats and "window_match_fraction" in stats
    policy_digest = cert_obj["policy_digest"]
    assert policy_digest["policy_version"] == cert.POLICY_VERSION
    assert policy_digest["thresholds_hash"]


def test_make_certificate_end_to_end_populates_optional_sections_and_validations(
    monkeypatch,
):
    monkeypatch.setattr(
        cert, "_normalize_and_validate_report", lambda value: value, raising=False
    )
    monkeypatch.setattr(cert, "_normalize_baseline", lambda value: value, raising=False)
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)

    report["guards"] = [
        {
            "name": "spectral",
            "metrics": {
                "caps_applied": 10,
                "caps_exceeded": True,
                "max_spectral_norm_final": 2.0,
                "mean_spectral_norm_final": 1.5,
                "violations_detected": 10,
            },
            "policy": {"max_caps": 3},
        },
        {
            "name": "rmt",
            "metrics": {
                "stable": False,
                "flagged_rate": 0.8,
                "rmt_outliers": 5,
                "baseline_outliers_per_family": {"ffn": 1},
                "outliers_per_family": {"ffn": 5},
            },
        },
    ]

    report["metrics"]["stats"] = {
        "requested_preview": 2,
        "requested_final": 2,
        "actual_preview": 2,
        "actual_final": 2,
        "coverage_ok": True,
    }
    report["metrics"]["bootstrap"]["coverage"] = {"preview": {"used": 3}}
    report["metrics"]["window_match_fraction"] = 0.92
    report["metrics"]["window_overlap_fraction"] = 0.4
    report["metrics"]["window_pairing_reason"] = "id_match"
    report["metrics"]["window_plan"]["profile"] = "dev"

    report["metrics"]["latency_ms_p50"] = 12.0
    report["metrics"]["latency_ms_p95"] = 20.0
    report["metrics"]["throughput_sps"] = 95.0
    baseline["metrics"]["latency_ms_p50"] = 10.0
    baseline["metrics"]["throughput_sps"] = 110.0

    report["metrics"]["classification"] = {
        "subgroups": {
            "preview": {
                "group_counts": {"A": 10, "B": 20},
                "correct_counts": {"A": 8, "B": 15},
            },
            "final": {
                "group_counts": {"A": 10, "B": 20, "C": 5},
                "correct_counts": {"A": 7, "B": 16, "C": 4},
            },
        }
    }

    report["metrics"]["secondary_metrics"] = [
        {"kind": "valid_metric", "final": 1.0, "unit": "%"},
        {"final": 99.0},
    ]

    report["metrics"]["moe"] = {
        "top_k": 2,
        "capacity_factor": 1.3,
        "load_balance_loss": 0.02,
        "router_entropy": 1.5,
        "utilization": [0.8, 0.9],
    }
    baseline["metrics"]["moe"] = {
        "top_k": 2,
        "capacity_factor": 1.3,
        "load_balance_loss": 0.01,
        "router_entropy": 1.6,
        "utilization": [0.7, 0.8],
    }

    report["guard_overhead"] = {
        "overhead_threshold": 0.01,
        "bare_report": {
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}
        },
        "guarded_report": {
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.05}}
        },
    }

    report["metrics"]["spectral"]["caps_applied"] = 10
    report["metrics"]["spectral"]["max_caps"] = 3
    report["metrics"]["spectral"]["caps_exceeded"] = True
    report["metrics"]["rmt"]["stable"] = False

    certificate = cert.make_certificate(report, baseline)

    stats = certificate["dataset"]["windows"]["stats"]
    assert stats["pairing"]
    assert stats["paired_windows"] >= 1
    assert stats["coverage"]["preview"]["used"] == 3
    assert stats["window_match_fraction"] == pytest.approx(0.92)
    assert stats["window_pairing_reason"] == "id_match"

    system_overhead = certificate["system_overhead"]
    lat_entry = system_overhead["latency_ms_p50"]
    assert lat_entry["edited"] == 12.0
    assert lat_entry["baseline"] == 10.0
    assert lat_entry["ratio"] == pytest.approx(1.2)
    throughput_entry = system_overhead["throughput_sps"]
    assert throughput_entry["edited"] == 95.0
    assert throughput_entry["baseline"] == 110.0
    assert throughput_entry["ratio"] == pytest.approx(95.0 / 110.0)
    assert "baseline" not in system_overhead["latency_ms_p95"]  # baseline lacked p95

    subgroups = certificate["classification"]["subgroups"]
    assert subgroups["A"]["delta_pp"] == pytest.approx(-10.0)
    assert subgroups["B"]["delta_pp"] == pytest.approx(5.0)
    assert math.isnan(subgroups["C"]["preview"])

    secondary = certificate["secondary_metrics"]
    assert len(secondary) == 1 and secondary[0]["kind"] == "valid_metric"

    validation = certificate["validation"]
    assert validation["spectral_stable"] is False
    assert validation["rmt_stable"] is False
    assert validation.get("moe_observed") is True
    assert validation.get("moe_identity_ok") is True

    guard = certificate["guard_overhead"]
    assert guard["evaluated"] is True
    assert guard["passed"] is True
    assert guard["bare_ppl"] == pytest.approx(10.0)
    assert guard["guarded_ppl"] == pytest.approx(10.05)


def test_make_certificate_policy_digest_changes_when_policy_override_differs(
    monkeypatch,
):
    monkeypatch.setattr(
        cert, "_normalize_and_validate_report", lambda value: value, raising=False
    )
    monkeypatch.setattr(cert, "_normalize_baseline", lambda value: value, raising=False)
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["guards"].append(
        {"name": "spectral", "policy": {"max_caps": 10}, "metrics": {"caps_applied": 0}}
    )
    baseline["guards"] = []
    baseline["meta"]["auto"]["tier"] = "conservative"

    certificate = cert.make_certificate(report, baseline)

    assert certificate["policy_digest"]["changed"] is True


def test_make_certificate_provenance_and_guard_schedule_fallback(monkeypatch):
    monkeypatch.setattr(
        cert, "_normalize_and_validate_report", lambda value: value, raising=False
    )
    monkeypatch.setattr(cert, "_normalize_baseline", lambda value: value, raising=False)
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["provenance"] = {}
    report["evaluation_windows"]["final"]["window_ids"] = [11, 12, 13]
    report["guard_overhead"] = {}
    report["metrics"]["window_plan"]["profile"] = "dev"

    certificate = cert.make_certificate(report, baseline)

    prov = certificate["provenance"]
    assert "provider_digest" in prov
    assert (
        prov.get("window_ids_digest")
        == certificate["guard_overhead"]["schedule_digest"]
        == prov["provider_digest"]["ids_sha256"]
    )


def test_make_certificate_emits_telemetry_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        cert, "_normalize_and_validate_report", lambda value: value, raising=False
    )
    monkeypatch.setattr(cert, "_normalize_baseline", lambda value: value, raising=False)
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)

    cert.make_certificate(report, baseline)
    out = capsys.readouterr().out
    assert "INVARLOCK_TELEMETRY" in out
