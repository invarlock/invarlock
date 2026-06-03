from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

import invarlock.eval.primary_metric as primary_metric_mod
from invarlock.reporting import report_make as cert
from invarlock.reporting import report_normalization as report_normalization_mod
from invarlock.reporting.report_make import make_report
from tests.reporting.builder._support_full_context import _rich_run_report


def test_make_evaluation_report_rich_context_generates_diagnostics(monkeypatch):
    def fake_compute(report, *, kind, baseline=None):
        metrics = (report.get("metrics") or {}).get("primary_metric", {})
        return {"final": metrics.get("final", 1.0), "direction": "lower"}

    monkeypatch.setattr(
        cert.primary_metric_mod,
        "compute_primary_metric_from_report",
        fake_compute,
    )
    monkeypatch.setattr(
        primary_metric_mod, "get_metric", lambda *_: SimpleNamespace(direction="lower")
    )
    monkeypatch.setattr(
        cert.report_overhead_mod,
        "compute_quality_overhead_from_guard",
        lambda *_args, **_kwargs: {
            "basis": "ratio",
            "value": 1.01,
            "kind": "ppl_causal",
        },
    )

    report, baseline = _rich_run_report()
    cert_obj = make_report(report, baseline)
    assert cert_obj["quality_overhead"]["basis"] == "ratio"
    stats = cert_obj["dataset"]["windows"]["stats"]
    assert stats["paired_windows"] >= 1
    structure = cert_obj["structure"]["compression_diagnostics"]
    assert structure["execution_status"] in {"successful", "partial"}
    assert cert_obj["provenance"]["edit_digest"]["family"] == "quantization"


def test_make_evaluation_report_surfaces_pairing_and_policy_digest():
    report, baseline = _rich_run_report()
    cert_obj = make_report(report, baseline)
    stats = cert_obj["dataset"]["windows"]["stats"]
    assert "pairing" in stats and stats["paired_windows"] >= 1
    assert "coverage" in stats and "window_match_fraction" in stats
    policy_digest = cert_obj["policy_digest"]
    assert policy_digest["policy_version"] == cert.POLICY_VERSION
    assert policy_digest["thresholds_hash"]


def test_make_evaluation_report_end_to_end_populates_optional_sections_and_validations(
    monkeypatch,
):
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_baseline",
        lambda value: value,
        raising=False,
    )
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

    evaluation_report = make_report(report, baseline)

    stats = evaluation_report["dataset"]["windows"]["stats"]
    assert stats["pairing"]
    assert stats["paired_windows"] >= 1
    assert stats["coverage"]["preview"]["used"] == 3
    assert stats["window_match_fraction"] == pytest.approx(0.92)
    assert stats["window_pairing_reason"] == "id_match"

    system_overhead = evaluation_report["system_overhead"]
    lat_entry = system_overhead["latency_ms_p50"]
    assert lat_entry["edited"] == 12.0
    assert lat_entry["baseline"] == 10.0
    assert lat_entry["ratio"] == pytest.approx(1.2)
    throughput_entry = system_overhead["throughput_sps"]
    assert throughput_entry["edited"] == 95.0
    assert throughput_entry["baseline"] == 110.0
    assert throughput_entry["ratio"] == pytest.approx(95.0 / 110.0)
    assert "baseline" not in system_overhead["latency_ms_p95"]  # baseline lacked p95

    subgroups = evaluation_report["classification"]["subgroups"]
    assert subgroups["A"]["delta_pp"] == pytest.approx(-10.0)
    assert subgroups["B"]["delta_pp"] == pytest.approx(5.0)
    assert math.isnan(subgroups["C"]["preview"])

    secondary = evaluation_report["secondary_metrics"]
    assert len(secondary) == 1 and secondary[0]["kind"] == "valid_metric"

    validation = evaluation_report["validation"]
    assert validation["spectral_stable"] is False
    assert validation["rmt_stable"] is False
    assert validation.get("moe_observed") is True
    assert validation.get("moe_identity_ok") is True

    guard = evaluation_report["guard_overhead"]
    assert guard["evaluated"] is True
    assert guard["passed"] is True
    assert guard["bare_ppl"] == pytest.approx(10.0)
    assert guard["guarded_ppl"] == pytest.approx(10.05)


def test_make_evaluation_report_policy_digest_changes_when_policy_override_differs(
    monkeypatch,
):
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_baseline",
        lambda value: value,
        raising=False,
    )
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["guards"].append(
        {"name": "spectral", "policy": {"max_caps": 10}, "metrics": {"caps_applied": 0}}
    )
    baseline["guards"] = []
    baseline["meta"]["auto"]["tier"] = "conservative"

    evaluation_report = make_report(report, baseline)

    assert evaluation_report["policy_digest"]["changed"] is True


def test_make_evaluation_report_provenance_and_guard_schedule_fallback(monkeypatch):
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_baseline",
        lambda value: value,
        raising=False,
    )
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["provenance"] = {}
    report["evaluation_windows"]["final"]["window_ids"] = [11, 12, 13]
    report["guard_overhead"] = {}
    report["metrics"]["window_plan"]["profile"] = "dev"

    evaluation_report = make_report(report, baseline)

    prov = evaluation_report["provenance"]
    assert "provider_digest" in prov
    assert (
        prov.get("window_ids_digest")
        == evaluation_report["guard_overhead"]["schedule_digest"]
        == prov["provider_digest"]["ids_sha256"]
    )


def test_make_evaluation_report_embeds_telemetry_summary(monkeypatch):
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization_mod,
        "normalize_baseline",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)

    evaluation_report = make_report(report, baseline)
    assert evaluation_report["telemetry"]["summary_line"].startswith(
        "INVARLOCK_TELEMETRY"
    )
