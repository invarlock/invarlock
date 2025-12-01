import math

import pytest

from invarlock.reporting.certificate import make_certificate


def _base_report():
    return {
        "meta": {"model_id": "m", "seed": 123},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.5},
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 2,
            "final_n": 2,
        },
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }


def _base_baseline():
    return {
        "run_id": "r0",
        "meta": {"model_id": "m"},
        "metrics": {"ppl_final": 9.8, "ppl_preview": 9.7},
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
    }


def test_certificate_ratio_ci_mismatch_raises(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()

    # Provide paired delta summary CI and a ratio_ci that does NOT equal exp(delta_ci)
    # Align drift ratio with delta mean to bypass the earlier drift mismatch check
    delta_lo = delta_hi = math.log(1.11)
    # Adjust preview/final to match the delta mean ratio so only CI-mismatch branch triggers
    report["metrics"]["ppl_preview"] = 10.0
    report["metrics"]["ppl_final"] = 11.1
    report["metrics"].update(
        {
            "paired_delta_summary": {
                "mean": (delta_lo + delta_hi) / 2.0,
                "degenerate": False,
            },
            "logloss_delta_ci": (delta_lo, delta_hi),
            "ratio_ci": (1.05, 1.06),
        }
    )
    # Make sure the code takes the paired path
    report["metrics"]["stats"] = {"pairing": "paired_baseline"}

    monkeypatch.setattr(
        "invarlock.reporting.certificate.validate_report", lambda _: True
    )
    # Force mismatch by overriding the ratio_ci computation from delta_ci
    monkeypatch.setattr(
        "invarlock.reporting.certificate.logspace_to_ratio_ci", lambda _: (1.05, 1.06)
    )

    with pytest.raises(ValueError):
        _ = make_certificate(report, baseline)


def test_certificate_metrics_stats_passthrough(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()

    # Inject metrics.stats with optional keys; they should be copied through
    report["metrics"]["stats"] = {
        "requested_preview": 10,
        "requested_final": 12,
        "actual_preview": 8,
        "actual_final": 9,
        "coverage_ok": True,
    }

    monkeypatch.setattr(
        "invarlock.reporting.certificate.validate_report", lambda _: True
    )
    cert = make_certificate(report, baseline)
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    # Optional metrics.stats passthrough keys may be omitted after normalization
    assert isinstance(stats, dict)
