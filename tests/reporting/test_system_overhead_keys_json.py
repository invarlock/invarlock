from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_report_with_overhead(metrics: dict) -> dict:
    return {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 42},
        "metrics": {
            # primary metric and overhead metrics
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (1.0, 1.0),
            },
            # overhead metrics
            **metrics,
        },
        "evaluation_windows": {
            "preview": {"window_ids": [1], "logloss": [1.0], "token_counts": [10]},
            "final": {"window_ids": [2], "logloss": [1.0], "token_counts": [10]},
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_system_overhead_json_keys():
    report = _mk_report_with_overhead(
        {"latency_ms_p50": 2.0, "latency_ms_p95": 3.5, "throughput_sps": 77.7}
    )
    baseline = _mk_report_with_overhead(
        {"latency_ms_p50": 1.5, "latency_ms_p95": 3.0, "throughput_sps": 80.0}
    )
    cert = make_certificate(report, baseline)
    so = cert.get("system_overhead", {})
    assert isinstance(so, dict)
