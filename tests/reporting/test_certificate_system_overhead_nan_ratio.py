from __future__ import annotations

import math

from invarlock.reporting.certificate import make_certificate


def _reports_with_sys_overhead_zero_base() -> tuple[dict, dict]:
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            # Provide explicit p50 latency for edited
            "latency_ms_p50": 20.0,
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "meta": {"model_id": "m"},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
            # Explicit p50 latency baseline = 0 → ratio becomes NaN
            "latency_ms_p50": 0.0,
        },
    }
    return report, baseline


def test_system_overhead_ratio_nan_when_baseline_zero() -> None:
    rep, base = _reports_with_sys_overhead_zero_base()
    cert = make_certificate(rep, base)
    sys = cert.get("system_overhead", {})
    assert isinstance(sys, dict)
    entry = sys.get("latency_ms_p50") or sys.get("latency_ms_per_tok")
    assert isinstance(entry, dict)
    ratio = entry.get("ratio")
    assert isinstance(ratio, float) and math.isnan(ratio)
