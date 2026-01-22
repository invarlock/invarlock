from __future__ import annotations

from typing import Any

from invarlock.reporting.certificate import make_certificate


def _mock_report_with_seed_and_device() -> dict[str, Any]:
    return {
        "meta": {
            "model_id": "stub",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 11,
            "seeds": {"python": 11},
        },
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
            },
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_device_and_seeds_captured() -> None:
    report = _mock_report_with_seed_and_device()
    baseline = _mock_report_with_seed_and_device()
    cert = make_certificate(report, baseline)
    meta = cert.get("meta", {})
    assert meta.get("seed") == 11
    seeds = meta.get("seeds", {})
    assert isinstance(seeds, dict)
    assert seeds.get("python") == 11
    assert isinstance(meta.get("device"), str)
