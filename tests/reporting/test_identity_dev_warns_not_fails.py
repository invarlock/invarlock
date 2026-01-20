from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def test_identity_deviation_in_dev_does_not_raise() -> None:
    # Construct a minimal report with drift ratio 1.0 but paired delta implying a different ratio
    report = {
        "meta": {"model_id": "bert-tiny", "adapter": "hf_mlm", "seed": 42},
        "metrics": {
            "paired_delta_summary": {
                "mean": 0.5,
                "degenerate": False,
            },  # exp(0.5)=1.65 → mismatch
            "window_plan": {"profile": "dev"},
            "primary_metric": {"kind": "ppl_mlm", "preview": 100.0, "final": 100.0},
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2, 3],
                "logloss": [1.0, 1.2, 0.9],
                "token_counts": [10, 10, 10],
            }
        },
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 128,
            "windows": {"preview": 1, "final": 1},
        },
        "artifacts": {},
    }
    baseline = {
        "meta": {"model_id": "bert-tiny"},
        "metrics": {"primary_metric": {"kind": "ppl_mlm", "final": 50.0}},
    }
    # Should not raise
    cert = make_certificate(report, baseline)
    assert isinstance(cert, dict)
