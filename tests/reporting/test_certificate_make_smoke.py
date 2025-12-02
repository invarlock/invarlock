from typing import Any

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _minimal_run_report_with_windows(kind: str = "ppl_causal") -> dict[str, Any]:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "abc",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00Z",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
            "tokenizer_name": "tok",
        },
        "edit": {
            "name": "noop",
            "plan_digest": "deadbeef",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {"kind": kind, "preview": 100.0, "final": 101.0},
            "logloss_preview": 4.6052,
            "logloss_final": 4.6152,
            "preview_total_tokens": 100,
            "final_total_tokens": 100,
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "paired_windows": 2,
            "bootstrap": {"replicates": 200},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [4.6052, 4.6052],
                "input_ids": [[1, 2], [3, 4]],
            },
            "final": {
                "window_ids": [1, 2],
                "logloss": [4.6152, 4.6152],
                "input_ids": [[5, 6], [7, 8]],
            },
        },
    }


def test_make_certificate_smoke_and_render():
    report = _minimal_run_report_with_windows()
    baseline = _minimal_run_report_with_windows()

    cert = make_certificate(report, baseline)
    assert isinstance(cert, dict)
    assert "primary_metric" in cert and "dataset" in cert
    assert cert.get("meta", {}).get("seed") == 1
    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict)
    # Markdown render should succeed and contain some baseline info
    md = render_certificate_markdown(cert)
    assert isinstance(md, str) and len(md) > 0
