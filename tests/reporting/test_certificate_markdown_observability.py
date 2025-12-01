from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_render_certificate_markdown_observability_sections():
    rep = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "ts": "2024-01-01T00:00:00",
        },
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 4.0,
                "final": 4.0,
                "ratio_vs_baseline": 1.0,
            },
            "classification": {
                "subgroups": {
                    "preview": {"group_counts": {"A": 10}, "correct_counts": {"A": 8}},
                    "final": {"group_counts": {"A": 12}, "correct_counts": {"A": 10}},
                }
            },
            "latency_ms_per_tok": 1.5,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    base = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 4.0},
            "latency_ms_per_tok": 1.6,
        }
    }
    cert = make_certificate(rep, base)
    md = render_certificate_markdown(cert)
    assert "System Overhead" in md
    assert "Policy Version:" in md and "Thresholds Digest:" in md
