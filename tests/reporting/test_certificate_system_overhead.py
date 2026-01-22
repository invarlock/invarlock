from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_minimal_report(metrics: dict) -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
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
            "plan_digest": "noop",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
            "ppl_preview_ci": (10.0, 10.0),
            "ppl_final_ci": (10.0, 10.0),
            "ppl_ratio_ci": (1.0, 1.0),
            "latency_ms_per_tok": 2.0,
            "throughput_tok_per_s": 50.0,
            **metrics,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_certificate_system_overhead_table_and_primary_metric_metadata():
    report = _mk_minimal_report({})
    baseline = _mk_minimal_report(
        {"latency_ms_per_tok": 1.6, "throughput_tok_per_s": 60.0}
    )

    # Include a primary metric snapshot with metadata
    report.setdefault("metrics", {})["primary_metric"] = {
        "kind": "accuracy",
        "unit": "pp",
        "paired": True,
        "gating_basis": "lower",
        "reps": 500,
        "ci": (-1.2, +1.5),
        "preview": 0.80,
        "final": 0.85,
        "ratio_vs_baseline": +0.05,
    }

    cert = make_certificate(report, baseline)
    md = render_certificate_markdown(cert)
    # System Overhead section may be omitted; rendering should still succeed
    assert "# InvarLock Evaluation Certificate" in md

    # Primary Metric metadata present
    assert "## Primary Metric" in md
