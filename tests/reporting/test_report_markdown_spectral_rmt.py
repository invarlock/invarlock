from __future__ import annotations

from invarlock.reporting import report_make as C
from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_make import make_report


def _mk_report() -> dict:
    import math

    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {"name": "noop", "plan_digest": "noop"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 50.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (0.98, 1.02),
            },
            "bootstrap": {"replicates": 50, "alpha": 0.1},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [math.log(50.0), math.log(50.0)],
                "token_counts": [10, 20],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [math.log(50.0), math.log(50.0)],
                "token_counts": [10, 20],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def _cert_skeleton() -> dict:
    return {
        "schema_version": C.REPORT_SCHEMA_VERSION,
        "run_id": "r1",
        "edit_name": "noop",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "display_ci": [1.0, 1.0],
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "seed": 1,
        },
        "auto": {"tier": "balanced", "probes_used": 0},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {"preview": 1, "final": 1, "seed": 1},
            "hash": {"preview_tokens": 10, "final_tokens": 10, "total_tokens": 20},
            "tokenizer": {},
        },
        "policy_digest": {
            "policy_version": C.POLICY_VERSION,
            "thresholds_hash": "deadbeefcafe0123",
            "changed": True,
        },
        "confidence": {"label": "High"},
        "provenance": {
            "provider_digest": {
                "tokenizer_sha256": "t" * 20,
                "ids_sha256": "i" * 20,
                "masking_sha256": "m" * 20,
            }
        },
    }


def test_render_report_markdown_general_sections() -> None:
    # Build and render a real evaluation_report; spot-check core headings render
    report = _mk_report()
    baseline = _mk_report()
    cert = make_report(report, baseline)
    out = render_report_markdown(cert)
    assert "InvarLock Evaluation Report" in out
    assert "Executive Summary" in out
    assert "Primary Metric" in out
