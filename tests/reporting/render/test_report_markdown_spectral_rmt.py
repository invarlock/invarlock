from __future__ import annotations

from invarlock.reporting import report_make as C
from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import canonical_run_report
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _mk_report() -> dict:
    import math

    return canonical_run_report(
        {
            "meta": {
                "model_id": "gpt2",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 42,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev"},
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
    )


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
            "guard_metric_impact_acceptable": True,
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


def test_guard_markdown_uses_caps_language_and_warning_section() -> None:
    cert = _cert_skeleton()
    cert["spectral"] = {"caps_applied": 1, "max_caps": 5}
    cert["guard_warnings"] = {
        "present": True,
        "warning_count": 1,
        "warnings": [
            {
                "guard": "spectral",
                "kind": "new_capped_module",
                "severity": "warning",
                "family": "ffn",
                "module": "layers.31.mlp.up_proj",
                "policy_gate": "pass",
                "message": (
                    "Policy passes, but subject has a new capped module versus baseline."
                ),
            }
        ],
    }
    cert["validation"]["guard_warnings_present"] = True
    cert["validation"]["guard_warning_policy_acceptable"] = True

    out = render_report_markdown(cert)

    assert "1 caps applied" in out
    assert "<= 5" in out
    assert "1 violations" not in out
    assert "## Guard Warnings" in out
    assert "Policy passes" in out
