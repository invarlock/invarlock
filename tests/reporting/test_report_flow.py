from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.reporting.report import save_report, to_certificate


def _minimal_report() -> dict:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "abc",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "auto": None,
        },
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
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
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 100.0,
                "preview": 100.0,
                "ratio_vs_baseline": 1.0,
            },
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 256.0,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_to_certificate_unsupported_format_raises():
    rep = _minimal_report()
    base = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
    }
    with pytest.raises(ValueError):
        to_certificate(rep, base, format="xml")


def test_save_report_with_compare_suffix(tmp_path: Path):
    rep = _minimal_report()
    out = save_report(
        rep,
        tmp_path,
        formats=["json", "markdown", "html"],
        compare=rep,
        filename_prefix="r",
    )
    # Ensure comparison suffix reflected
    assert out["json"].name.endswith("_comparison.json")
    assert out["markdown"].name.endswith("_comparison.md")
    assert out["html"].name.endswith("_comparison.html")


def test_generate_comparison_markdown_with_guard_violations():
    from invarlock.reporting.report import to_markdown

    rep1 = _minimal_report()
    rep2 = _minimal_report()
    # Add guard entries with violations to both
    rep1["guards"] = [
        {
            "name": "spectral",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": ["cap_exceeded"],
        }
    ]
    rep2["guards"] = [
        {
            "name": "spectral",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": ["cap_exceeded", "another"],
        }
    ]
    md = to_markdown(rep1, compare=rep2)
    # Expect Guard Reports section
    assert "Guard Reports" in md and "Violations" in md
