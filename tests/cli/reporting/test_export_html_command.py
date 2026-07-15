from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.commands.report import export_html_command
from invarlock.reporting.report_make import make_report
from tests.cli._support_runtime_policy import bind_runtime_policy


def _make_min_report():
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 1,
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
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
            "name": "quant_rtn",
            "plan_digest": "abc123",
            "deltas": {
                "params_changed": 1,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 1,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            "preview_total_tokens": 100,
            "final_total_tokens": 100,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def _make_baseline():
    baseline = _make_min_report()
    baseline["meta"]["commit"] = "beefdead"
    baseline["edit"]["name"] = "noop"
    baseline["edit"]["deltas"]["params_changed"] = 0
    return baseline


def test_export_html_happy_and_no_overwrite(tmp_path: Path):
    report = _make_min_report()
    base = _make_baseline()
    report_json = json.dumps(
        make_report(bind_runtime_policy(report), bind_runtime_policy(base)),
        indent=2,
        ensure_ascii=False,
    )
    inp = tmp_path / "cert.json"
    out = tmp_path / "out.html"
    inp.write_text(report_json, encoding="utf-8")

    # First export, strip css
    export_html_command(input=str(inp), output=str(out), embed_css=False, force=False)
    html = out.read_text(encoding="utf-8")
    assert "<html" in html and "<style" not in html

    # Attempt overwrite without --force should exit(1)
    import typer

    with pytest.raises(typer.Exit):
        export_html_command(
            input=str(inp), output=str(out), embed_css=True, force=False
        )
