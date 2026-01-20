from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.commands.explain_gates import explain_gates_command
from invarlock.cli.doctor_helpers import get_adapter_rows


def test_doctor_helpers_rows(monkeypatch):
    # Avoid ModuleNotFoundError from importlib.util.find_spec on nested packages
    import importlib.util as iu

    orig_find_spec = iu.find_spec

    def _safe_find_spec(name, package=None):  # type: ignore[override]
        try:
            return orig_find_spec(name, package)
        except ModuleNotFoundError:
            return None

    monkeypatch.setattr(iu, "find_spec", _safe_find_spec)

    rows = get_adapter_rows()
    assert isinstance(rows, list) and rows
    names = {r["name"] for r in rows}
    # Expect some known adapters
    assert {"hf_causal", "hf_causal_onnx"}.issubset(names)
    # When extras missing, hf_causal_onnx should show needs_extra
    onnx_row = next((r for r in rows if r["name"] == "hf_causal_onnx"), None)
    assert onnx_row is not None
    # status can be ready if extras installed; tolerate both
    assert onnx_row["status"] in {"needs_extra", "ready"}


def _make_min_report():
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 1,
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
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
            "plan_digest": "abc",
            "deltas": {
                "params_changed": 1,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 1,
            },
        },
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def _make_baseline():
    return {
        "schema_version": "baseline-v1",
        "meta": {"commit": "beefdead"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }


def test_explain_gates_happy_and_missing(tmp_path: Path):
    rpt = _make_min_report()
    base = _make_baseline()
    rp = tmp_path / "r.json"
    bp = tmp_path / "b.json"
    rp.write_text(json.dumps(rpt), encoding="utf-8")
    bp.write_text(json.dumps(base), encoding="utf-8")
    # Happy path prints content
    explain_gates_command(report=str(rp), baseline=str(bp))
    # Missing file triggers exit(1)
    import click

    with pytest.raises(click.exceptions.Exit):
        explain_gates_command(report=str(rp), baseline=str(tmp_path / "missing.json"))
