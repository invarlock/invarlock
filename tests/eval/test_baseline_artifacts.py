"""Unit tests for baseline utilities and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.reporting.certificate import _normalize_baseline
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.validate import save_baseline as _save_baseline


def _build_baseline_report(ppl_final: float) -> dict:
    """Construct a minimal baseline RunReport for normalization tests."""
    report = create_empty_report()

    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "commit": "abc123deadbeef",
            "seed": 42,
        }
    )

    report["data"].update(
        {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 256,
            "stride": 128,
            "preview_n": 50,
            "final_n": 100,
        }
    )

    report["metrics"].update(
        {
            "ppl_preview": ppl_final,
            "ppl_final": ppl_final,
            "ppl_ratio": 1.0 if ppl_final > 0 else 0.0,
            "spectral": {},
            "rmt": {},
            "invariants": {},
        }
    )

    report["edit"].update(
        {
            "name": "baseline",
            "plan_digest": "baseline_noop",
        }
    )
    report["edit"]["deltas"]["params_changed"] = 0
    report["edit"]["plan"] = {"target_sparsity": 0.0}

    return report


def test_normalize_baseline_falls_back_for_invalid_ppl():
    """Invalid baseline PPL values should fall back to the computed constant."""
    baseline = _build_baseline_report(ppl_final=0.0)

    normalized = _normalize_baseline(baseline)

    assert normalized["ppl_final"] == pytest.approx(50.797)
    assert normalized["ppl_preview"] == pytest.approx(50.797)


def test_normalize_baseline_preserves_valid_values():
    """Valid baseline inputs should be preserved without modification."""
    baseline = _build_baseline_report(ppl_final=35.2)
    baseline["metrics"]["ppl_preview"] = 34.8

    normalized = _normalize_baseline(baseline)

    assert normalized["ppl_final"] == pytest.approx(35.2)
    assert normalized["ppl_preview"] == pytest.approx(34.8)


def test_save_baseline_metrics_serializes_expected_schema(tmp_path: Path):
    """Write a baseline-v1 payload and verify expected fields."""
    output_path = tmp_path / "baseline.json"

    payload = {
        "schema_version": "baseline-v1",
        "metrics": {"ppl_final": 42.0, "ppl_ratio": 1.0},
        "dataset": {"split": "validation"},
    }
    _save_baseline(payload, output_path)

    with output_path.open() as fp:
        out = json.load(fp)

    assert out["schema_version"] == "baseline-v1"
    assert out["metrics"]["ppl_final"] == pytest.approx(42.0)
    assert out["dataset"]["split"] == "validation"
    assert out["metrics"]["ppl_ratio"] == 1.0
