"""Unit tests for baseline utilities and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.reporting.report_normalization import normalize_baseline
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.validate import save_baseline as _save_baseline


def _build_baseline_report(ppl_final: float) -> dict:
    """Construct a minimal baseline RunReport for normalization tests."""
    report = create_empty_report()

    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
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
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": ppl_final,
                "final": ppl_final,
            },
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
            "name": "noop",
            "plan_digest": "baseline_noop",
        }
    )
    report["edit"]["deltas"]["params_changed"] = 0
    report["edit"]["plan"] = {"target_sparsity": 0.0}

    return report


def test_normalize_baseline_raises_for_invalid_ppl():
    """Invalid baseline PPL values should fail closed."""
    baseline = _build_baseline_report(ppl_final=0.0)

    with pytest.raises(ValueError, match="Invalid canonical RunReport structure"):
        normalize_baseline(baseline)


def test_normalize_baseline_preserves_valid_values():
    """Valid baseline inputs should be preserved without modification."""
    baseline = _build_baseline_report(ppl_final=35.2)
    baseline["metrics"]["ppl_preview"] = 34.8

    normalized = normalize_baseline(baseline)

    assert normalized["ppl_final"] == pytest.approx(35.2)
    assert normalized["ppl_preview"] == pytest.approx(34.8)


def test_save_baseline_metrics_serializes_expected_schema(tmp_path: Path):
    """Write a canonical baseline RunReport and verify expected fields."""
    output_path = tmp_path / "baseline.json"
    payload = _build_baseline_report(ppl_final=42.0)
    _save_baseline(payload, output_path)

    with output_path.open() as fp:
        out = json.load(fp)

    assert out["edit"]["name"] == "noop"
    assert out["metrics"]["ppl_final"] == pytest.approx(42.0)
    assert out["data"]["split"] == "validation"
    assert out["metrics"]["ppl_ratio"] == 1.0
