from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.reporting.report import save_report


def _minimal_run_report() -> dict:
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
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "deadbeef",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [
            {
                "name": "variance",
                "policy": {
                    "deadband": 0.02,
                    "min_abs_adjust": 0.01,
                    "max_scale_step": 0.03,
                },
                "metrics": {},
                "actions": [],
                "violations": [],
            }
        ],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 100.0,
                "preview": 100.0,
                "ratio_vs_baseline": 1.0,
            },
            "latency_ms_per_tok": 1.0,
            "memory_mb_peak": 256.0,
            "bootstrap": {
                "replicates": 10,
                "alpha": 0.05,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [4.0, 4.0],
                "token_counts": [100, 100],
            }
        },
    }


def _baseline_v1() -> dict:
    # Minimal baseline-v1 that relies on evaluation_windows rather than explicit PM
    return {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 4.0}},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [4.0, 4.0],
                "token_counts": [100, 100],
            }
        },
    }


def test_save_report_cert_bundle_writes_manifest_and_evidence(
    tmp_path: Path, monkeypatch
):
    rep = _minimal_run_report()
    base = _baseline_v1()
    # Gate small debug evidence emission via env
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    out = save_report(
        rep, tmp_path, formats=["cert"], baseline=base, filename_prefix="r"
    )
    assert out["cert"].exists()
    assert out["cert_md"].exists()
    # Manifest is best-effort but should exist in this path
    assert (tmp_path / "manifest.json").exists()
    # Evidence file gets created when env is set (even when payload is tiny)
    assert (tmp_path / "guards_evidence.json").exists()


def test_save_report_cert_requires_baseline(tmp_path: Path):
    rep = _minimal_run_report()
    with pytest.raises(ValueError):
        save_report(rep, tmp_path, formats=["cert"], baseline=None)
