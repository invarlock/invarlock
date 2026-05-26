from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.reporting.report_bundle import save_evaluation_bundle
from invarlock.reporting.report_files import save_report
from invarlock.reporting.report_make import make_report
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


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


def test_save_report_bundle_writes_manifest_and_evidence(tmp_path: Path, monkeypatch):
    rep = _minimal_run_report()
    base = _baseline_v1()
    # Gate small debug evidence emission via env
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    out = save_evaluation_bundle(
        run_report=rep,
        output_dir=tmp_path,
        evaluation_report=make_report(rep, base),
    )
    assert out["report"].exists()
    assert out["report_md"].exists()
    # Manifest is best-effort but should exist in this path
    assert (tmp_path / "manifest.json").exists()
    # Evidence file gets created when env is set (even when payload is tiny)
    assert (tmp_path / "guards_evidence.json").exists()


def test_save_report_bundle_can_defer_optional_rendering(tmp_path: Path, monkeypatch):
    rep = _minimal_run_report()
    base = _baseline_v1()
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    out = save_evaluation_bundle(
        run_report=rep,
        output_dir=tmp_path,
        evaluation_report=make_report(rep, base),
        render_optional=False,
    )

    assert out["report"].exists()
    assert "report_md" not in out
    assert not (tmp_path / "evaluation_report.md").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "reviewer_summary.txt").exists()
    assert not (tmp_path / "guards_evidence.json").exists()


def test_save_report_requires_baseline(tmp_path: Path):
    rep = _minimal_run_report()
    with pytest.raises(ValueError, match="save_evaluation_bundle"):
        save_report(rep, tmp_path, formats=["report"])


def test_save_report_bundle_copies_runtime_manifest_when_source_run_path_provided(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_report_path = run_dir / "report.json"
    run_report_path.write_text("{}", encoding="utf-8")
    runtime_manifest_path = run_dir / RUNTIME_MANIFEST_FILENAME
    runtime_manifest_path.write_text(
        """
        {
          "execution_mode": "container",
          "manifest_version": 1,
          "runtime": {
            "container_execution": true,
            "image_ref": "local",
            "image_digest": "sha256:test"
          },
          "report": {
            "filename": "report.json",
            "path": "/tmp/run/report.json",
            "sha256": "old"
          },
          "verifier_contract_version": "runtime-manifest-v1"
        }
        """.strip(),
        encoding="utf-8",
    )

    rep = _minimal_run_report()
    base = _baseline_v1()

    out = save_evaluation_bundle(
        run_report=rep,
        output_dir=tmp_path / "out",
        evaluation_report=make_report(rep, base),
        source_run_path=run_report_path,
    )

    copied_manifest = tmp_path / "out" / RUNTIME_MANIFEST_FILENAME
    assert copied_manifest.exists()
    assert out["runtime_manifest"] == copied_manifest
    manifest_payload = json.loads(copied_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["runtime"] == {
        "container_execution": True,
        "image_ref": "local",
        "image_digest": "sha256:test",
    }
    assert manifest_payload["report"]["filename"] == "evaluation.report.json"
    assert manifest_payload["report"]["path"] == str(
        tmp_path / "out" / "evaluation.report.json"
    )
    assert (
        manifest_payload["report"]["sha256"]
        == hashlib.sha256(
            (tmp_path / "out" / "evaluation.report.json").read_bytes()
        ).hexdigest()
    )
