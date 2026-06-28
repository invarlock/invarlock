from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.reporting import report_bundle as report_bundle_mod
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
    evaluation_report = make_report(rep, base)

    out = save_evaluation_bundle(
        run_report=rep,
        output_dir=tmp_path,
        evaluation_report=evaluation_report,
    )
    assert out["report"].exists()
    assert out["report_md"].exists()
    report_text = out["report"].read_text(encoding="utf-8")
    assert report_text.endswith("\n")
    assert "\n  " not in report_text
    assert json.loads(report_text) == json.loads(json.dumps(evaluation_report))
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
    assert not (tmp_path / "evidence_summary.txt").exists()
    assert not (tmp_path / "guards_evidence.json").exists()


def test_save_report_bundle_rejects_invalid_evaluation_report(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid evaluation report"):
        save_evaluation_bundle(
            run_report=_minimal_run_report(),
            output_dir=tmp_path,
            evaluation_report={"schema_version": "not-valid"},
        )


def test_save_report_bundle_copies_inventory_and_invalid_runtime_manifest(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_report_path = run_dir / "report.json"
    run_report_path.write_text("{}", encoding="utf-8")
    inventory_path = run_dir / "backend_inventory.json"
    inventory_path.write_text('{"backend": "test"}', encoding="utf-8")
    runtime_manifest_path = run_dir / RUNTIME_MANIFEST_FILENAME
    runtime_manifest_path.write_text("{not-json", encoding="utf-8")

    out = save_evaluation_bundle(
        run_report=_minimal_run_report(),
        output_dir=tmp_path / "out",
        evaluation_report=make_report(_minimal_run_report(), _baseline_v1()),
        source_run_path=run_report_path,
    )

    assert out["backend_inventory"].read_text(encoding="utf-8") == '{"backend": "test"}'
    assert out["runtime_manifest"].read_text(encoding="utf-8") == "{not-json"


def test_save_report_bundle_keeps_same_directory_source_sidecars(tmp_path: Path):
    run_report_path = tmp_path / "report.json"
    run_report_path.write_text("{}", encoding="utf-8")
    inventory_path = tmp_path / "backend_inventory.json"
    inventory_path.write_text('{"backend": "same-dir"}', encoding="utf-8")
    runtime_manifest_path = tmp_path / RUNTIME_MANIFEST_FILENAME
    runtime_manifest_path.write_text("{not-json", encoding="utf-8")

    out = save_evaluation_bundle(
        run_report=_minimal_run_report(),
        output_dir=tmp_path,
        evaluation_report=make_report(_minimal_run_report(), _baseline_v1()),
        source_run_path=run_report_path,
        render_optional=False,
    )

    assert out["backend_inventory"] == inventory_path
    assert out["runtime_manifest"] == runtime_manifest_path
    assert inventory_path.read_text(encoding="utf-8") == '{"backend": "same-dir"}'
    assert runtime_manifest_path.read_text(encoding="utf-8") == "{not-json"


def test_save_report_bundle_ignores_invalid_source_run_path(tmp_path: Path):
    out = save_evaluation_bundle(
        run_report=_minimal_run_report(),
        output_dir=tmp_path / "out",
        evaluation_report=make_report(_minimal_run_report(), _baseline_v1()),
        source_run_path=tmp_path / "missing" / "report.json",
        render_optional=False,
    )

    assert "runtime_manifest" not in out


def test_save_report_bundle_records_generated_backend_inventory(
    tmp_path: Path, monkeypatch
):
    generated_inventory = tmp_path / "generated_backend_inventory.json"

    def _write_backend_inventory_sidecar(*_args, **_kwargs):
        generated_inventory.write_text('{"backend": "generated"}', encoding="utf-8")
        return generated_inventory

    monkeypatch.setattr(
        report_bundle_mod,
        "write_backend_inventory_sidecar",
        _write_backend_inventory_sidecar,
    )

    out = save_evaluation_bundle(
        run_report=_minimal_run_report(),
        output_dir=tmp_path,
        evaluation_report=make_report(_minimal_run_report(), _baseline_v1()),
        render_optional=False,
    )

    assert out["backend_inventory"] == generated_inventory
    assert generated_inventory.exists()


def test_write_report_manifest_is_best_effort(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        report_bundle_mod,
        "build_report_manifest_summary",
        lambda *_: (_ for _ in ()).throw(ValueError("bad summary")),
    )
    saved_files: dict[str, Path] = {}

    report_bundle_mod.write_report_manifest(
        report=_minimal_run_report(),
        output_path=tmp_path,
        evaluation_report={},
        report_json_path=tmp_path / "evaluation.report.json",
        report_md_path=tmp_path / "evaluation_report.md",
        saved_files=saved_files,
    )

    assert saved_files == {}
    assert not (tmp_path / "manifest.json").exists()


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
