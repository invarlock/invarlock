from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from invarlock.reporting.html import render_report_html
from invarlock.reporting.report_schema import validate_report
from invarlock.runtime_verify import verify_runtime_manifest


def test_structural_failure_report_marks_structural_detection(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evidence_packs" / "python" / "task_tools.py"
    source_report = tmp_path / "source.report.json"
    source_report.write_text(
        json.dumps(
            {
                "run_id": "source-run",
                "meta": {
                    "model_id": "m",
                    "adapter": "hf_causal",
                    "seed": 7,
                    "device": "cpu",
                },
                "data": {
                    "dataset": "dummy",
                    "split": "validation",
                    "seq_len": 8,
                    "stride": 8,
                    "preview_n": 2,
                    "final_n": 2,
                },
                "edit": {
                    "name": "noop",
                    "plan_digest": "x",
                    "deltas": {"params_changed": 0, "layers_modified": 0},
                },
                "guards": [],
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "unit": "ppl",
                        "direction": "lower",
                        "aggregation_scope": "token",
                        "paired": True,
                        "gating_basis": "upper",
                        "supports_bootstrap": True,
                        "preview": 9.429,
                        "final": 8.893,
                        "drift_band": {"min": 0.8878, "max": 1.0859},
                    }
                },
                "evaluation_windows": {
                    "final": {
                        "window_ids": [1, 2],
                        "logloss": [2.30, 2.31],
                        "token_counts": [100, 100],
                    }
                },
                "artifacts": {
                    "events_path": "",
                    "logs_path": "",
                    "checkpoint_path": None,
                },
                "flags": {"guard_recovered": False, "rollback_reason": None},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_runtime_manifest = tmp_path / "source.runtime.manifest.json"
    source_runtime_manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "generated_at_utc": "2026-04-19T08:00:00+00:00",
                "verifier_contract_version": "runtime-manifest-v1",
                "report": {
                    "path": str(source_report.resolve()),
                    "filename": source_report.name,
                    "sha256": hashlib.sha256(source_report.read_bytes()).hexdigest(),
                },
                "config": {"path": None, "sha256": None, "source": "missing"},
                "execution_mode": "container",
                "runtime": {
                    "image_ref": f"invarlock-runtime:cuda-local@sha256:{'a' * 64}",
                    "image_digest": f"sha256:{'a' * 64}",
                    "container_execution": True,
                    "allow_network": True,
                    "allow_remote_code": True,
                    "allow_third_party_plugins": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "evaluation.report.json"

    subprocess.run(
        [
            "python3",
            str(script),
            "structural-failure-report",
            "--error-type",
            "inf_injection",
            "--source-report",
            str(source_report),
            "--source-runtime-manifest",
            str(source_runtime_manifest),
            "--message",
            "Window pairing mismatch detected (fraction=0.000, reason=preview_missing_ids:[0, 1, 2])",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert validate_report(payload)
    assert payload["validation"]["invariants_pass"] is False
    assert payload["validation"]["primary_metric_acceptable"] is False
    assert payload["primary_metric"]["invalid"] is True
    assert payload["primary_metric"]["degraded"] is True
    assert payload["invariants"]["status"] == "fail"
    assert payload["_evidence_pack_structural_failure"]["error_type"] == "inf_injection"
    assert payload["_evidence_pack_structural_failure"]["source_report"] == str(
        source_report
    )
    assert payload["schema_version"] == "v1"
    assert payload["run_id"].endswith("structural-failure-inf_injection")

    html = render_report_html(payload)
    assert "<html" in html.lower()

    runtime_manifest = out_path.parent / "runtime.manifest.json"
    assert runtime_manifest.is_file()
    verify = verify_runtime_manifest(out_path, runtime_manifest)
    assert verify.ok, verify.errors
