from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "_shared"
    / "validate_source_matrix_artifacts.py"
)


def _load_validator():
    module_name = "source_matrix_artifact_validator"
    spec = importlib.util.spec_from_file_location(module_name, VALIDATOR)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_matrix_artifact_set(report_dir: Path) -> None:
    report_dir.mkdir(parents=True)
    (report_dir / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": True, "reason": "ok"},
                "results": [
                    {
                        "verification": {
                            "runtime_provenance": {
                                "declared_mode": "container",
                                "verified": True,
                            }
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "runtime.manifest.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "evaluation.html").write_text("<html></html>\n", encoding="utf-8")
    (report_dir / "backend_inventory.json").write_text(
        '{"backend": "hqq"}\n', encoding="utf-8"
    )
    (report_dir / "lane_artifact.json").write_text(
        json.dumps({"lane_artifact_label": "cuda-container-strict"}) + "\n",
        encoding="utf-8",
    )
    (report_dir / "run_command.txt").write_text(
        "wrapper: run_tiny_hf_hqq.sh --lane cuda\n", encoding="utf-8"
    )
    (report_dir / "run_summary.txt").write_text(
        "status: success\n"
        "lane_artifact_label: cuda-container-strict\n"
        "verify_status: ok\n"
        "verify_runtime_provenance_declared: container\n"
        "verify_runtime_provenance_verified: true\n",
        encoding="utf-8",
    )
    (report_dir / "checkpoint_refs.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "adapter_runtime_summary.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "fixture_summary.json").write_text("{}\n", encoding="utf-8")


def _write_test_source_matrix(repo_root: Path) -> Path:
    matrix_path = repo_root / "examples" / "integrations" / "source_matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "invarlock.integration_source_matrix.v1",
                "entries": [
                    {
                        "target": "hqq",
                        "readme": "examples/integrations/hqq/README.md",
                        "runner": "examples/integrations/hqq/run_tiny_hf_hqq.sh",
                        "report_path": "reports/tiny-hf-hqq/<artifact-lane>",
                        "lane": "cuda-container-strict",
                        "expected": {
                            "lane_artifact_label": "cuda-container-strict",
                            "verify_status": "ok",
                            "runtime_provenance_declared": "container",
                            "runtime_provenance_verified": True,
                        },
                        "required_artifacts": [
                            "evaluation.report.json",
                            "verify.json",
                            "runtime.manifest.json",
                            "evaluation.html",
                            "backend_inventory.json",
                            "lane_artifact.json",
                            "run_command.txt",
                            "run_summary.txt",
                            "checkpoint_refs.json",
                            "adapter_runtime_summary.json",
                            "fixture_summary.json",
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix_path


def _report_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "examples"
        / "integrations"
        / "hqq"
        / "reports"
        / "tiny-hf-hqq"
        / "cuda-container-strict"
    )


def test_source_matrix_artifact_validator_accepts_complete_artifacts(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    _write_matrix_artifact_set(_report_dir(tmp_path))

    selected, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
    )

    assert selected == ["hqq"]
    assert issues == []


def test_source_matrix_artifact_validator_reports_artifact_and_status_mismatches(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    (report_dir / "backend_inventory.json").unlink()
    (report_dir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": False, "reason": "policy_fail"},
                "results": [
                    {
                        "verification": {
                            "runtime_provenance": {
                                "declared_mode": "host",
                                "verified": False,
                            }
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
    )
    messages = [issue.message for issue in issues]

    assert "required artifact is missing" in messages
    assert any("verify status mismatch" in message for message in messages)
    assert any(
        "runtime provenance declared mode mismatch" in message for message in messages
    )
    assert any(
        "runtime provenance verified flag mismatch" in message for message in messages
    )
