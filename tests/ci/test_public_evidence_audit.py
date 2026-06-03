from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from invarlock.evidence_pack import EvidencePackStatus, verify_evidence_pack
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "checks" / "check_public_evidence.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("check_public_evidence", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_evidence_audit_passes() -> None:
    module = _load_audit_module()
    assert module.check_public_evidence() == []


def test_public_evidence_audit_respects_root_override(tmp_path: Path) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "fixtures" / "demo"
    artifact_dir.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    (artifact_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "runtime.manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": "contract_fixture",
                "summary": "fixture report",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )

    assert module.check_public_evidence(evidence_root) == []


def test_real_run_reports_and_signed_packs_verify_release_strict() -> None:
    real_run_dirs = sorted((REPO_ROOT / "public_evidence" / "real_runs").iterdir())
    assert real_run_dirs

    for evidence_dir in real_run_dirs:
        if not evidence_dir.is_dir():
            continue
        metadata_path = evidence_dir / "evidence.meta.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("evidence_class") != "real_model_run":
            continue

        report_path = evidence_dir / "evaluation.report.json"

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )
        assert result.outcome == VerifyOutcome.OK
        verification = result.payload["results"][0]["verification"]
        assert verification["runtime_provenance"]["status"] == "verified"

        pack_result = verify_evidence_pack(
            evidence_dir / "evidence_pack",
            strict=True,
            profile="release",
            report_assurance="strict",
            expected_fingerprint=metadata["expected_fingerprint"],
        )
        assert pack_result.status == EvidencePackStatus.OK
        assert pack_result.payload["authenticity"] == "pinned"


def test_external_byoe_real_run_uses_custom_subject_checkpoint() -> None:
    evidence_dir = (
        REPO_ROOT
        / "public_evidence"
        / "real_runs"
        / "tiny_gpt2_external_magnitude_prune"
    )
    metadata = json.loads(
        (evidence_dir / "evidence.meta.json").read_text(encoding="utf-8")
    )
    report_path = evidence_dir / "evaluation.report.json"
    refs_path = evidence_dir / "checkpoint_refs.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    refs = json.loads(refs_path.read_text(encoding="utf-8"))

    assert metadata["evidence_class"] == "real_model_run"
    assert "fixture" not in metadata["summary"].lower()
    assert report["edit"]["name"] == "custom"
    assert report["edit_name"] == "custom"
    assert report["plugins"] == {}
    assert refs["weights_vendored"] is False
    assert refs["subject_checkpoint"]["external_edit_type"] == "magnitude_prune"
    assert refs["subject_checkpoint"]["built_in_edit_plugin"] is False
    assert refs["subject_checkpoint"]["materialized_by"] == "external_edit_recipe.py"
