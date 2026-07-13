from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.evidence_catalog import catalog_digest
from invarlock.evidence_pack import verify_evidence_pack
from invarlock.evidence_pack_support import EvidencePackStatus
from invarlock.reporting.verify_contract import run_verify_reports
from invarlock.reporting.verify_contract_types import VerifyOutcome
from scripts.model_evidence.run_catalog_lane import (
    CatalogLaneArtifacts,
    CatalogLaneError,
    _run,
    _run_environment,
    assemble_signed_catalog_pack,
    build_evaluate_command,
    run_catalog_lane,
    validate_staging_output,
)
from tests.reporting.evidence_pack._support_catalog_evidence import (
    IMAGE_DIGEST,
    SOURCE_BUNDLE_DIGEST,
    SOURCE_COMMIT,
    write_catalog_evidence_fixture,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _catalog_fixture(tmp_path: Path):
    root = tmp_path / "fixture"
    root.mkdir()
    return write_catalog_evidence_fixture(root)


def _verification_receipt(path: Path, *, ok: bool = True) -> Path:
    return write_json(
        path,
        {
            "component": "cli",
            "format_version": "verify-v1",
            "results": [{"ok": ok, "reason": "ok" if ok else "failed"}],
            "summary": {"ok": ok, "reason": "ok" if ok else "failed"},
        },
    )


def test_assemble_signed_catalog_pack_passes_the_real_strict_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _catalog_fixture(tmp_path)
    binding = write_json(
        tmp_path / "evaluation-input-binding.json",
        fixture.evaluation_binding,
    )
    verification = _verification_receipt(tmp_path / "verify.json")
    artifacts = CatalogLaneArtifacts(
        catalog=fixture.catalog,
        lane_id="text-a",
        evaluation_report=fixture.report,
        runtime_manifest=fixture.runtime_manifest,
        baseline_report=fixture.baseline,
        policy_pack=fixture.policy_pack,
        resolved_inputs=fixture.resolved_inputs,
        resolved_config=fixture.runtime_config,
        preset=fixture.preset,
        evaluation_input_binding=binding,
        verification_receipt=verification,
        source_commit=SOURCE_COMMIT,
        source_bundle_sha256=SOURCE_BUNDLE_DIGEST,
    )

    pack, fingerprint = assemble_signed_catalog_pack(
        artifacts,
        tmp_path / "staged-pack",
        signing_key=fixture.signing_key,
    )
    report_result = run_verify_reports(
        [fixture.report],
        baseline=fixture.baseline,
        policy_pack=fixture.policy_pack,
        profile="ci",
        assurance_mode="strict",
        expected_runtime_image_digest=IMAGE_DIGEST,
    )
    assert report_result.outcome is VerifyOutcome.OK, report_result.diagnostics
    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda *_args, **_kwargs: report_result,
    )

    result = verify_evidence_pack(
        pack,
        strict=True,
        report_assurance="strict",
        expected_fingerprint=fingerprint,
        expected_catalog_digest=catalog_digest(fixture.catalog),
        expected_runtime_image_digest=IMAGE_DIGEST,
        policy_pack_path=fixture.policy_pack,
    )
    assert result.status is EvidencePackStatus.OK, result.payload
    assert result.payload["ok"] is True
    assert not (pack / "results" / "verification.json").exists()
    assert (pack / "metadata" / "evaluation-input-binding.json").is_file()


def test_assembler_rejects_a_failed_report_verification_receipt(tmp_path: Path) -> None:
    fixture = _catalog_fixture(tmp_path)
    artifacts = CatalogLaneArtifacts(
        catalog=fixture.catalog,
        lane_id="text-a",
        evaluation_report=fixture.report,
        runtime_manifest=fixture.runtime_manifest,
        baseline_report=fixture.baseline,
        policy_pack=fixture.policy_pack,
        resolved_inputs=fixture.resolved_inputs,
        resolved_config=fixture.runtime_config,
        preset=fixture.preset,
        evaluation_input_binding=write_json(
            tmp_path / "evaluation-input-binding.json",
            fixture.evaluation_binding,
        ),
        verification_receipt=_verification_receipt(tmp_path / "verify.json", ok=False),
        source_commit=SOURCE_COMMIT,
        source_bundle_sha256=SOURCE_BUNDLE_DIGEST,
    )

    with pytest.raises(
        CatalogLaneError, match="strict report verification did not pass"
    ):
        assemble_signed_catalog_pack(
            artifacts,
            tmp_path / "staged-pack",
            signing_key=fixture.signing_key,
        )
    assert not (tmp_path / "staged-pack").exists()


def test_evaluate_command_is_derived_from_the_catalog_and_pinned_inputs(
    tmp_path: Path,
) -> None:
    fixture = _catalog_fixture(tmp_path)

    command = build_evaluate_command(
        catalog=fixture.catalog,
        lane_id="text-a",
        resolved_inputs=fixture.resolved_inputs,
        prepared_preset=tmp_path / "prepared-preset.yaml",
        evaluation_input_binding=tmp_path / "evaluation-input-binding.json",
        work_dir=tmp_path / "work",
        device="cuda",
        allow_network=True,
    )

    rendered = " ".join(command)
    assert "--baseline strict-test-model" in rendered
    assert "--subject strict-test-model" in rendered
    assert "--baseline-revision " + ("b" * 40) in rendered
    assert "--subject-revision " + ("b" * 40) in rendered
    assert "--profile release" in rendered
    assert "--tier balanced" in rendered
    assert "--assurance strict" in rendered
    assert "--execution-mode container" in rendered
    assert "--allow-network" in command
    assert "public_evidence" not in rendered


def test_run_environment_propagates_explicit_network_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")

    allowed = _run_environment(
        runtime_image="invarlock-runtime:test",
        runtime_image_digest=IMAGE_DIGEST,
        source_commit=SOURCE_COMMIT,
        source_bundle_sha256=SOURCE_BUNDLE_DIGEST,
        allow_network=True,
    )
    denied = _run_environment(
        runtime_image="invarlock-runtime:test",
        runtime_image_digest=IMAGE_DIGEST,
        source_commit=SOURCE_COMMIT,
        source_bundle_sha256=SOURCE_BUNDLE_DIGEST,
        allow_network=False,
    )

    assert allowed["INVARLOCK_ALLOW_NETWORK"] == "1"
    assert denied["INVARLOCK_ALLOW_NETWORK"] == "0"


def test_run_failure_preserves_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            ["invarlock", "evaluate"],
            output="actionable failure detail",
            stderr="non-fatal download warning",
        )

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(CatalogLaneError) as captured:
        _run(["invarlock", "evaluate"], cwd=tmp_path, env={})

    message = str(captured.value)
    assert "actionable failure detail" in message
    assert "non-fatal download warning" in message


def test_catalog_lane_can_retain_a_failed_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _catalog_fixture(tmp_path)
    failed_workspace = tmp_path / "failed-workspace"
    repo_root = tmp_path / "repo"
    preset = repo_root / "configs/presets/catalog-test.yaml"
    preset.parent.mkdir(parents=True)
    preset.write_bytes(fixture.preset.read_bytes())

    def fail_after_preparation(*, workspace: Path, **_kwargs: object) -> None:
        (workspace / "diagnostic.json").write_text("{}\n", encoding="utf-8")
        raise CatalogLaneError("diagnostic failure")

    monkeypatch.setattr(
        "scripts.model_evidence.run_catalog_lane._prepare_lane_inputs",
        fail_after_preparation,
    )
    monkeypatch.setattr("scripts.model_evidence.run_catalog_lane.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "scripts.model_evidence.run_catalog_lane.PUBLIC_EVIDENCE_ROOT",
        repo_root / "public_evidence",
    )
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    args = SimpleNamespace(
        lane="text-a",
        catalog=fixture.catalog,
        resolved_inputs=fixture.resolved_inputs,
        policy_pack=fixture.policy_pack,
        signing_key=fixture.signing_key,
        runtime_image="invarlock-runtime:test",
        runtime_image_digest=IMAGE_DIGEST,
        source_commit=SOURCE_COMMIT,
        source_bundle_sha256=SOURCE_BUNDLE_DIGEST,
        out=tmp_path / "candidate-pack",
        device="cpu",
        allow_network=False,
        failed_workspace_out=failed_workspace,
    )

    with pytest.raises(CatalogLaneError, match="diagnostic failure"):
        run_catalog_lane(args)

    assert (failed_workspace / "diagnostic.json").is_file()
    assert not list(tmp_path.glob(".candidate-pack.run.*"))


def test_staging_output_cannot_publish_or_overwrite(tmp_path: Path) -> None:
    with pytest.raises(CatalogLaneError, match="outside public_evidence"):
        validate_staging_output(REPO_ROOT / "public_evidence" / "published_basis" / "x")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(CatalogLaneError, match="already exists"):
        validate_staging_output(existing)


def test_pack_contains_no_absolute_execution_paths(tmp_path: Path) -> None:
    fixture = _catalog_fixture(tmp_path)
    verification = _verification_receipt(tmp_path / "verify.json")
    receipt = json.loads(verification.read_text(encoding="utf-8"))
    receipt["results"][0]["id"] = str(tmp_path / "report/evaluation.report.json")
    write_json(verification, receipt)
    artifacts = CatalogLaneArtifacts(
        catalog=fixture.catalog,
        lane_id="text-a",
        evaluation_report=fixture.report,
        runtime_manifest=fixture.runtime_manifest,
        baseline_report=fixture.baseline,
        policy_pack=fixture.policy_pack,
        resolved_inputs=fixture.resolved_inputs,
        resolved_config=fixture.runtime_config,
        preset=fixture.preset,
        evaluation_input_binding=write_json(
            tmp_path / "evaluation-input-binding.json",
            fixture.evaluation_binding,
        ),
        verification_receipt=verification,
        source_commit=SOURCE_COMMIT,
        source_bundle_sha256=SOURCE_BUNDLE_DIGEST,
    )
    pack, _fingerprint = assemble_signed_catalog_pack(
        artifacts,
        tmp_path / "staged-pack",
        signing_key=fixture.signing_key,
    )

    for path in pack.rglob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in payload
        json.loads(payload)
