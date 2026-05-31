from __future__ import annotations

import hashlib
import json
from pathlib import Path

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_edit_metadata as edit_metadata_mod
from invarlock.evidence_pack import (
    EvidencePackStatus,
    validate_manifest,
    verify_evidence_pack,
    verify_manifest_provenance,
)
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_ref(path: Path, rel_path: str) -> dict[str, str]:
    return {
        "path": rel_path,
        "digest": f"sha256:{_sha256_file(path)}",
    }


def _write_runtime_manifest(report_path: Path) -> None:
    _write_json(
        report_path.parent / RUNTIME_MANIFEST_FILENAME,
        {
            "manifest_version": 1,
            "generated_at_utc": "2026-03-21T00:00:00+00:00",
            "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
            "execution_mode": "container",
            "report": {
                "filename": report_path.name,
                "path": report_path.as_posix(),
                "sha256": _sha256_file(report_path),
            },
            "config": {
                "path": None,
                "sha256": None,
                "source": "missing",
            },
            "runtime": {
                "container_execution": True,
                "image_digest": _VALID_TEST_IMAGE_DIGEST,
                "image_ref": "invarlock-runtime:local",
                "allow_network": False,
                "allow_remote_code": False,
                "allow_third_party_plugins": False,
            },
        },
    )


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_pack(pack_dir: Path, *, report_rel_path: str) -> Path:
    final_verdict = pack_dir / "results/final_verdict.json"
    source_repo = pack_dir / "metadata/source_repo.json"
    environment = pack_dir / "metadata/environment.json"
    materials = pack_dir / "metadata/model_revisions.json"
    report = pack_dir / report_rel_path

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(materials, {"models": {"org/model": {"revision": "rev1"}}})
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    _write_runtime_manifest(report)

    covered = [
        "results/final_verdict.json",
        "metadata/source_repo.json",
        "metadata/environment.json",
        "metadata/model_revisions.json",
        report_rel_path,
        str((Path(report_rel_path).parent / RUNTIME_MANIFEST_FILENAME).as_posix()),
    ]
    _write_checksums(pack_dir, covered)

    manifest = {
        "format": "evidence-pack-v1",
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_file(pack_dir / "checksums.sha256"),
        "subject": {
            "name": "final_verdict",
            **_digest_ref(final_verdict, "results/final_verdict.json"),
        },
        "invocation": {
            "config_source": _digest_ref(source_repo, "metadata/source_repo.json")
        },
        "environment": _digest_ref(environment, "metadata/environment.json"),
        "materials": [
            {
                "name": "model_revisions",
                **_digest_ref(materials, "metadata/model_revisions.json"),
            }
        ],
    }
    _write_json(pack_dir / "manifest.json", manifest)
    return pack_dir


def _allow_unsigned_pack(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_pack_mod,
        "_verify_signature",
        lambda pack_dir, strict: ([], [], None),
        raising=True,
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
        raising=True,
    )


def test_evidence_pack_manifest_and_provenance_round_trip(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    assert validate_manifest(pack_dir / "manifest.json") == []
    assert verify_manifest_provenance(pack_dir) == []

    result = verify_evidence_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == EvidencePackStatus.SIGNATURE
    assert payload["ok"] is False
    assert payload["errors"] == [
        "manifest.signature.json missing; signed manifest required by default."
    ]


def test_evidence_pack_verify_rejects_json_out_inside_pack(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = verify_evidence_pack(
        pack_dir, json_out_path=pack_dir / "verify.json", skip_verify=True
    )
    payload = result.payload
    exit_code = result.status

    assert exit_code == EvidencePackStatus.USAGE
    assert payload["ok"] is False
    assert "--json-out must point outside the pack directory." in payload["errors"]


def test_evidence_pack_verify_strict_rejects_extra_files(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    (pack_dir / "extra.txt").write_text("extra", encoding="utf-8")
    original_verify_signature = evidence_pack_mod._verify_signature
    evidence_pack_mod._verify_signature = lambda pack_dir, strict: ([], [], None)

    try:
        result = verify_evidence_pack(pack_dir, skip_verify=True, strict=True)
    finally:
        evidence_pack_mod._verify_signature = original_verify_signature

    payload = result.payload
    exit_code = result.status
    assert exit_code == EvidencePackStatus.INTEGRITY
    assert payload["ok"] is False
    assert any("extra files not covered" in error for error in payload["errors"])


def test_evidence_pack_verify_requires_clean_reports(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/errors/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
        raising=True,
    )

    result = verify_evidence_pack(pack_dir)
    payload = result.payload
    exit_code = result.status

    assert exit_code == EvidencePackStatus.REPORTS
    assert payload["ok"] is False
    assert any("No reports expected to pass" in error for error in payload["errors"])


def test_evidence_pack_verify_requires_validation_edit_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/quant_4bit_clean/run_1/evaluation.report.json",
    )
    _write_json(
        pack_dir / "metadata/scenarios.json",
        {
            "scenarios": [
                {
                    "id": "quant_4bit_clean",
                    "artifact_class": "validation_subject_checkpoint",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": "quant_rtn:clean:ffn",
                    },
                }
            ]
        },
    )
    _allow_unsigned_pack(monkeypatch)

    result = verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status == EvidencePackStatus.INTEGRITY
    assert any(
        "quant_4bit_clean: edit_metadata.json missing next to report" in error
        for error in result.payload["errors"]
    )


def test_evidence_pack_verify_requires_deployable_sidecars(
    monkeypatch, tmp_path: Path
) -> None:
    report_rel = "reports/model/deploy_bnb_8bit_clean/run_1/evaluation.report.json"
    pack_dir = _build_pack(tmp_path / "pack", report_rel_path=report_rel)
    _write_json(
        pack_dir / "metadata/scenarios.json",
        {
            "scenarios": [
                {
                    "id": "deploy_bnb_8bit_clean",
                    "artifact_class": "deployable_optimized_subject",
                    "generation": {
                        "kind": "deployable_edit",
                        "edit_spec": "bnb_8bit:clean:ffn",
                    },
                }
            ]
        },
    )
    _write_json(
        pack_dir / Path(report_rel).parent / "edit_metadata.json",
        {
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "deployable_optimized_subject",
            "edit_type": "bnb_8bit",
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
            "coverage": {},
        },
    )
    _allow_unsigned_pack(monkeypatch)

    result = verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status == EvidencePackStatus.INTEGRITY
    assert any(
        "deploy_bnb_8bit_clean: deployable sidecar missing" in error
        for error in result.payload["errors"]
    )


def test_evidence_pack_metadata_helper_edges(tmp_path: Path) -> None:
    assert (
        edit_metadata_mod._infer_scenario_artifact_class(
            {"artifact_class": "custom_class", "generation": {"kind": "error"}}
        )
        == "custom_class"
    )
    assert (
        edit_metadata_mod._infer_scenario_artifact_class(
            {"generation": {"kind": "error"}}
        )
        == "fault_injection_fixture"
    )
    assert (
        edit_metadata_mod._infer_scenario_artifact_class(
            {"generation": {"kind": "deployable_edit"}}
        )
        == "deployable_optimized_subject"
    )
    assert (
        edit_metadata_mod._infer_scenario_artifact_class(
            {"generation": {"kind": "edit"}}
        )
        == "validation_subject_checkpoint"
    )
    assert edit_metadata_mod._infer_scenario_artifact_class({}) == ""

    invalid_pack = tmp_path / "invalid-pack"
    (invalid_pack / "metadata").mkdir(parents=True)
    (invalid_pack / "metadata" / "scenarios.json").write_text("{", encoding="utf-8")
    assert edit_metadata_mod._scenario_index_from_pack(invalid_pack) == {}

    non_list_pack = tmp_path / "non-list-pack"
    _write_json(non_list_pack / "metadata" / "scenarios.json", {"scenarios": {}})
    assert edit_metadata_mod._scenario_index_from_pack(non_list_pack) == {}

    mixed_pack = tmp_path / "mixed-pack"
    _write_json(
        mixed_pack / "metadata" / "scenarios.json",
        {"scenarios": ["bad", {"id": ""}, {"id": "ok"}]},
    )
    assert edit_metadata_mod._scenario_index_from_pack(mixed_pack) == {
        "ok": {"id": "ok"}
    }

    assert (
        edit_metadata_mod._report_scenario_id(
            mixed_pack,
            tmp_path / "outside" / "evaluation.report.json",
        )
        is None
    )
    assert (
        edit_metadata_mod._report_scenario_id(
            mixed_pack,
            mixed_pack / "reports" / "evaluation.report.json",
        )
        is None
    )
    assert (
        edit_metadata_mod._report_scenario_id(
            mixed_pack,
            mixed_pack
            / "reports"
            / "model"
            / "errors"
            / "bad"
            / "evaluation.report.json",
        )
        == "bad"
    )

    assert (
        edit_metadata_mod._load_json_sidecar(
            invalid_pack / "metadata" / "scenarios.json"
        )[0]
        is None
    )
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("[]", encoding="utf-8")
    assert edit_metadata_mod._load_json_sidecar(sidecar) == (
        None,
        "JSON sidecar must contain an object",
    )

    assert (
        edit_metadata_mod._expected_edit_type(
            {"failure_class": "deployable_edit.bnb_8bit"}
        )
        == "bnb_8bit"
    )
    assert edit_metadata_mod._expected_edit_type({}) == ""

    deployable_errors = edit_metadata_mod._metadata_consistency_errors(
        scenario_id="deploy",
        spec={
            "artifact_class": "deployable_optimized_subject",
            "generation": {"edit_spec": "bnb_8bit:clean:ffn"},
        },
        metadata={
            "schema": "wrong",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "other",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
        },
    )
    assert any("unrecognized schema" in error for error in deployable_errors)
    assert any("artifact_class mismatch" in error for error in deployable_errors)
    assert any("edit_type mismatch" in error for error in deployable_errors)
    assert any(
        "optimized_deployment_backend=true" in error for error in deployable_errors
    )
    assert any("packed_quantized_storage=true" in error for error in deployable_errors)

    validation_errors = edit_metadata_mod._metadata_consistency_errors(
        scenario_id="quant",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "quant_rtn:clean:ffn"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "quant_rtn",
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
        },
    )
    assert any(
        "optimized_deployment_backend=false" in error for error in validation_errors
    )
    assert any("packed_quantized_storage=false" in error for error in validation_errors)

    assert (
        edit_metadata_mod._metadata_consistency_errors(
            scenario_id="quant",
            spec={
                "artifact_class": "validation_subject_checkpoint",
                "generation": {"edit_spec": "quant_rtn:clean:ffn"},
            },
            metadata={
                "schema": "invarlock/evidence-pack-edit-metadata-v1",
                "artifact_class": "validation_subject_checkpoint",
                "edit_type": "quant_rtn",
                "optimized_deployment_backend": False,
                "packed_quantized_storage": False,
            },
        )
        == []
    )


def test_evidence_pack_metadata_consistency_helper_edges(tmp_path: Path) -> None:
    assert edit_metadata_mod._verify_edit_metadata_consistency(tmp_path / "empty") == []

    pack_dir = tmp_path / "pack"
    _write_json(
        pack_dir / "metadata" / "scenarios.json",
        {
            "scenarios": [
                {"id": "short", "artifact_class": "validation_subject_checkpoint"},
                {"id": "fault", "artifact_class": "fault_injection_fixture"},
                {
                    "id": "badmeta",
                    "artifact_class": "validation_subject_checkpoint",
                    "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn"},
                },
                {
                    "id": "valid_validation",
                    "artifact_class": "validation_subject_checkpoint",
                    "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn"},
                },
                {
                    "id": "deploy_missing_report",
                    "artifact_class": "deployable_optimized_subject",
                    "generation": {
                        "kind": "deployable_edit",
                        "edit_spec": "bnb_8bit:clean:ffn",
                    },
                },
                {
                    "id": "deploy",
                    "artifact_class": "deployable_optimized_subject",
                    "generation": {
                        "kind": "deployable_edit",
                        "edit_spec": "bnb_8bit:clean:ffn",
                    },
                },
            ]
        },
    )
    (pack_dir / "reports").mkdir()
    (pack_dir / "reports" / "evaluation.report.json").write_text(
        "{}",
        encoding="utf-8",
    )
    fault_report = pack_dir / "reports" / "model" / "fault" / "run_1"
    fault_report.mkdir(parents=True)
    (fault_report / "evaluation.report.json").write_text("{}", encoding="utf-8")

    badmeta_report = pack_dir / "reports" / "model" / "badmeta" / "run_1"
    badmeta_report.mkdir(parents=True)
    (badmeta_report / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (badmeta_report / "edit_metadata.json").write_text("[", encoding="utf-8")

    valid_report = pack_dir / "reports" / "model" / "valid_validation" / "run_1"
    valid_report.mkdir(parents=True)
    (valid_report / "evaluation.report.json").write_text("{}", encoding="utf-8")
    _write_json(
        valid_report / "edit_metadata.json",
        {
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "quant_rtn",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
        },
    )

    deploy_report = pack_dir / "reports" / "model" / "deploy" / "run_1"
    deploy_report.mkdir(parents=True)
    (deploy_report / "evaluation.report.json").write_text("{}", encoding="utf-8")
    _write_json(
        deploy_report / "edit_metadata.json",
        {
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "deployable_optimized_subject",
            "edit_type": "bnb_8bit",
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
        },
    )
    _write_json(deploy_report / "deployable_artifact_validation.json", {"ok": False})
    (deploy_report / "backend_inventory.json").write_text("[]", encoding="utf-8")
    _write_json(deploy_report / "memory_report.json", {"ok": False})
    _write_json(
        deploy_report / "load_smoke.json",
        {"schema": "invarlock/deployable-load-smoke-v1", "ok": False},
    )
    _write_json(deploy_report / "inference_smoke.json", {"ok": True})

    errors = edit_metadata_mod._verify_edit_metadata_consistency(pack_dir)

    assert any("badmeta: edit_metadata.json invalid" in error for error in errors)
    assert any(
        "deployable sidecar invalid (backend_inventory.json)" in error
        for error in errors
    )
    assert any(
        "deployable sidecar did not pass: deployable_artifact_validation.json" in error
        for error in errors
    )
    assert any(
        "deployable sidecar did not pass: memory_report.json" in error
        for error in errors
    )
    assert any(
        "deployable sidecar schema mismatch (memory_report.json)" in error
        for error in errors
    )
    assert any(
        "deployable sidecar did not pass: load_smoke.json" in error for error in errors
    )
    assert any(
        "deploy_missing_report: deployable scenario has no deployability report sidecars"
        in error
        for error in errors
    )


def test_evidence_pack_metadata_consistency_skips_non_mapping_specs(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_dir = pack_dir / "reports" / "model" / "bad" / "run_1"
    report_dir.mkdir(parents=True)
    (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        edit_metadata_mod,
        "_scenario_index_from_pack",
        lambda _pack_dir: {"bad": "not-a-mapping"},
        raising=True,
    )

    assert edit_metadata_mod._verify_edit_metadata_consistency(pack_dir) == []


def test_verify_reports_runs_nested_verification_with_assurance_off(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    json_out = tmp_path / "verify.json"
    seen: list[tuple[list[Path], str, str]] = []

    def fake_run_verify_command(
        reports: list[Path], *, profile: str, report_assurance: str = "report"
    ) -> VerifyExecutionResult:
        seen.append((reports, profile, report_assurance))
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={
                "ok": True,
                "profile": profile,
                "report_assurance": report_assurance,
                "reports": len(reports),
            },
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        fake_run_verify_command,
        raising=True,
    )

    errors, payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=json_out,
        profile="dev",
        report_assurance="off",
    )

    assert errors == []
    assert payload == {
        "ok": True,
        "profile": "dev",
        "report_assurance": "off",
        "reports": 1,
    }
    assert json.loads(json_out.read_text(encoding="utf-8")) == payload
    assert seen[0][1:] == ("dev", "off")


def test_evidence_pack_invalid_report_assurance_modes(tmp_path: Path) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report_path, {"ok": True})

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        report_assurance="weak",
    )
    assert result.status == EvidencePackStatus.USAGE
    assert any(
        "Report assurance must be" in error for error in result.payload["errors"]
    )

    verify_result = verify_evidence_pack(
        tmp_path / "missing",
        report_assurance="weak",
        skip_verify=True,
    )
    assert verify_result.status == EvidencePackStatus.USAGE
    assert any(
        "--report-assurance must be" in error
        for error in verify_result.payload["errors"]
    )


def test_release_review_requires_explicit_profile_and_pass_verdict(
    tmp_path: Path,
) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    runtime_manifest = tmp_path / RUNTIME_MANIFEST_FILENAME
    _write_json(final_verdict, {"verdict": "WARN"})
    _write_json(report_path, {"ok": True})
    _write_json(runtime_manifest, {"ok": True})

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-release-review",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        profile="",
        report_assurance="strict",
        signing_key_path=tmp_path / "signing.key",
        release_review=True,
    )

    assert result.status == EvidencePackStatus.USAGE
    assert any("explicit profile" in error for error in result.payload["errors"])
    assert any("final verdict PASS" in error for error in result.payload["errors"])


def test_release_review_rejects_invalid_final_verdict_json(tmp_path: Path) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    signing_key = tmp_path / "signing.key"
    final_verdict.write_text("{", encoding="utf-8")
    _write_json(report_path, {"ok": True})
    evidence_pack_mod._generate_signing_keypair(
        signing_key,
        public_key_path=signing_key.with_suffix(".pub"),
    )

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-release-review-invalid-json",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        profile="ci",
        report_assurance="strict",
        signing_key_path=signing_key,
        release_review=True,
    )

    assert result.status == EvidencePackStatus.USAGE
    assert any(
        "Final verdict is not valid JSON" in error for error in result.payload["errors"]
    )


def test_release_review_build_passes_hardened_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    runtime_manifest = tmp_path / RUNTIME_MANIFEST_FILENAME
    signing_key = tmp_path / "signing.key"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report_path, {"ok": True})
    _write_json(runtime_manifest, {"ok": True})
    evidence_pack_mod._generate_signing_keypair(
        signing_key,
        public_key_path=signing_key.with_suffix(".pub"),
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-release-review-ok",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        profile="ci",
        report_assurance="strict",
        signing_key_path=signing_key,
        release_review=True,
    )

    assert result.status == EvidencePackStatus.OK
    assert result.payload["release_review"] is True
