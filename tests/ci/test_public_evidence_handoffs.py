from __future__ import annotations

import hashlib
import json
from pathlib import Path

from invarlock.evidence_pack import (
    EvidencePackStatus,
    _generate_signing_keypair,
    build_evidence_pack,
    verify_evidence_pack,
)
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json_file(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_docs_route_to_mistral_guard_value_evidence() -> None:
    docs = {
        "README.md": REPO_ROOT / "README.md",
        "public_evidence/README.md": REPO_ROOT / "public_evidence" / "README.md",
        "scripts/evidence_packs/README.md": (
            REPO_ROOT / "scripts" / "evidence_packs" / "README.md"
        ),
        "docs/user-guide/evidence-packs-internals.md": (
            REPO_ROOT / "docs" / "user-guide" / "evidence-packs-internals.md"
        ),
        "docs/reference/guards.md": REPO_ROOT / "docs" / "reference" / "guards.md",
    }

    for label, path in docs.items():
        text = path.read_text(encoding="utf-8")
        assert "public_evidence/published_basis/mistral_7b/guard_value_demo" in text, (
            label
        )
        assert "baseline-relative" in text, label

    public_evidence_readme = docs["public_evidence/README.md"].read_text(
        encoding="utf-8"
    )
    assert "PM-only accepts" in public_evidence_readme
    assert "PM+guards" in public_evidence_readme
    assert "caught_regressions/` entries remain useful verifier fixtures" in (
        public_evidence_readme
    )

    pack_readme = docs["scripts/evidence_packs/README.md"].read_text(encoding="utf-8")
    assert "Guard-value publishing rule" in pack_readme
    assert "Clean confirmation reruns are required" in pack_readme
    assert "guard_value_all_guard_probe_sweep.json" in pack_readme


def test_policy_failure_fixtures_fail_expected_policy_predicate() -> None:
    cases = {
        "invariants_failure": (
            "validation.invariants_pass == true",
            "invariants did not pass",
        ),
        "primary_metric_failure": (
            "Primary metric policy gate failed",
            "validation.primary_metric_acceptable == true",
        ),
        "runtime_provenance_failure": (
            "runtime.manifest.json marks evaluation.report.json as 'host-bypass'",
            "strict assurance requires verified runtime provenance",
        ),
    }

    for directory, expected_messages in cases.items():
        report_path = (
            REPO_ROOT
            / "public_evidence"
            / "policy_failures"
            / directory
            / "evaluation.report.json"
        )

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.POLICY_FAIL
        diagnostics = "\n".join(item.message for item in result.diagnostics)
        for expected in expected_messages:
            assert expected in diagnostics


def test_byoe_examples_verify_release_strict() -> None:
    examples = {
        "magnitude_prune_byoe": "magnitude_prune",
        "lora_merge_byoe": "lora_merge",
        "fine_tune_byoe": "fine_tune",
    }

    for directory, edit_type in examples.items():
        example_dir = REPO_ROOT / "public_evidence" / "byoe_examples" / directory
        report_path = example_dir / "evaluation.report.json"
        refs_path = example_dir / "checkpoint_refs.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        refs = json.loads(refs_path.read_text(encoding="utf-8"))

        assert validate_report(report) is True
        assert report["artifacts"]["byoe_example"] is True
        assert report["artifacts"]["external_edit_type"] == edit_type
        assert report["artifacts"]["built_in_edit_plugin"] is False
        assert report["plugins"]["edits"] == []
        assert refs["weights_vendored"] is False
        assert refs["subject_checkpoint"]["external_edit_type"] == edit_type
        assert refs["subject_checkpoint"]["built_in_edit_plugin"] is False
        if directory in {"lora_merge_byoe", "fine_tune_byoe"}:
            edit = report["edit"]
            assert edit["edit_provenance"]["edit_family"] == edit_type
            assert edit["edit_provenance"]["edit_count"] == 1
            assert edit["edit_provenance"]["dynamic_runtime_required"] is False
            if directory == "lora_merge_byoe":
                assert edit["edit_provenance"]["edit_method"] == "custom"
                assert edit["edit_impact"]["scenario_types"] == [
                    "target_success",
                    "near_neighbor",
                    "unrelated_locality",
                    "general_ability_sentinel",
                ]
            else:
                assert (
                    edit["edit_provenance"]["edit_method"]
                    == "external_cpu_tiny_fine_tune"
                )
            assert (
                refs["subject_checkpoint"]["edit_provenance"]
                == (edit["edit_provenance"])
            )
            if "edit_impact" in edit:
                assert refs["subject_checkpoint"]["edit_impact"] == edit["edit_impact"]

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.OK
        verification = result.payload["results"][0]["verification"]
        assert verification["runtime_provenance"]["status"] == "verified"


def test_model_editing_evidence_bundle_v0_lanes_verify_release_strict() -> None:
    bundle_dir = REPO_ROOT / "public_evidence" / "model_editing_evidence_bundle_v0"
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = REPO_ROOT / manifest["verification_summary"]
    training_plan_path = REPO_ROOT / manifest["training_evidence_matrix_plan"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_families = {"quantization", "magnitude_prune", "lora_merge", "fine_tune"}
    lanes = manifest["lanes"]
    assert {lane["edit_family"] for lane in lanes} == expected_families
    assert manifest["evidence_scope"] == "release-evidence wiring only"
    assert summary["schema"] == (
        "invarlock.public_evidence.model_editing_bundle_verification.v1"
    )
    assert summary["bundle_id"] == manifest["bundle_id"]
    assert summary["evidence_scope"] == manifest["evidence_scope"]
    assert summary["verification"] == {
        "assurance": "strict",
        "lane_count": 4,
        "outcome": "all_lanes_verified",
        "profile": "release",
    }
    assert training_plan_path.is_file()
    training_plan = training_plan_path.read_text(encoding="utf-8")
    assert "PEFT LoRA train-and-merge subject" in training_plan
    assert "Full fine-tune subject" in training_plan
    assert "/private/tmp" not in training_plan
    assert "root@" not in training_plan

    summary_lanes = {lane["edit_family"]: lane for lane in summary["lanes"]}
    assert set(summary_lanes) == expected_families
    assert {
        summary_lanes["quantization"]["evidence_mode"],
        summary_lanes["magnitude_prune"]["evidence_mode"],
    } == {"real_tiny_model_run", "real_tiny_model_external_edit_run"}
    assert summary_lanes["lora_merge"]["evidence_mode"] == "public_byoe_subject_fixture"
    assert summary_lanes["fine_tune"]["evidence_mode"] == "public_byoe_subject_fixture"

    for lane in lanes:
        report_path = REPO_ROOT / lane["evaluation_report"]
        refs_path = REPO_ROOT / lane["checkpoint_refs"]
        note_path = REPO_ROOT / lane["evidence_note"]
        summary_lane = summary_lanes[lane["edit_family"]]

        report = json.loads(report_path.read_text(encoding="utf-8"))
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        note = " ".join(note_path.read_text(encoding="utf-8").split())

        assert validate_report(report) is True
        assert (
            refs["subject_checkpoint"]["external_edit_type"]
            == lane["external_edit_type"]
        )
        assert "Evidence takeaways" in note
        assert "Artifact mode:" in note
        assert "Verification surface:" in note
        assert "Companion benchmark evidence:" in note
        assert "/private/tmp" not in note
        assert "root@" not in note
        assert summary_lane["external_edit_type"] == lane["external_edit_type"]
        assert summary_lane["weights_vendored"] is False
        assert summary_lane["strict_verification"] == {
            "assurance": "strict",
            "outcome": "ok",
            "profile": "release",
            "runtime_provenance_status": "verified",
        }
        for key, expected_path in {
            "evaluation_report": lane["evaluation_report"],
            "runtime_manifest": lane["runtime_manifest"],
            "checkpoint_refs": lane["checkpoint_refs"],
            "evidence_note": lane["evidence_note"],
        }.items():
            artifact = summary_lane["artifacts"][key]
            assert artifact["path"] == expected_path
            assert artifact["sha256"] == _sha256_file(REPO_ROOT / expected_path)

        if lane["edit_family"] in {"lora_merge", "fine_tune"}:
            assert (
                report["edit"]["edit_provenance"]["edit_family"] == lane["edit_family"]
            )

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.OK
        verification = result.payload["results"][0]["verification"]
        assert verification["runtime_provenance"]["status"] == "verified"


def test_lora_byoe_metadata_builds_and_verifies_signed_evidence_pack(
    tmp_path: Path,
) -> None:
    example_dir = REPO_ROOT / "public_evidence" / "byoe_examples" / "lora_merge_byoe"
    report_path = example_dir / "evaluation.report.json"
    refs_path = example_dir / "checkpoint_refs.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_provenance = report["edit"]["edit_provenance"]
    expected_impact = report["edit"]["edit_impact"]

    final_verdict = tmp_path / "final_verdict.json"
    signing_key = tmp_path / "evidence-pack-signing-key.pem"
    public_key = tmp_path / "evidence-pack-signing-key.pub.pem"
    pack_dir = tmp_path / "lora_byoe_evidence_pack"
    _write_json_file(
        final_verdict,
        {
            "verdict": "PASS",
            "scope": "lora_merge_byoe_optional_edit_metadata_fixture",
        },
    )
    fingerprint = _generate_signing_keypair(
        signing_key,
        public_key_path=public_key,
    )

    build_result = build_evidence_pack(
        pack_dir,
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        material_specs=[("checkpoint_refs", refs_path)],
        signing_key_path=signing_key,
        profile="release",
        report_assurance="strict",
        release_review=True,
    )

    assert build_result.status == EvidencePackStatus.OK
    assert build_result.payload["ok"] is True
    assert build_result.payload["signature"]["present"] is True
    assert build_result.payload["signature"]["signer_fingerprint"] == fingerprint
    assert build_result.payload["verify"]["summary"]["ok"] is True
    assert build_result.payload["verify"]["results"][0]["ok"] is True

    copied_reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    assert len(copied_reports) == 1
    copied_report = json.loads(copied_reports[0].read_text(encoding="utf-8"))
    assert validate_report(copied_report) is True
    assert copied_report["edit"]["edit_provenance"] == expected_provenance
    assert copied_report["edit"]["edit_impact"] == expected_impact

    copied_refs = json.loads(
        (pack_dir / "metadata" / "checkpoint_refs.json").read_text(encoding="utf-8")
    )
    assert copied_refs["subject_checkpoint"]["edit_provenance"] == expected_provenance
    assert copied_refs["subject_checkpoint"]["edit_impact"] == expected_impact

    verify_result = verify_evidence_pack(
        pack_dir,
        strict=True,
        expected_fingerprint=fingerprint,
        profile="release",
        report_assurance="strict",
    )

    assert verify_result.status == EvidencePackStatus.OK
    assert verify_result.payload["ok"] is True
    assert verify_result.payload["authenticity"] == "pinned"
    assert verify_result.payload["verify"]["summary"]["ok"] is True
    assert verify_result.payload["verify"]["results"][0]["ok"] is True
    verification = verify_result.payload["verify"]["results"][0]["verification"]
    assert verification["runtime_provenance"]["status"] == "verified"
