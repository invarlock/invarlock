from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.evidence_pack_edit_verifier import _verify_edit_metadata_consistency
from invarlock.training_evidence import with_training_evidence_proof_digest
from invarlock.training_model_load import load_diagnostics_sha256
from scripts.evidence_packs.python.editing.training_receipt import with_receipt_digest
from tests.evidence_packs._support_training_evidence_proof import _proof_for
from tests.evidence_packs._support_training_pack import (
    _build_training_pack,
    _canonical_sha256,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = REPO_ROOT / "scripts/evidence_packs/training_profiles.json"


@pytest.mark.parametrize(
    ("profile_id", "edit_spec", "scope"),
    [
        ("tiny_gpt2_lora_v1", "lora_merge:2:4:attn", "attn"),
        ("tiny_gpt2_full_ft_v1", "fine_tune:0.00001:2:all", "all"),
    ],
)
def test_package_verifier_binds_real_training_parameters_to_profile_and_receipt(
    tmp_path: Path,
    profile_id: str,
    edit_spec: str,
    scope: str,
) -> None:
    pack_dir, _ = _build_training_pack(
        tmp_path,
        profile_id=profile_id,
        edit_spec=edit_spec,
        scope=scope,
    )

    assert _verify_edit_metadata_consistency(pack_dir) == []


@pytest.mark.parametrize(
    ("profile_id", "edit_spec", "scope"),
    [
        ("tiny_gpt2_lora_v1", "lora_merge:2:4:attn", "attn"),
        ("tiny_gpt2_full_ft_v1", "fine_tune:0.00001:2:all", "all"),
    ],
)
def test_generic_package_verifier_rejects_training_coverage_forgery(
    tmp_path: Path,
    profile_id: str,
    edit_spec: str,
    scope: str,
) -> None:
    pack_dir, report_dir = _build_training_pack(
        tmp_path,
        profile_id=profile_id,
        edit_spec=edit_spec,
        scope=scope,
    )
    metadata_path = report_dir / "edit_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["coverage"] = {
        "edited_tensors": 0,
        "edited_params": 0,
        "total_params": 0,
        "coverage_ratio": 0.0,
    }
    _write_json(metadata_path, metadata)

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert any("must be positive for a proof-routed model edit" in e for e in errors)
    assert any(
        "training edit coverage.edited_params does not bind receipt" in e
        for e in errors
    )


def test_generic_package_verifier_rejects_training_provider_forgery(
    tmp_path: Path,
) -> None:
    pack_dir, report_dir = _build_training_pack(
        tmp_path,
        profile_id="tiny_gpt2_full_ft_v1",
        edit_spec="fine_tune:0.00001:2:all",
        scope="all",
    )
    report_path = report_dir / "evaluation.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dataset"] = {
        "provider": "hf_text",
        "dataset_name": "forged/dataset",
        "revision": "a" * 40,
    }
    _write_json(report_path, report)

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert any("dataset provider does not bind report" in error for error in errors)


def test_package_verifier_requires_an_evaluation_report_for_training_scenario(
    tmp_path: Path,
) -> None:
    pack_dir, report_dir = _build_training_pack(
        tmp_path,
        profile_id="tiny_gpt2_lora_v1",
        edit_spec="lora_merge:2:4:attn",
        scope="attn",
    )
    (report_dir / "evaluation.report.json").unlink()

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert errors == [
        "training_subject: active training scenario has no evaluation report"
    ]


@pytest.mark.parametrize(
    ("retired_surface", "expected_fragment"),
    [
        ("schema", "training evidence proof has an unrecognized schema"),
        ("history_claim", "provenance has unbound"),
    ],
)
def test_package_verifier_rejects_retired_training_proof_semantics(
    tmp_path: Path,
    retired_surface: str,
    expected_fragment: str,
) -> None:
    pack_dir, report_dir = _build_training_pack(
        tmp_path,
        profile_id="tiny_gpt2_full_ft_v1",
        edit_spec="fine_tune:0.00001:2:all",
        scope="all",
    )
    proof_path = report_dir / "training_evidence_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if retired_surface == "schema":
        proof["schema"] = "invarlock/training-evidence-proof-v2"
    else:
        proof["provenance"] = {
            "kind": "real_optimized_training",
            "training_backend": "full_parameter_optimizer_training",
            "synthetic": False,
        }
    _write_json(proof_path, with_training_evidence_proof_digest(proof))

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert any(expected_fragment in error for error in errors)


def test_package_verifier_rejects_lora_scenario_parameter_profile_mismatch(
    tmp_path: Path,
) -> None:
    pack_dir, _ = _build_training_pack(
        tmp_path,
        profile_id="tiny_gpt2_lora_v1",
        edit_spec="lora_merge:2:4:attn",
        scope="attn",
    )
    scenarios_path = pack_dir / "metadata/scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios["scenarios"][0]["generation"]["edit_spec"] = "lora_merge:4:4:attn"
    _write_json(scenarios_path, scenarios)

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert any("training profile LoRA rank mismatch" in error for error in errors)


def test_package_verifier_rejects_fully_rehashed_load_policy_receipt_tamper(
    tmp_path: Path,
) -> None:
    pack_dir, report_dir = _build_training_pack(
        tmp_path,
        profile_id="tiny_gpt2_full_ft_v1",
        edit_spec="fine_tune:0.00001:2:all",
        scope="all",
    )
    receipt_path = report_dir / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    baseline_load = receipt["model"]["baseline_load"]
    baseline_load["diagnostics"]["unexpected_keys"] = ["injected.weight"]
    baseline_load["diagnostics_sha256"] = load_diagnostics_sha256(
        baseline_load["diagnostics"]
    )
    receipt = with_receipt_digest(receipt)
    proof, _, _ = _proof_for(receipt)
    _write_json(receipt_path, receipt)
    _write_json(report_dir / "training_evidence_proof.json", proof)

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert any(
        "model_load expected_unexpected_keys does not bind receipt" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        (
            lambda model_load: model_load.update(extra=True),
            "training profile model_load shape is invalid",
        ),
        (
            lambda model_load: model_load.update(loss_function="fallback"),
            "training profile model_load loss_function is invalid",
        ),
        (
            lambda model_load: model_load.update(
                expected_unexpected_keys=["z.weight", "a.weight"]
            ),
            "training profile model_load expected_unexpected_keys is invalid",
        ),
    ],
)
def test_package_verifier_rejects_rehashed_invalid_profile_load_policy(
    tmp_path: Path,
    mutate,
    expected_fragment: str,
) -> None:
    pack_dir, _ = _build_training_pack(
        tmp_path,
        profile_id="tiny_gpt2_full_ft_v1",
        edit_spec="fine_tune:0.00001:2:all",
        scope="all",
    )
    snapshot_path = pack_dir / "metadata/training_profiles/tiny_gpt2_full_ft_v1.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    profile = snapshot["profile"]
    mutate(profile["model_load"])
    profile_without_digest = dict(profile)
    profile_without_digest.pop("profile_sha256")
    profile["profile_sha256"] = _canonical_sha256(profile_without_digest)
    snapshot["profile_sha256"] = profile["profile_sha256"]
    _write_json(snapshot_path, snapshot)

    scenarios_path = pack_dir / "metadata/scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    binding = scenarios["scenarios"][0]["training_profile"]
    binding["profile_sha256"] = profile["profile_sha256"]
    binding["snapshot_sha256"] = (
        "sha256:" + hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    )
    _write_json(scenarios_path, scenarios)

    errors = _verify_edit_metadata_consistency(pack_dir)

    assert any(expected_fragment in error for error in errors)
