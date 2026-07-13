from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from invarlock.training_evidence import with_training_evidence_proof_digest
from scripts.evidence_packs.python.editing.training_profile_snapshot import (
    produce_training_profile_snapshot,
)
from scripts.evidence_packs.python.editing.training_receipt import with_receipt_digest
from tests.scripts._support_source_matrix_artifact_validator import (
    REPO_ROOT,
    _load_validator,
    _write_training_binding_set,
    _write_training_evidence_set,
)


def test_training_binding_requires_canonical_profile_bound_receipt(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    report_dir = tmp_path / "training-report"
    receipt, binding = _write_training_binding_set(report_dir)

    assert (
        validator._validate_training_binding(
            repo_root=REPO_ROOT,
            target="peft_lora",
            report_dir=report_dir,
            expected_training_profile="tiny_gpt2_lora_v1",
        )
        == []
    )

    (report_dir / "training_receipt.json").write_text("{}\n", encoding="utf-8")
    malformed = validator._validate_training_binding(
        repo_root=REPO_ROOT,
        target="peft_lora",
        report_dir=report_dir,
        expected_training_profile="tiny_gpt2_lora_v1",
    )
    assert any(
        "training receipt/profile contract failed" in issue.message
        for issue in malformed
    )

    receipt_path = report_dir / "training_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    forged_binding = copy.deepcopy(binding)
    forged_binding["training_receipt_file_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    forged_binding["receipt_sha256"] = "sha256:" + ("0" * 64)
    forged_binding["subject_tree_sha256"] = "sha256:" + ("1" * 64)
    (report_dir / "training_binding.json").write_text(
        json.dumps(forged_binding), encoding="utf-8"
    )
    forged = validator._validate_training_binding(
        repo_root=REPO_ROOT,
        target="peft_lora",
        report_dir=report_dir,
        expected_training_profile="tiny_gpt2_lora_v1",
    )
    forged_messages = [issue.message for issue in forged]
    assert (
        "training binding receipt_sha256 does not match canonical receipt"
        in forged_messages
    )
    assert any("subject_tree_sha256" in message for message in forged_messages)

    wrong_profile = validator._validate_training_binding(
        repo_root=REPO_ROOT,
        target="peft_lora",
        report_dir=report_dir,
        expected_training_profile="tiny_gpt2_full_ft_v1",
    )
    assert any(
        "training receipt/profile contract failed" in issue.message
        for issue in wrong_profile
    )

    wrong_schema = copy.deepcopy(receipt)
    wrong_schema["schema"] = "unknown"
    wrong_schema = with_receipt_digest(wrong_schema)
    receipt_path.write_text(json.dumps(wrong_schema), encoding="utf-8")
    schema_failure = validator._validate_training_binding(
        repo_root=REPO_ROOT,
        target="peft_lora",
        report_dir=report_dir,
        expected_training_profile="tiny_gpt2_lora_v1",
    )
    assert any(
        "training receipt/profile contract failed" in issue.message
        for issue in schema_failure
    )


def test_training_binding_rejects_ambiguous_receipt_and_extra_binding_field(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    report_dir = tmp_path / "training-report"
    _, binding = _write_training_binding_set(report_dir)
    receipt_path = report_dir / "training_receipt.json"
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        receipt_text.replace(
            '"schema": "invarlock/evidence-pack-training-receipt-v1"',
            '"schema": "wrong", '
            '"schema": "invarlock/evidence-pack-training-receipt-v1"',
        ),
        encoding="utf-8",
    )

    ambiguous = validator._validate_training_binding(
        repo_root=REPO_ROOT,
        target="peft_lora",
        report_dir=report_dir,
        expected_training_profile="tiny_gpt2_lora_v1",
    )
    assert any("duplicate JSON key 'schema'" in issue.message for issue in ambiguous)

    receipt_path.write_text(receipt_text, encoding="utf-8")
    binding["claimed_green"] = True
    (report_dir / "training_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    extra_field = validator._validate_training_binding(
        repo_root=REPO_ROOT,
        target="peft_lora",
        report_dir=report_dir,
        expected_training_profile="tiny_gpt2_lora_v1",
    )
    assert any(
        "training binding fields must match v1 exactly" in issue.message
        for issue in extra_field
    )


def test_training_evidence_requires_bound_proof_and_profile_snapshot(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    report_dir = tmp_path / "training-evidence-report"
    _, proof = _write_training_evidence_set(report_dir)

    validation_kwargs = {
        "repo_root": REPO_ROOT,
        "target": "peft_lora",
        "report_dir": report_dir,
        "expected_training_profile": "tiny_gpt2_lora_v1",
        "expected_training_scope": "attn",
    }
    assert validator._validate_training_evidence(**validation_kwargs) == []

    binding_path = report_dir / "training_binding.json"
    binding_text = binding_path.read_text(encoding="utf-8")
    binding_path.write_text(
        binding_text.replace(
            '"verified": true',
            '"verified": false, "verified": true',
        ),
        encoding="utf-8",
    )
    ambiguous_binding = validator._validate_training_evidence(**validation_kwargs)
    assert any(
        "training binding is not strict JSON" in issue.message
        for issue in ambiguous_binding
    )
    binding_path.write_text(binding_text, encoding="utf-8")

    proof_path = report_dir / "training_evidence_proof.json"
    proof_path.unlink()
    missing_proof = validator._validate_training_evidence(**validation_kwargs)
    assert any(
        "training evidence proof is missing" in issue.message for issue in missing_proof
    )

    forged_proof = dict(proof)
    forged_proof["artifact_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + ("0" * 64),
    }
    proof_path.write_text(
        json.dumps(with_training_evidence_proof_digest(forged_proof)),
        encoding="utf-8",
    )
    tampered_proof = validator._validate_training_evidence(**validation_kwargs)
    assert any(
        "artifact_identity does not match expected artifact" in issue.message
        for issue in tampered_proof
    )

    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    snapshot_path = report_dir / "training_profile_snapshot.json"
    snapshot_path.unlink()
    missing_snapshot = validator._validate_training_evidence(**validation_kwargs)
    assert any(
        "training profile snapshot is missing" in issue.message
        for issue in missing_snapshot
    )

    produce_training_profile_snapshot(
        profile_id="tiny_gpt2_lora_v1",
        scope="attn",
        output_path=snapshot_path,
        repo_root=REPO_ROOT,
    )
    tampered_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tampered_profile = tampered_snapshot["profile"]
    assert isinstance(tampered_profile, dict)
    tampered_profile["seed"] = 0
    snapshot_path.write_text(json.dumps(tampered_snapshot), encoding="utf-8")
    snapshot_failure = validator._validate_training_evidence(**validation_kwargs)
    assert any(
        "training profile snapshot profile does not match immutable profile"
        in issue.message
        for issue in snapshot_failure
    )


def test_training_source_matrix_requires_complete_evidence_contract(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    entry = {
        "target": "peft_lora",
        "training_profile": "tiny_gpt2_lora_v1",
        "training_scope": "attn",
        "readme": "examples/integrations/peft_lora/README.md",
        "verification_profile": "release",
        "lane": "cuda-container-strict",
        "report_path": "reports/tiny-peft-lora/<artifact-lane>",
        "required_artifacts": [
            "training_receipt.json",
            "training_binding.json",
        ],
        "expected": {},
    }

    issues = validator.validate_entry(
        tmp_path,
        entry,
        acceptance_inputs=None,
    )

    messages = [issue.message for issue in issues]
    assert any(
        "training source matrix is missing required evidence artifacts" in message
        for message in messages
    )


def test_validate_entry_dispatches_complete_training_evidence_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validator = _load_validator()
    report_dir = (
        tmp_path
        / "examples"
        / "integrations"
        / "peft_lora"
        / "reports"
        / "tiny-peft-lora"
        / "cuda-container-strict"
    )
    report_dir.mkdir(parents=True)
    required_artifacts = sorted(validator.TRAINING_EVIDENCE_ARTIFACTS)
    for artifact in required_artifacts:
        (report_dir / artifact).write_text("{}\n", encoding="utf-8")

    dispatched: list[dict[str, object]] = []

    def validate_training_evidence(**kwargs):
        dispatched.append(kwargs)
        return [
            validator.ValidationIssue(
                target="peft_lora",
                path="training_evidence_proof.json",
                message="training evidence sentinel",
            )
        ]

    monkeypatch.setattr(
        validator,
        "_validate_training_binding",
        lambda **_: [],
    )
    monkeypatch.setattr(
        validator,
        "_validate_training_evidence",
        validate_training_evidence,
    )
    entry = {
        "target": "peft_lora",
        "training_profile": "tiny_gpt2_lora_cuda_v1",
        "training_scope": "attn",
        "readme": "examples/integrations/peft_lora/README.md",
        "verification_profile": "release",
        "lane": "cuda-container-strict",
        "report_path": "reports/tiny-peft-lora/<artifact-lane>",
        "required_artifacts": required_artifacts,
        "expected": {},
    }

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert dispatched == [
        {
            "repo_root": tmp_path,
            "target": "peft_lora",
            "report_dir": report_dir,
            "expected_training_profile": "tiny_gpt2_lora_cuda_v1",
            "expected_training_scope": "attn",
        }
    ]
    assert any(issue.message == "training evidence sentinel" for issue in issues)
