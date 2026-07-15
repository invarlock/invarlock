from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from invarlock.evidence_pack_scenario_contract import (
    ProofHandler,
    parse_scenario_contract,
)
from invarlock.evidence_pack_training_validation import (
    _require_training_evidence_proof,
    _training_canonical_digest,
    _training_profile_digest,
    _training_profile_snapshot_errors,
)
from invarlock.training_evidence_contracts.common import canonical_json_sha256
from scripts.evidence_packs.python.editing.training_receipt import with_receipt_digest
from tests.evidence_packs._support_training_evidence_proof import _proof_for
from tests.evidence_packs._support_training_pack import _build_training_pack


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _values(
    tmp_path: Path,
    *,
    profile_id: str = "tiny_gpt2_lora_v1",
    edit_spec: str = "lora_merge:2:4:attn",
    scope: str = "attn",
):
    pack, report_dir = _build_training_pack(
        tmp_path,
        profile_id=profile_id,
        edit_spec=edit_spec,
        scope=scope,
    )
    scenario = json.loads(
        (pack / "metadata/scenarios.json").read_text(encoding="utf-8")
    )["scenarios"][0]
    contract = parse_scenario_contract(scenario)
    receipt = json.loads(
        (report_dir / "training_receipt.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (report_dir / "evaluation.report.json").read_text(encoding="utf-8")
    )
    return pack, report_dir, contract, receipt, report


def test_training_digest_helpers_reject_non_json_and_non_profile() -> None:
    assert _training_canonical_digest({"bad": {1, 2}}) is None
    assert _training_profile_digest([]) is None
    profile = {"profile_sha256": "ignored", "steps": 2}
    assert _training_profile_digest(profile) == _training_canonical_digest({"steps": 2})


def test_training_snapshot_binding_and_file_failures(tmp_path: Path) -> None:
    pack, _, contract, receipt, _ = _values(tmp_path)
    assert (
        _training_profile_snapshot_errors(
            pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
        )
        == []
    )
    assert _training_profile_snapshot_errors(
        pack_dir=pack,
        scenario_id="training",
        contract=replace(contract, training_profile=None),
        receipt=receipt,
    ) == ["training: training scenario has no profile snapshot binding"]
    assert _training_profile_snapshot_errors(
        pack_dir=pack,
        scenario_id="training",
        contract=replace(contract, edit=None),
        receipt=receipt,
    ) == ["training: training scenario has no typed parameter scope"]
    snapshot_path = pack / contract.training_profile.snapshot_path
    snapshot_path.unlink()
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("snapshot is unavailable" in error for error in errors)


def test_training_snapshot_shape_and_binding_matrix(tmp_path: Path) -> None:
    pack, _, contract, receipt, _ = _values(tmp_path)
    snapshot_path = pack / contract.training_profile.snapshot_path
    original = json.loads(snapshot_path.read_text(encoding="utf-8"))

    bad = {**original, "extra": True}
    _write_json(snapshot_path, bad)
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("missing or unsupported fields" in error for error in errors)

    bad = {**original, "profile": []}
    _write_json(snapshot_path, bad)
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("profile must be an object" in error for error in errors)

    bad = deepcopy(original)
    bad["profile"]["extra"] = True
    _write_json(snapshot_path, bad)
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("profile shape is invalid" in error for error in errors)

    bad = deepcopy(original)
    bad.update(
        schema="wrong",
        profile_id="wrong",
        profile_sha256="sha256:" + "0" * 64,
        scope="all",
    )
    bad["profile"]["profile_sha256"] = "sha256:" + "0" * 64
    bad["profile"]["edit_type"] = "fine_tune"
    _write_json(snapshot_path, bad)
    bad_receipt = deepcopy(receipt)
    bad_receipt.update(
        profile_id="wrong", profile_sha256="sha256:" + "0" * 64, edit_type="fine_tune"
    )
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=bad_receipt
    )
    for fragment in (
        "snapshot digest mismatch",
        "snapshot schema mismatch",
        "snapshot profile_id mismatch",
        "snapshot profile_sha256 mismatch",
        "snapshot scope mismatch",
        "profile digest field mismatch",
        "profile digest does not bind content",
        "profile edit_type mismatch",
        "receipt profile_id mismatch",
        "receipt profile_sha256 mismatch",
        "receipt edit_type mismatch",
    ):
        assert any(fragment in error for error in errors), fragment


def test_training_profile_and_receipt_block_matrix(tmp_path: Path) -> None:
    pack, _, contract, receipt, _ = _values(tmp_path)
    snapshot_path = pack / contract.training_profile.snapshot_path
    original = json.loads(snapshot_path.read_text(encoding="utf-8"))

    bad = deepcopy(original)
    bad["profile"]["optimizer"] = None
    bad["profile"]["model_load"] = None
    _write_json(snapshot_path, bad)
    bad_receipt = deepcopy(receipt)
    for field in ("training", "optimizer", "model", "training_data"):
        bad_receipt[field] = None
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=bad_receipt
    )
    for fragment in (
        "profile optimizer must be an object",
        "receipt training block must be an object",
        "receipt optimizer block must be an object",
        "receipt model block must be an object",
        "receipt training_data block must be an object",
        "profile model_load must be an object",
    ):
        assert any(fragment in error for error in errors), fragment

    for model_load, fragment in (
        ({"extra": True}, "model_load shape is invalid"),
        (
            {"loss_function": "wrong", "expected_unexpected_keys": [" x", "x"]},
            "model_load loss_function is invalid",
        ),
    ):
        bad = deepcopy(original)
        bad["profile"]["model_load"] = model_load
        _write_json(snapshot_path, bad)
        errors = _training_profile_snapshot_errors(
            pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
        )
        assert any(fragment in error for error in errors)
        if "expected_unexpected_keys" in model_load:
            assert any(
                "expected_unexpected_keys is invalid" in error for error in errors
            )


def test_training_baseline_load_and_cross_binding_matrix(tmp_path: Path) -> None:
    pack, _, contract, receipt, _ = _values(tmp_path)
    cases = (
        (None, "baseline_load must be an object"),
        ({"extra": True}, "baseline_load shape is invalid"),
        (
            {
                "loss_function": "ForCausalLM",
                "diagnostics": None,
                "diagnostics_sha256": "bad",
            },
            "baseline_load diagnostics must be an object",
        ),
        (
            {
                "loss_function": "ForCausalLM",
                "diagnostics": {},
                "diagnostics_sha256": "bad",
            },
            "baseline_load diagnostics shape is invalid",
        ),
    )
    for baseline_load, fragment in cases:
        bad = deepcopy(receipt)
        bad["model"]["baseline_load"] = baseline_load
        errors = _training_profile_snapshot_errors(
            pack_dir=pack, scenario_id="training", contract=contract, receipt=bad
        )
        assert any(fragment in error for error in errors), (fragment, errors)

    bad = deepcopy(receipt)
    diagnostics = bad["model"]["baseline_load"]["diagnostics"]
    diagnostics.update(schema="wrong", policy="wrong", missing_keys=["x"])
    bad["model"]["baseline_load"]["diagnostics_sha256"] = "bad"
    bad["model"]["model_id"] = "wrong"
    bad["training_data"]["path"] = "wrong"
    bad["training"]["completed_steps"] = 0
    bad["optimizer"]["name"] = "wrong"
    bad["optimizer"]["learning_rate"] = 99.0
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=bad
    )
    for fragment in (
        "diagnostics schema is invalid",
        "diagnostics policy is invalid",
        "diagnostics missing_keys must be empty",
        "diagnostics do not bind digest",
        "model_id does not bind receipt",
        "training_data.path does not bind receipt",
        "steps does not bind receipt",
        "optimizer.name does not bind receipt",
        "optimizer.learning_rate does not bind receipt",
    ):
        assert any(fragment in error for error in errors), fragment


def test_training_lora_and_full_ft_parameter_matrix(tmp_path: Path) -> None:
    pack, _, contract, receipt, _ = _values(tmp_path)
    snapshot_path = pack / contract.training_profile.snapshot_path
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    bad = deepcopy(snapshot)
    bad["profile"]["lora"] = None
    _write_json(snapshot_path, bad)
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("LoRA block must be an object" in error for error in errors)

    bad = deepcopy(snapshot)
    bad["profile"]["lora"].update(rank=99, alpha=99.0, extra=True)
    _write_json(snapshot_path, bad)
    bad_receipt = deepcopy(receipt)
    bad_receipt["lora"] = None
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=bad_receipt
    )
    for fragment in (
        "LoRA block shape is invalid",
        "LoRA rank mismatch",
        "LoRA alpha mismatch",
        "LoRA configuration does not bind receipt",
    ):
        assert any(fragment in error for error in errors), fragment

    pack, _, contract, receipt, _ = _values(
        tmp_path / "full",
        profile_id="tiny_gpt2_full_ft_v1",
        edit_spec="fine_tune:0.00001:2:all",
        scope="all",
    )
    snapshot_path = pack / contract.training_profile.snapshot_path
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["profile"]["optimizer"]["learning_rate"] = 1.0
    snapshot["profile"]["steps"] = 99
    _write_json(snapshot_path, snapshot)
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("learning_rate mismatch" in error for error in errors)
    assert any("steps mismatch" in error for error in errors)

    snapshot["profile"]["optimizer"] = None
    _write_json(snapshot_path, snapshot)
    errors = _training_profile_snapshot_errors(
        pack_dir=pack, scenario_id="training", contract=contract, receipt=receipt
    )
    assert any("optimizer must be an object" in error for error in errors)


def test_require_training_proof_sidecar_and_identity_matrix(tmp_path: Path) -> None:
    pack, report_dir, contract, _, report = _values(tmp_path)
    assert (
        _require_training_evidence_proof(
            pack_dir=pack,
            scenario_id="training",
            contract=contract,
            report_dir=report_dir,
            report=report,
        )
        == []
    )
    assert _require_training_evidence_proof(
        pack_dir=pack,
        scenario_id="training",
        contract=replace(contract, proof_handler=ProofHandler.ERROR_INJECTION),
        report_dir=report_dir,
        report=report,
    ) == ["training: internal training proof dispatch mismatch"]
    assert _require_training_evidence_proof(
        pack_dir=pack,
        scenario_id="training",
        contract=replace(contract, edit=None),
        report_dir=report_dir,
        report=report,
    ) == ["training: training scenario has no typed edit contract"]

    receipt_path = report_dir / "training_receipt.json"
    proof_path = report_dir / "training_evidence_proof.json"
    receipt_path.unlink()
    proof_path.unlink()
    errors = _require_training_evidence_proof(
        pack_dir=pack,
        scenario_id="training",
        contract=contract,
        report_dir=report_dir,
        report=report,
    )
    assert errors == [
        "training: training receipt sidecar missing",
        "training: training evidence proof sidecar missing",
    ]

    pack, report_dir, contract, _, report = _values(tmp_path / "invalid")
    (report_dir / "training_receipt.json").write_text("[]", encoding="utf-8")
    (report_dir / "training_evidence_proof.json").write_text("[]", encoding="utf-8")
    errors = _require_training_evidence_proof(
        pack_dir=pack,
        scenario_id="training",
        contract=contract,
        report_dir=report_dir,
        report=report,
    )
    assert errors == [
        "training: training receipt sidecar is invalid",
        "training: training evidence proof sidecar is invalid",
    ]

    pack, report_dir, contract, _, _ = _values(tmp_path / "identity")
    errors = _require_training_evidence_proof(
        pack_dir=pack,
        scenario_id="training",
        contract=contract,
        report_dir=report_dir,
        report={},
    )
    assert errors == [
        "training: evaluation subject identity missing",
        "training: evaluation baseline identity missing",
    ]


def test_training_proof_rejects_rehashed_forged_dataset_provider(
    tmp_path: Path,
) -> None:
    """A receipt and proof cannot replace the sealed provider-policy identity."""

    pack, report_dir, contract, receipt, report = _values(tmp_path)
    forged_provider = {"kind": "forged-provider", "revision": "f" * 40}
    receipt["dataset_provider"] = {
        "provider": forged_provider,
        "provider_sha256": canonical_json_sha256(forged_provider),
    }
    forged_receipt = with_receipt_digest(receipt)
    forged_proof, _, _ = _proof_for(forged_receipt)
    _write_json(report_dir / "training_receipt.json", forged_receipt)
    _write_json(report_dir / "training_evidence_proof.json", forged_proof)

    errors = _require_training_evidence_proof(
        pack_dir=pack,
        scenario_id="training",
        contract=contract,
        report_dir=report_dir,
        report=report,
    )

    assert (
        "training: training receipt dataset provider does not bind sealed provider "
        "policy" in errors
    )
