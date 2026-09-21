"""Authenticate the retained judge campaign and reject altered reference data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "examples/integrations/evaluator-live/references/mistral-7b-sentinel"
SPEC = importlib.util.spec_from_file_location(
    "sentinel_judge_replay", DIRECTORY / "judge_replay.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    root = tmp_path_factory.mktemp("judge-reference")
    physical = root / "physical"
    physical.mkdir()
    alias = root / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    output = alias / "replay"
    report = MODULE.replay(output, require_installed=False)
    return output.resolve(strict=True), report


def test_complete_original_judge_receipts_replay(replayed):
    output, report = replayed
    assert report["original_receipts_replayed"] == 52
    assert report["counts"]["attempt_status"] == {
        "completed": 992,
        "timeout_ambiguous": 32,
    }
    assert report["counts"]["trials"] == 1024
    assert report["counts"]["decisions"] == {"insufficient_evidence": 52}
    for pack in report["results"]:
        result = pack["verification"]
        assert result["authenticated"] and result["verified"] and result["replayed"]
        assert not result["accepted"]
        assert result["decision"] == "insufficient_evidence"
    reference = json.loads((output / "reference.json").read_bytes())
    corrections = [pack for pack in reference["packs"] if pack["correction_of"]]
    assert len(corrections) == 2
    for corrected in corrections:
        original = next(
            pack
            for pack in reference["packs"]
            if pack["id"] == corrected["correction_of"]
        )
        assert original["attempt_status"] == {"timeout_ambiguous": 16}
        assert corrected["attempt_status"] == {"completed": 16}


def test_changed_measurement_rejects_original_receipt(replayed, tmp_path):
    import shutil

    from invarlock.judge_measurements.acceptance import (
        replay_signed_judge_verification_receipt,
    )

    output, _ = replayed
    reference = json.loads((output / "reference.json").read_bytes())
    pack = reference["packs"][0]
    entry = output / pack["directory"]
    evidence = tmp_path / "changed-evidence"
    shutil.copytree(entry / "evidence", evidence)
    path = evidence / "measurements.json"
    original = path.read_bytes()
    changed = json.loads(original)
    changed["sources"][0]["content"] += "\n"
    path.write_text(json.dumps(changed))
    assert path.read_bytes() != original
    result = replay_signed_judge_verification_receipt(
        entry / "recipient/verification.receipt.json",
        evidence_path=evidence,
        recipient_policy_path=entry / "recipient/trust.json",
        expected_verifier_identity=pack["verifier"]["identity"],
        expected_verifier_fingerprint=pack["verifier"]["signing_key_fingerprint"],
    )
    assert not result.verified and not result.accepted
    assert result.errors


def test_source_checkout_cannot_claim_installed_recipient(tmp_path):
    with pytest.raises(ValueError, match="independently installed"):
        MODULE.replay(tmp_path / "unused")


def test_archive_replay_does_not_overwrite_existing_output(replayed):
    output, _ = replayed
    with pytest.raises(FileExistsError):
        MODULE.replay(output, require_installed=False)


def test_archive_inventory_excludes_secrets_and_execution_authorizations(replayed):
    output, _ = replayed
    manifest = json.loads((output / "archive-manifest.json").read_bytes())["files"]
    for name in manifest:
        path = Path(name)
        assert path.suffix not in {".pem", ".key", ".pyc"}
        assert not {".cache", "__pycache__", ".git"}.intersection(path.parts)
        assert not any(
            term in path.name
            for term in ("admission", "authorization", "proposal", "runner.log")
        )
        assert b"PRIVATE KEY-----" not in (output / name).read_bytes()


def test_cli_installs_offline_guard_and_prints_summary(monkeypatch, tmp_path, capsys):
    import invarlock.security

    guards = []
    monkeypatch.setattr(invarlock.security, "enforce_network_policy", guards.append)
    monkeypatch.setattr(
        MODULE, "replay", lambda output: {"output": str(output), "results": ["large"]}
    )
    MODULE.main(["--output", str(tmp_path / "out")])
    assert guards == [False]
    assert json.loads(capsys.readouterr().out) == {"output": str(tmp_path / "out")}
