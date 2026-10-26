"""Fresh evidence-set conjunction and embedded receipt parity against Python."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from invarlock.captured_contracts import sha
from invarlock.evidence_sets.verification import (
    verify_evidence_set,
    verify_stored_evidence_set_result,
)
from tests.evidence_sets.test_verification import fixture, write

ROOT = Path(__file__).resolve().parents[2]


def native(config, operation="verify-evidence-set"):
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "native/target"))
    binary = os.environ.get(
        "INVARLOCK_NATIVE_VERIFY", str(target / "debug/invarlock-native-verify")
    )
    process = subprocess.run(
        [binary, operation, "--request", "-"],
        input=json.dumps(config),
        text=True,
        capture_output=True,
    )
    return process.returncode, json.loads(process.stdout)


@pytest.mark.parametrize(
    "mode",
    [
        "pass",
        "advisory",
        "incomplete",
        "minimum",
        "pin-index",
        "pin-cases",
        "pin-artifact",
        "pin-profile",
        "case-input",
        "case-expected",
        "case-output",
        "case-metadata",
        "unsigned",
        "profile-symlink",
        "profile-malformed",
        "statement",
        "nested",
        "budget-bool",
        "budget-negative",
        "budget-float",
        "budget-zero",
    ],
)
def test_set_transaction(mode, tmp_path):
    directory = tmp_path.resolve()
    options = (
        {"role": "advisory"}
        if mode == "advisory"
        else {"incomplete": True}
        if mode == "incomplete"
        else {"captured_minimum": 20}
        if mode == "minimum"
        else {}
    )
    if mode.startswith("case-"):
        field = mode[5:]

        def change(baseline, subject):
            for run in (baseline, subject):
                run["records"][0][field] = (
                    {"changed": "yes"} if field == "metadata" else "changed"
                )

        options["captured_change"] = change
    root, policy = fixture(directory, **options)
    if mode.startswith("pin-"):
        paths = {
            "pin-index": ["index_sha256"],
            "pin-cases": ["shared_inputs", "case_set_sha256"],
            "pin-artifact": ["shared_inputs", "subject_artifact_sha256"],
            "pin-profile": ["members", "judge", "trust_profile_sha256"],
        }
        value = json.loads(policy.read_bytes())
        node = value
        parts = paths[mode]
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = "sha256:" + "f" * 64
        write(policy, value)
    if mode == "unsigned":
        envelope = root / "judge/envelope.json"
        value = json.loads(envelope.read_bytes())
        value.update(signer=None, signature=None)
        write(envelope, value)
        index = json.loads((root / "evidence-set.json").read_bytes())
        index["members"]["judge"]["statement_sha256"] = sha(envelope.read_bytes())
        write(root / "evidence-set.json", index)
        value = json.loads(policy.read_bytes())
        value["index_sha256"] = sha((root / "evidence-set.json").read_bytes())
        write(policy, value)
    if mode == "profile-symlink":
        path = policy.parent / "judge.json"
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.symlink_to(saved)
    if mode == "profile-malformed":
        path = policy.parent / "judge.json"
        write(path, {"format": "invalid"})
        value = json.loads(policy.read_bytes())
        value["members"]["judge"]["trust_profile_sha256"] = sha(path.read_bytes())
        write(policy, value)
    if mode == "statement":
        path = root / "judge/envelope.json"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"\n")
    if mode == "nested":
        (root / "judge").chmod(0o755)
        (root / "judge/evidence-set.json").write_bytes(b"{}")
    maximum = {
        "budget-bool": True,
        "budget-negative": -1,
        "budget-float": 1.5,
        "budget-zero": 0,
    }.get(mode, 102400000)
    expected = verify_evidence_set(
        root,
        recipient_policy=policy,
        receipt=directory / "reference.json",
        max_bootstrap_draws=maximum,
    )
    config = {
        "evidence_path": str(root),
        "recipient_policy_path": str(policy),
        "receipt_path": str(directory / "native.json"),
        "max_bootstrap_draws": maximum,
    }
    status, actual = native(config)
    assert actual == expected.payload
    assert status == expected.exit_code
    assert (directory / "reference.json").read_bytes() == (
        directory / "native.json"
    ).read_bytes()
    if mode == "pass":
        config.pop("receipt_path")
        config["stored_result_path"] = str(directory / "native.json")
        stored = verify_stored_evidence_set_result(
            directory / "native.json", root, recipient_policy=policy
        )
        status, actual = native(config, "verify-stored-evidence-set")
        assert actual == stored.payload
        assert status == stored.exit_code
        altered = json.loads((directory / "native.json").read_bytes())
        altered["members"]["judge"]["decision"] = "regression"
        write(directory / "native.json", altered)
        stored = verify_stored_evidence_set_result(
            directory / "native.json", root, recipient_policy=policy
        )
        status, actual = native(config, "verify-stored-evidence-set")
        assert actual == stored.payload
        assert status == stored.exit_code


@pytest.mark.parametrize("mutation", ["none", "different", "missing", "mixed"])
def test_hosted_set_identity(mutation, tmp_path, monkeypatch):
    from tests.evidence_sets.test_hosted_identity import hosted_runs

    hosted_runs.__wrapped__(monkeypatch)
    root, policy = fixture(tmp_path.resolve())
    if mutation != "none":
        value = json.loads(policy.read_bytes())
        if mutation == "different":
            value["shared_inputs"]["subject_service_identity_sha256"] = (
                "sha256:" + "f" * 64
            )
        elif mutation == "missing":
            value["shared_inputs"].pop("subject_service_identity_sha256")
        else:
            value["shared_inputs"]["subject_artifact_sha256"] = "sha256:" + "e" * 64
        write(policy, value)
    expected = verify_evidence_set(root, recipient_policy=policy)
    status, actual = native(
        {"evidence_path": str(root), "recipient_policy_path": str(policy)}
    )
    assert actual == expected.payload
    assert status == expected.exit_code


def test_index_publication_and_no_clobber(tmp_path):
    root, _ = fixture(tmp_path.resolve())
    target = root / "evidence-set.json"
    expected = target.read_bytes()
    target.unlink()
    config = {
        "evidence_path": str(root),
        "deterministic": "deterministic",
        "judge": "judge",
    }
    status, result = native(config, "write-evidence-set-index")
    assert status == 0, result
    assert target.read_bytes() == expected
    status, _ = native(config, "write-evidence-set-index")
    assert status != 0
    assert target.read_bytes() == expected
