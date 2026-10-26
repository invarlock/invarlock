"""Full native captured transactions against the existing Python verifier."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from invarlock.captured_verification import (
    CapturedVerificationError,
    CapturedVerificationIncomplete,
    verify_captured_evidence,
    verify_captured_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_binary():
    binary = os.environ.get("INVARLOCK_NATIVE_VERIFY")
    assert binary and Path(binary).is_file(), "build and set INVARLOCK_NATIVE_VERIFY"
    return binary


@pytest.fixture(scope="module")
def fixture_roots(tmp_path_factory):
    import sys

    sys.path.insert(0, str(ROOT / "experiments/rust/python"))
    from invarlock_rust_experiment.fixtures import make_fixture

    root = tmp_path_factory.mktemp("captured-native").resolve()
    result = {}
    for scenario in ("pass", "regression"):
        path = root / scenario
        make_fixture(path, 12, scenario)
        result[scenario] = path
    return result


def execute(binary, command, request):
    completed = subprocess.run(
        [binary, command, "--request", "-"],
        input=json.dumps(request, ensure_ascii=False).encode(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.stdout, completed.stderr.decode()
    return completed.returncode, json.loads(completed.stdout)


def config_for(root, receipt):
    pins = json.loads((root / "trust-pins.json").read_bytes())
    return {
        "evidence_path": str(root / "pack"),
        "policy_path": str(root / "policy.json"),
        **{
            key: value
            for key, value in pins.items()
            if key
            in {
                "expected_baseline_run",
                "expected_subject_run",
                "expected_request_digest",
                "expected_signer",
            }
        },
        "receipt_path": str(receipt),
        "verifier_signing_key_path": str(root / "verifier.pem"),
        "verifier_identity": pins["expected_verifier_identity"],
    }


@pytest.mark.parametrize("scenario", ["pass", "regression"])
def test_full_replay_and_signed_receipt(
    native_binary, fixture_roots, tmp_path, scenario
):
    root = fixture_roots[scenario]
    native = tmp_path.resolve() / "native"
    reference = tmp_path.resolve() / "reference"
    native.mkdir()
    reference.mkdir()
    request = config_for(root, native / "receipt.json")
    status, actual = execute(native_binary, "verify-captured", request)
    expected_request = {**request, "receipt_path": reference / "receipt.json"}
    expected = verify_captured_evidence(**expected_request)
    assert actual == json.loads(json.dumps(expected))
    assert status == (0 if scenario == "pass" else 7)
    assert (native / "receipt.json").read_bytes() == (
        reference / "receipt.json"
    ).read_bytes()
    pins = json.loads((root / "trust-pins.json").read_bytes())
    verified = verify_captured_receipt(
        native / "receipt.json", root / "pack", policy_path=root / "policy.json", **pins
    )
    assert verified.ok


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_signer",
        "local_budget",
        "report_bytes",
        "extra_file",
        "missing_file",
        "noncanonical_manifest",
        "invalid_signature",
        "unsafe_payload",
        "existing_receipt",
        "receipt_inside",
    ],
)
def test_failure_and_receipt_parity(native_binary, fixture_roots, tmp_path, mutation):
    root = tmp_path.resolve() / "fixture"
    shutil.copytree(fixture_roots["pass"], root)
    for path in (root / "pack").rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    native = tmp_path.resolve() / "native"
    reference = tmp_path.resolve() / "reference"
    native.mkdir()
    reference.mkdir()
    request = config_for(root, native / "receipt.json")
    if mutation == "wrong_signer":
        request["expected_signer"] = "sha256:" + "0" * 64
    elif mutation == "local_budget":
        request["max_bootstrap_draws"] = 0
    elif mutation == "report_bytes":
        (root / "pack/reports/evaluation.report.json").write_bytes(b"{}\n")
    elif mutation == "extra_file":
        (root / "pack/unexpected").write_text("unexpected")
    elif mutation == "missing_file":
        (root / "pack/records/subject.json").unlink()
    elif mutation == "noncanonical_manifest":
        path = root / "pack/manifest.json"
        path.write_bytes(b" " + path.read_bytes())
    elif mutation == "invalid_signature":
        path = root / "pack/manifest.signature.json"
        signature = json.loads(path.read_bytes())
        signature["signature"]["value"] = "A" * 88
        path.write_text(
            json.dumps(signature, sort_keys=True, separators=(",", ":")) + "\n"
        )
    elif mutation == "unsafe_payload":
        path = root / "pack/records/subject.json"
        path.unlink()
        path.symlink_to(root / "subject.json")
    elif mutation == "existing_receipt":
        (native / "receipt.json").write_bytes(b"retained")
        (reference / "receipt.json").write_bytes(b"retained")
    elif mutation == "receipt_inside":
        request["receipt_path"] = str(root / "pack/receipt.json")
    reference_request = {**request, "receipt_path": str(reference / "receipt.json")}
    if mutation == "receipt_inside":
        reference_request["receipt_path"] = request["receipt_path"]
    with pytest.raises(
        (CapturedVerificationError, CapturedVerificationIncomplete)
    ) as failure:
        verify_captured_evidence(**reference_request)
    status, actual = execute(native_binary, "verify-captured", request)
    assert status == failure.value.exit_code
    assert actual["error"] == str(failure.value)
    native_receipt = Path(request["receipt_path"])
    reference_receipt = Path(reference_request["receipt_path"])
    assert native_receipt.exists() == reference_receipt.exists()
    if reference_receipt.exists():
        assert native_receipt.read_bytes() == reference_receipt.read_bytes()


def test_native_publication_and_receipt_authentication(
    native_binary, fixture_roots, tmp_path
):
    root = fixture_roots["pass"]
    target = tmp_path.resolve() / "native-pack"
    request = {
        "baseline": json.loads((root / "baseline.json").read_bytes()),
        "subject": json.loads((root / "subject.json").read_bytes()),
        "policy": json.loads((root / "policy.json").read_bytes()),
        "request": json.loads((root / "request.json").read_bytes()),
        "destination": str(target),
        "signing_key_path": str(root / "signer.pem"),
    }
    status, result = execute(native_binary, "publish-captured", request)
    assert status == 0, result
    for path in (root / "pack").rglob("*"):
        if path.is_file():
            assert (
                target / path.relative_to(root / "pack")
            ).read_bytes() == path.read_bytes()
    config = config_for(root, tmp_path.resolve() / "new" / "parents" / "receipt.json")
    config["evidence_path"] = str(target)
    status, result = execute(native_binary, "verify-captured", config)
    assert status == 0, result
    pins = json.loads((root / "trust-pins.json").read_bytes())
    for signer in (pins["expected_verifier_fingerprint"], "sha256:" + "0" * 64):
        args = {**pins, "expected_verifier_fingerprint": signer}
        reference = verify_captured_receipt(
            Path(config["receipt_path"]),
            target,
            policy_path=root / "policy.json",
            **args,
        )
        status, received = execute(
            native_binary,
            "verify-captured-receipt",
            {
                "receipt_path": config["receipt_path"],
                "evidence_path": str(target),
                "policy_path": str(root / "policy.json"),
                **args,
            },
        )
        assert status == (0 if reference.ok else 4), received
        assert received == {
            "ok": reference.ok,
            "signed": reference.signed,
            "statement": reference.statement,
            "verifier_fingerprint": reference.verifier_fingerprint,
            "errors": list(reference.errors),
        }
