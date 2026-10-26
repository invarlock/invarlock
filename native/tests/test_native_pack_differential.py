"""Whole native pack transaction differential against the Python reference.

Build invarlock-native-verify first. Tests compare payloads, process statuses,
and exact Ed25519 receipt bytes, including authenticated rejection receipts.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from invarlock.evidence_pack import (
    EvidenceObservation,
    InputIdentity,
    publish_comparison_evidence,
)
from invarlock.evidence_pack_contract import (
    build_comparison_report,
    canonical_json_bytes,
    sha256_digest,
)
from invarlock.evidence_verification import EvidenceVerificationError, verify_evidence
from tests.evidence_packs.test_evidence_pack import (
    _nll_policy_bytes,
    _publish,
    _rebind_and_resign_pack,
    _side_evidence,
    _verification_anchors,
)

ROOT = Path(__file__).resolve().parents[2]


def run(config):
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "native/target"))
    binary = Path(
        os.environ.get(
            "NATIVE_VERIFIER_BINARY",
            os.environ.get(
                "INVARLOCK_NATIVE_VERIFY", target / "debug/invarlock-native-verify"
            ),
        )
    )
    assert binary.is_file(), f"Build invarlock-native-verify first: {binary}"
    command = (
        "verify-native-receipt" if config.get("op") == "receipt" else "verify-native"
    )
    process = subprocess.run(
        [str(binary), command, "--request", "-"],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=False,
    )
    return {"payload": json.loads(process.stdout), "status": process.returncode}


@pytest.mark.parametrize(
    "mode",
    [
        "exact",
        "fail",
        "nll",
        "v1",
        "v2",
        "observation",
        "paired",
        "report",
        "missing-signature",
        "wrong-artifact",
        "output-forged",
    ],
)
def test_native_pack_transaction(mode, tmp_path):
    directory = tmp_path.resolve()
    directory.mkdir(exist_ok=True)
    obs = (
        (
            EvidenceObservation(
                "note", "comparison", "note", canonical_json_bytes({"note": "hello"})
            ),
        )
        if mode == "observation"
        else ()
    )
    pack, policy, signer, runtimes, key, args = _publish(
        directory, delta_min_pp=-1 if mode == "fail" else -100, observations=obs
    )
    if mode == "nll":
        policy.write_bytes(_nll_policy_bytes(1.3))
        pd = sha256_digest(policy.read_bytes())
        args["policy_bytes"] = policy.read_bytes()
        args["policy"] = InputIdentity(pd, locator="inputs/policy.json")
        args["normalized_request"]["comparison"]["metric"] = (
            "normalized_nll_per_utf8_byte"
        )
        for side, marker, probs in [
            ("baseline", "c", (-2.0, -2.0)),
            ("subject", "d", (-2.5, -2.0)),
        ]:
            args[side + "_evidence"] = _side_evidence(
                directory / ("nll-" + side),
                schedule=args["schedule"],
                image_digest=runtimes[side],
                artifact_marker=marker,
                logprob_sums=probs,
                role=side,
                policy_digest=pd,
            )
        pack = directory / "nll"
        publish_comparison_evidence(pack, **args)
    if mode in ["v1", "v2"]:
        report = pack / "reports/evaluation.report.json"
        report.chmod(0o644)
        report.write_bytes(
            canonical_json_bytes(
                build_comparison_report(
                    comparison_id="single-comparison",
                    paired_records=json.loads(
                        (pack / "records/paired-records.json").read_bytes()
                    ),
                    policy=json.loads(policy.read_bytes()),
                    policy_digest=sha256_digest(policy.read_bytes()),
                    report_format="invarlock/comparison-report-" + mode,
                )
            )
        )
        _rebind_and_resign_pack(pack, key)
    if mode in ["paired", "report", "output-forged"]:
        rel = {
            "paired": "records/paired-records.json",
            "report": "reports/evaluation.report.json",
            "output-forged": "providers/subject/runtime-scoring.observation.json",
        }[mode]
        p = pack / rel
        data = json.loads(p.read_bytes())
        p.chmod(0o644)
        if mode == "paired":
            data["records"][0]["baseline"]["score"] = 0.0
        elif mode == "report":
            data["verdict"] = "fail"
        else:
            data["records"][0]["output_text"] = "FORGED"
        p.write_bytes(canonical_json_bytes(data))
        _rebind_and_resign_pack(pack, key)
    if mode == "missing-signature":
        pack.chmod(0o755)
        (pack / "manifest.signature.json").unlink()
    anchors = _verification_anchors(args)
    config = {
        "evidence_path": str(pack),
        "policy_path": str(policy),
        "expected_baseline_artifact": anchors["expected_artifact_digests"]["baseline"],
        "expected_subject_artifact": anchors["expected_artifact_digests"]["subject"],
        "expected_schedule": anchors["expected_schedule_digest"],
        "expected_baseline_runtime": runtimes["baseline"],
        "expected_subject_runtime": runtimes["subject"],
        "expected_signer": signer,
        "expected_request_digest": anchors["expected_request_digest"],
        "verifier_signing_key_path": str(key),
        "verifier_identity": "fixture-verifier",
    }
    if mode == "wrong-artifact":
        config["expected_subject_artifact"] = "sha256:" + "0" * 64
    for d in ["native", "reference"]:
        (directory / d).mkdir()
    ref_config = {
        k: Path(v)
        if k in ["evidence_path", "policy_path", "verifier_signing_key_path"]
        else v
        for k, v in config.items()
    }
    try:
        ref = verify_evidence(
            **ref_config, receipt_path=directory / "reference/receipt.json"
        )
        expected = ref.payload
        status = 0
    except EvidenceVerificationError as e:
        expected = e.payload
        status = e.exit_code
    config["receipt_path"] = str(directory / "native/receipt.json")
    out = run(config)
    assert out["status"] == status, out
    assert "payload" in out
    actual = out["payload"]
    diff = {
        k: (expected.get(k), actual.get(k))
        for k in set(expected) | set(actual)
        if expected.get(k) != actual.get(k)
    }
    assert not diff, diff
    nr = directory / "native/receipt.json"
    rr = directory / "reference/receipt.json"
    assert nr.read_bytes() == rr.read_bytes()
    rc = {
        **config,
        "op": "receipt",
        "expected_verifier_identity": "fixture-verifier",
        "expected_verifier_fingerprint": signer,
    }
    receipt_result = run(rc)["payload"]
    assert receipt_result["ok"], receipt_result


@pytest.mark.parametrize(
    "relative",
    [
        "request.json",
        "schedule/runtime-behavioral-schedule.json",
        "records/paired-records.json",
        "reports/evaluation.report.json",
        "providers/subject/runtime-scoring.observation.json",
        "providers/subject/runtime-provider.receipt.json",
    ],
)
@pytest.mark.parametrize("payload", [None, [], {}, 42, {"format": "invalid"}])
def test_malformed_signed_payload(relative, payload, tmp_path):
    directory = tmp_path.resolve()
    pack, policy, signer, runtimes, key, args = _publish(directory)
    target = pack / relative
    target.chmod(0o644)
    target.write_bytes(canonical_json_bytes(payload))
    _rebind_and_resign_pack(pack, key)
    anchors = _verification_anchors(args)
    config = {
        "evidence_path": str(pack),
        "policy_path": str(policy),
        "expected_baseline_artifact": anchors["expected_artifact_digests"]["baseline"],
        "expected_subject_artifact": anchors["expected_artifact_digests"]["subject"],
        "expected_schedule": anchors["expected_schedule_digest"],
        "expected_baseline_runtime": runtimes["baseline"],
        "expected_subject_runtime": runtimes["subject"],
        "expected_signer": signer,
        "expected_request_digest": anchors["expected_request_digest"],
        "verifier_signing_key_path": str(key),
        "verifier_identity": "fixture-verifier",
    }
    for name in ("reference", "native"):
        (directory / name).mkdir()
    source = {
        k: Path(v)
        if k in ("evidence_path", "policy_path", "verifier_signing_key_path")
        else v
        for k, v in config.items()
    }
    with pytest.raises(EvidenceVerificationError) as rejected:
        verify_evidence(**source, receipt_path=directory / "reference/receipt.json")
    config["receipt_path"] = str(directory / "native/receipt.json")
    result = run(config)
    assert result["status"] == rejected.value.exit_code
    assert result["payload"] == rejected.value.payload
    assert (directory / "native/receipt.json").read_bytes() == (
        directory / "reference/receipt.json"
    ).read_bytes()
