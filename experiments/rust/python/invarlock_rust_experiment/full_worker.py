"""Installed Python and whole Rust transaction workers for the expanded matrix."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from .adapters import candidate_adapter
from .worker import block_network, check_wheel_binding, runtime_identity, sha_file


def plain(value):
    return json.loads(json.dumps(value))


def python_step(operation, options):
    config = dict(options)
    for key in (
        "evidence_path",
        "policy_path",
        "receipt_path",
        "verifier_signing_key_path",
        "recipient_policy_path",
        "stored_result_path",
        "html_path",
        "markdown_path",
        "junit_path",
    ):
        if config.get(key) is not None:
            config[key] = Path(config[key])
    if operation == "report":
        from invarlock.evidence_reporting import render_evidence

        result = render_evidence(**config)
        return {
            "payload": {"text": result.text, "report": json.loads(result.as_json())},
            "status": 0,
        }
    if operation == "verify-captured":
        from invarlock.captured_verification import verify_captured_evidence

        evidence = config.pop("evidence_path")
        result = verify_captured_evidence(evidence, **config)
        return {"payload": result, "status": 0 if result["ok"] else 7}
    if operation == "verify-native":
        from invarlock.evidence_verification import (
            EvidenceVerificationError,
            verify_evidence,
        )

        evidence = config.pop("evidence_path")
        try:
            result = verify_evidence(evidence, **config)
        except EvidenceVerificationError as error:
            return {"payload": error.payload, "status": error.exit_code}
        return {"payload": result.payload, "status": 0}
    if operation == "verify-captured-receipt":
        from invarlock.captured_verification import verify_captured_receipt

        evidence, receipt = config.pop("evidence_path"), config.pop("receipt_path")
        result = verify_captured_receipt(receipt, evidence, **config)
        return {
            "payload": plain(dataclasses.asdict(result)),
            "status": 0 if result.ok else 4,
        }
    if operation == "verify-native-receipt":
        from invarlock.evidence_receipt import verify_signed_verification_receipt

        result = verify_signed_verification_receipt(
            config["receipt_path"],
            config["evidence_path"],
            policy_path=config["policy_path"],
            expected_artifact_digests={
                s: config[f"expected_{s}_artifact"] for s in ("baseline", "subject")
            },
            expected_runtime_digests={
                s: config[f"expected_{s}_runtime"] for s in ("baseline", "subject")
            },
            expected_schedule_digest=config["expected_schedule"],
            expected_pack_signer_fingerprint=config["expected_signer"],
            expected_request_digest=config.get("expected_request_digest"),
            expected_verifier_identity=config["expected_verifier_identity"],
            expected_verifier_fingerprint=config["expected_verifier_fingerprint"],
        )
        return {
            "payload": plain(dataclasses.asdict(result)),
            "status": 0 if result.ok else 4,
        }
    if operation == "verify-judge":
        from invarlock.judge_measurements.acceptance import verify_judge_evidence

        result = verify_judge_evidence(
            config["evidence_path"],
            recipient_policy_path=config["recipient_policy_path"],
        )
        return {
            "payload": result.to_dict(),
            "status": 0 if result.accepted else 7 if result.verified else 4,
        }
    if operation == "write-judge-receipt":
        from invarlock.judge_measurements.acceptance import (
            _result_from_dict,
            write_signed_judge_verification_receipt,
        )

        fingerprint = write_signed_judge_verification_receipt(
            config["evidence_path"],
            _result_from_dict(config["result"]),
            config["receipt_path"],
            recipient_policy_path=config["recipient_policy_path"],
            verifier_identity=config["verifier_identity"],
            verifier_signing_key_path=config["verifier_signing_key_path"],
        )
        return {"payload": {"verifier_fingerprint": fingerprint}, "status": 0}
    if operation == "verify-judge-receipt":
        from invarlock.judge_measurements.acceptance import (
            verify_signed_judge_verification_receipt,
        )

        result = verify_signed_judge_verification_receipt(**config)
        payload = plain(dataclasses.asdict(result))
        if result.result is not None:
            payload["result"] = result.result.to_dict()
        return {"payload": payload, "status": 0 if result.ok else 4}
    if operation == "verify-evidence-set":
        from invarlock.evidence_sets.verification import verify_evidence_set

        result = verify_evidence_set(
            config["evidence_path"],
            recipient_policy=config["recipient_policy_path"],
            receipt=config.get("receipt_path"),
        )
        return {"payload": result.payload, "status": result.exit_code}
    if operation == "verify-stored-evidence-set":
        from invarlock.evidence_sets.verification import (
            verify_stored_evidence_set_result,
        )

        result = verify_stored_evidence_set_result(
            config["stored_result_path"],
            config["evidence_path"],
            recipient_policy=config["recipient_policy_path"],
        )
        return {"payload": result.payload, "status": result.exit_code}
    raise ValueError(f"unsupported reference operation: {operation}")


def execute(args):
    identity_started = time.perf_counter()
    identity = runtime_identity()
    identity["wheel_bindings"] = check_wheel_binding(args.metadata)
    identity["check_wall_seconds"] = time.perf_counter() - identity_started
    request = json.loads(Path(args.request).read_bytes())
    previous = None
    outputs = []
    native_calls = 0
    if args.candidate == "rust-ffi":
        import invarlock_native

        started = time.perf_counter()
        for step in request["steps"]:
            config = dict(step["config"])
            if step.get("previous_result"):
                config["result"] = previous
            result = json.loads(
                invarlock_native.dispatch_json(
                    step["operation"], json.dumps(config).encode()
                )
            )
            result = {key: result[key] for key in ("status", "payload")}
            outputs.append(result)
            previous = result["payload"]
            native_calls += 1
        elapsed = time.perf_counter() - started
    else:
        with candidate_adapter(args.candidate):
            started = time.perf_counter()
            for step in request["steps"]:
                config = dict(step["config"])
                if step.get("previous_result"):
                    config["result"] = previous
                result = python_step(step["operation"], config)
                outputs.append(result)
                previous = result["payload"]
            elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "identity": identity,
        "operation_wall_seconds": elapsed,
        "native_transactions": native_calls,
        "result": {
            "outputs": outputs,
            "files": {
                label: "sha256:" + sha_file(path)
                for label, path in request["files"].items()
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", choices=("reference", "optimized", "rust-ffi"), required=True
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)
    try:
        print(json.dumps(execute(args), allow_nan=False, sort_keys=True))
    except Exception as error:
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    return 0
