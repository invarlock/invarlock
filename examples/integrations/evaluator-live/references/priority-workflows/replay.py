#!/usr/bin/env python3
"""Replay original priority-workflow captures and signed EM/NLL evidence offline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORICAL_HTTP_HELPER = HERE / "http_service.py.txt"
HISTORICAL_HTTP_HELPER_SHA256 = (
    "sha256:3356817238c0d6a7632d3369fe744efeb4fd40a2cba4586187d69cab4ec81dda"
)
ARCHIVES = {
    "captures.zip": "f0cc24c07ea81306efae3146176e28b1416b427df1730fff3d1eba3497be9cd5",
    "exact-match.zip": "05268c70a541552a24e08784a941ea4535eb677737d87d17ab54f06a01c18faf",
    "normalized-nll.zip": "4e1d77ffd11f202d715a8e3fb40d599b1647f5a783bfb19d0104ef66a5467b6e",
}
MANIFEST_SHA256 = "e1b838872a981cae788ffc63da9c2bf8a4186340fa60c224349fe2142c910fad"
EVALUATORS = ("inspect-ai", "lm-evaluation-harness", "promptfoo", "langfuse")
PROFILES = {
    "local": (64, "local_artifact"),
    "http": (8, "controlled_http_task_service"),
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


SHARED = module(
    "priority_shared_reference", HERE.parent / "mistral-7b-sentinel/replay.py"
)


def verify_historical_http_helper(protocol, role):
    """Bind the archived source bytes to the original HTTP protocol pin."""
    source_digest = "sha256:" + SHARED.sha(HISTORICAL_HTTP_HELPER.read_bytes())
    if (
        source_digest != HISTORICAL_HTTP_HELPER_SHA256
        or protocol["http_services"][role]["helper_sha256"] != source_digest
    ):
        raise ValueError("historical HTTP helper source differs from its pin")


def validate_reference(reference):
    if (
        reference["format"] != "invarlock/evaluator-priority-reference-v1"
        or reference["evaluators"] != list(EVALUATORS)
        or set(reference["source_profiles"]) != set(PROFILES)
        or reference["signed_packs"] != 32
        or reference["fresh_model_case_requests"] != 576
    ):
        raise ValueError("reference scope differs from the retained campaign")
    for name, (count, kind) in PROFILES.items():
        profile = reference["source_profiles"][name]
        if (profile["cases_per_role"], profile["kind"]) != (count, kind):
            raise ValueError("reference source profile differs")
    expected = {
        (profile, evaluator, route, scorer)
        for profile in PROFILES
        for evaluator in EVALUATORS
        for route in ("envelope", "native-json")
        for scorer in ("exact_match", "normalized_nll")
    }
    actual = {
        (row["profile"], row["evaluator"], row["route"], row["scorer"])
        for row in reference["packs"]
    }
    if actual != expected or len(reference["packs"]) != len(expected):
        raise ValueError("reference pack inventory is incomplete or duplicated")
    expected_captures = {
        (profile, evaluator, role)
        for profile in PROFILES
        for evaluator in EVALUATORS
        for role in ("baseline", "subject")
    }
    if (
        len(reference["captures"]) != 16
        or {
            (row["profile"], row["evaluator"], row["role"])
            for row in reference["captures"]
        }
        != expected_captures
    ):
        raise ValueError("reference capture inventory differs")


def read_reference(directory):
    files = {}
    for name, pin in ARCHIVES.items():
        incoming = SHARED.read_archive(
            Path(directory) / name,
            expected_sha256=pin,
            maximum_bytes=SHARED.ARCHIVE_LIMIT,
            maximum_expanded_bytes=SHARED.EXPANDED_LIMIT,
        )
        if files.keys() & incoming.keys():
            raise ValueError("reference archives overlap")
        files.update(incoming)
    raw = files["archive-manifest.json"]
    if SHARED.sha(raw) != MANIFEST_SHA256:
        raise ValueError("reference manifest hash differs")
    manifest = json.loads(raw)
    if set(manifest) != {"files"} or set(manifest["files"]) != set(files) - {
        "archive-manifest.json"
    }:
        raise ValueError("reference member inventory differs")
    for name, expected in manifest["files"].items():
        if expected != {
            "sha256": "sha256:" + SHARED.sha(files[name]),
            "bytes": len(files[name]),
        }:
            raise ValueError("reference member bytes differ")
    reference = json.loads(files["reference.json"])
    validate_reference(reference)
    return files, reference


def check_records(recipient, run, results, protocol, evaluator, service_identity):
    """Apply the maintained recipient's record and retained-context checks."""
    indexed = {record["id"]: record for record in run["records"]}
    if len(indexed) != len(run["records"]) or set(indexed) != set(results):
        raise ValueError(
            "normalized capture differs from the complete planned schedule"
        )
    for case in protocol["cases"]:
        record, result = indexed[case["id"]], results[case["id"]]
        bound = recipient.bindings.check_record(
            record,
            result,
            case,
            evaluator,
            protocol["versions"][evaluator],
            cases=protocol["cases"],
            service_identity=service_identity,
        )
        if any(
            not recipient.contains(record, name, value)
            for name, value in case["metadata"].items()
        ):
            raise ValueError("native export omitted or changed frozen case metadata")
        metadata = {
            "invarlock_model_execution": result["metadata"][
                "invarlock_model_execution"
            ],
            "invarlock_task_outcome": {
                "output": result["output"],
                "error": result.get("error"),
            },
        }
        for name in ("invarlock_capture_binding", "invarlock_serialization_binding"):
            if name in bound["metadata"]:
                metadata[name] = bound["metadata"][name]
        if "invarlock_transport_replay" in result["metadata"]:
            metadata["invarlock_transport_replay"] = result["metadata"][
                "invarlock_transport_replay"
            ]
        if any(
            not recipient.contains(record, name, value)
            for name, value in metadata.items()
        ):
            raise ValueError(
                "native export omitted or changed original execution metadata"
            )


def verify_sources(root, reference):
    """Bind real SDK/task/HTTP journals to the original normalized signed runs."""
    from invarlock.engine import load_run
    from invarlock.evaluation_records.io import run_digest

    recipient = module("priority_capture_recipient", HERE.parents[1] / "recipient.py")
    captures = {}
    for row in reference["captures"]:
        profile = reference["source_profiles"][row["profile"]]
        raw = (root / profile["protocol"]).read_bytes()
        if SHARED.sha(raw) != profile["protocol_sha256"]:
            raise ValueError("source protocol hash differs")
        protocol = json.loads(raw)
        if row["profile"] == "http":
            verify_historical_http_helper(protocol, row["role"])
        native, _, results = recipient.capture(
            root / row["directory"], protocol, row["role"], row["evaluator"]
        )
        captures[row["profile"], row["evaluator"], row["role"]] = (
            native,
            results,
            protocol,
        )
    for row in reference["packs"]:
        directory = root / row["directory"]
        request = json.loads((directory / "request.json").read_bytes())
        for role in ("baseline", "subject"):
            source = request["comparison"][role]
            path = directory / (role + ".json")
            original, results, protocol = captures[
                row["profile"], row["evaluator"], role
            ]
            retained = json.loads(path.read_bytes())
            payload = retained if row["route"] == "native-json" else retained["payload"]
            if row["route"] == "envelope" and row["evaluator"] == "langfuse":
                payload = payload["result"]
            if not recipient.same(payload, original):
                raise ValueError(
                    "recipient input differs from its original SDK capture"
                )
            run = load_run(path, **{k: v for k, v in source.items() if k != "path"})
            check_records(
                recipient,
                run,
                results,
                protocol,
                row["evaluator"],
                source.get("service_identity"),
            )
            if run_digest(run) != row["anchors"][role + "_run_digest"]:
                raise ValueError(
                    "original native source does not reproduce the signed run"
                )
    return len(captures)


def replay(directory):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    files, reference = read_reference(directory)
    key = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with tempfile.TemporaryDirectory(prefix="priority-reference-") as temporary:
        root = Path(temporary).resolve()
        for name, raw in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        captures = verify_sources(root, reference)
        receipts = root / "fresh-receipts"
        receipts.mkdir()
        results = [
            SHARED.verify_pack(root, row, receipts, key) for row in reference["packs"]
        ]
    return {
        "ok": True,
        "evaluators": list(EVALUATORS),
        "source_profiles": reference["source_profiles"],
        "original_captures_checked": captures,
        "signed_packs_replayed": len(results),
        "new_model_or_judge_calls": 0,
        "historical_policy_acceptance": False,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    args = parser.parse_args()
    sys.addaudithook(SHARED.block_network)
    SHARED.require_installed()
    print(json.dumps(replay(args.directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
