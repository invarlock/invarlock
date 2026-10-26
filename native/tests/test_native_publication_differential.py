"""Native publication/import authoring bytes and fresh verifier replay."""

import base64
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import RuntimeDeviceFacts
from invarlock.evidence_pack import EvidenceObservation
from invarlock.evidence_pack_contract import canonical_json_bytes, schedule_bytes
from tests.evidence_packs.test_evidence_pack import _publish, _verification_anchors
from tests.runtime import test_runtime_import_authoring as imports

ROOT = Path(__file__).resolve().parents[2]


def native(operation, config):
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


def files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("mode", ["exact", "fail", "observation"])
def test_native_publication_bytes(mode, tmp_path):
    directory = tmp_path.resolve()
    observations = (
        (
            EvidenceObservation(
                "note", "comparison", "note", canonical_json_bytes({"note": "retained"})
            ),
        )
        if mode == "observation"
        else ()
    )
    pack, policy, signer, runtimes, key, args = _publish(
        directory,
        delta_min_pp=-1 if mode == "fail" else -100,
        observations=observations,
    )
    identities = {
        role: asdict(args[role])
        for role in (
            "baseline",
            "subject",
            "dataset",
            "baseline_runtime",
            "subject_runtime",
            "policy",
        )
    }
    config = {
        "destination": str(directory / "native"),
        "comparison_id": args["comparison_id"],
        "identities": identities,
        "normalized_request": args["normalized_request"],
        "schedule_path": str(pack / "schedule/runtime-behavioral-schedule.json"),
        "policy_path": str(policy),
        "signing_key_path": str(key),
        "observations": [],
    }
    for side in ("baseline", "subject"):
        path = directory / ("side-" + side)
        path.mkdir()
        config[side + "_directory"] = str(path)
        for field, name in [
            ("run_report", "report.json"),
            ("runtime_manifest", "runtime.manifest.json"),
            ("runtime_config", "run.yaml"),
            ("artifact_identity", "model-artifact.identity.json"),
            ("provider_receipt", "runtime-provider.receipt.json"),
            ("scoring_observation", "runtime-scoring.observation.json"),
        ]:
            (path / name).write_bytes(getattr(args[side + "_evidence"], field))
    for observation in observations:
        path = directory / (observation.observation_id + ".json")
        path.write_bytes(observation.payload)
        config["observations"].append(
            {
                "id": observation.observation_id,
                "kind": observation.kind,
                "scope": observation.scope,
                "payload_path": str(path),
            }
        )
    status, result = native("publish-native", config)
    assert status == 0, result
    assert files(directory / "native") == files(pack)
    for path in (directory / "native").rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o555 if path.is_dir() else 0o444)
    anchors = _verification_anchors(args)
    verify = {
        "evidence_path": str(directory / "native"),
        "policy_path": str(policy),
        "expected_baseline_artifact": anchors["expected_artifact_digests"]["baseline"],
        "expected_subject_artifact": anchors["expected_artifact_digests"]["subject"],
        "expected_schedule": anchors["expected_schedule_digest"],
        "expected_baseline_runtime": runtimes["baseline"],
        "expected_subject_runtime": runtimes["subject"],
        "expected_signer": signer,
        "expected_request_digest": anchors["expected_request_digest"],
        "verifier_signing_key_path": str(key),
        "verifier_identity": "publication-fixture",
        "receipt_path": str(directory / "receipt.json"),
    }
    status, result = native("verify-native", verify)
    assert status == (7 if mode == "fail" else 0), result
    assert result["integrity_ok"]
    status, _ = native("publish-native", config)
    assert status != 0
    assert files(directory / "native") == files(pack)


@pytest.mark.parametrize("jsonl", [False, True])
@pytest.mark.parametrize(
    "observed",
    [
        "none",
        "valid",
        "missing_receipt",
        "missing_observation",
        "receipt_image",
        "receipt_hash",
        "observation_aggregate",
        "receipt_backend",
    ],
)
def test_runtime_import_side_bytes(jsonl, observed, tmp_path):
    directory = tmp_path.resolve()
    artifact = imports._artifact("fixture.gguf", "1")
    expected = imports._write_side(
        directory / "reference",
        role="baseline",
        artifact=artifact,
        outputs=("A", "B"),
        image_marker="a",
    )
    schedule = imports._schedule()
    schedule_path = directory / "schedule.json"
    schedule_path.write_bytes(schedule_bytes(schedule))
    config = {
        "directory": str(directory / "native"),
        "role": "baseline",
        "schedule_path": str(schedule_path),
        "policy_digest": imports._POLICY_DIGEST,
        "artifact_identity": asdict(artifact),
        "records": [asdict(row) for row in imports._records(("A", "B"))],
        "plugin": asdict(imports._plugin()),
        "backend": asdict(imports._backend()),
        "capabilities": asdict(imports._capabilities()),
        "execution_settings": asdict(imports._execution_settings()),
        "device": asdict(
            RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu")
        ),
        "runtime_image_ref": "registry.example/pinned-evaluator",
        "runtime_image_digest": "sha256:" + "a" * 64,
        "generated_at_utc": imports._GENERATED_AT,
    }
    if jsonl:
        path = directory / "records.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in config.pop("records"))
        )
        config["records_path"] = str(path)
    expected_error = None
    if observed != "none":
        config["observed_observation"] = json.loads(
            (expected.directory / "runtime-scoring.observation.json").read_bytes()
        )
        config["observed_receipt"] = json.loads(
            (expected.directory / "runtime-provider.receipt.json").read_bytes()
        )
        if observed == "missing_receipt":
            del config["observed_receipt"]
            expected_error = "must be supplied together"
        elif observed == "missing_observation":
            del config["observed_observation"]
            expected_error = "must be supplied together"
        elif observed == "receipt_image":
            config["observed_receipt"]["outer_image_digest"] = "sha256:" + "b" * 64
            expected_error = "outer_image_digest does not match"
        elif observed == "receipt_hash":
            config["observed_receipt"]["scoring_observation_sha256"] = "0" * 64
            expected_error = "scoring_observation_sha256 does not match"
        elif observed == "observation_aggregate":
            config["observed_observation"]["aggregate_source_sha256"] = "0" * 64
            config["observed_receipt"]["scoring_observation_sha256"] = hashlib.sha256(
                canonical_json_bytes(config["observed_observation"], newline=False)
            ).hexdigest()
            expected_error = "observed scoring observation does not match"
        elif observed == "receipt_backend":
            config["observed_receipt"]["backend"]["source_sha256"] = "0" * 64
            expected_error = "observed provider receipt does not match"
    status, result = native("write-runtime-side", config)
    if expected_error:
        assert status != 0, result
        assert expected_error in result["error"]
        assert not (directory / "native").exists()
        return
    assert status == 0, result
    if observed == "valid":
        for name, encoded in result["material"].items():
            assert (
                base64.b64decode(encoded, validate=True)
                == (expected.directory / name).read_bytes()
            )
        assert (
            result["receipt_sha256"]
            == hashlib.sha256(
                (expected.directory / "runtime-provider.receipt.json").read_bytes()
            ).hexdigest()
        )
    else:
        assert "material" not in result
        assert "receipt_sha256" not in result
    assert files(directory / "native") == files(expected.directory)
    config["expected_runtime_image_digest"] = config["runtime_image_digest"]
    status, result = native("load-runtime-side", config)
    assert status == 0, result
    assert result["record_count"] == 2
