from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.release import gguf_runtime_blackbox as blackbox


def valid_receipt(
    *,
    image_digest: str,
    batch_size: int = 32,
    observation_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "artifact_identity": blackbox._expected_artifact(),
        "backend": {
            "binary_sha256": "a" * 64,
            "build_sha256": None,
            "name": "llama.cpp",
            "source_sha256": blackbox.LLAMA_CPP_SOURCE_SHA256,
            "version": (
                "version: 10015 "
                f"({blackbox.LLAMA_CPP_SOURCE_COMMIT}) "
                "built with Test for Linux x86_64"
            ),
        },
        "capabilities": {
            "metrics": ["exact_match"],
            "provider_name": "llama_cpp",
            "supported_claim_sets": ["invarlock-runtime-behavioral-regression-v1"],
            "tasks": ["text_causal"],
        },
        "device": {"device_kind": "cpu"},
        "execution_settings": {
            "allow_network": False,
            "batch_size": batch_size,
            "context_length": 256,
            "max_output_tokens": 16,
            "seed": 7,
            "timeout_seconds": 120,
        },
        "format_version": "invarlock/runtime-provider-receipt-v1",
        "outer_image_digest": image_digest,
        "plugin": {
            "distribution": "invarlock",
            "name": "llama_cpp",
            "provider_abi": "1",
        },
        "scoring_observation_sha256": (
            observation_sha256 or blackbox.SCORING_OBSERVATION_SHA256
        ),
    }


def valid_cli_journey(*, image_digest: str) -> dict[str, object]:
    observation = blackbox._expected_observation(
        schedule_sha256=blackbox.CLI_SCHEDULE_SHA256
    )
    receipt = valid_receipt(
        image_digest=image_digest,
        batch_size=1,
        observation_sha256=blackbox.CLI_SCORING_OBSERVATION_SHA256,
    )
    return {
        "artifact_identity_sha256": blackbox.ARTIFACT_IDENTITY_SHA256,
        "binding_sha256": "1" * 64,
        "execution_settings_sha256": blackbox.CLI_EXECUTION_SETTINGS_SHA256,
        "format_version": blackbox.CLI_JOURNEY_FORMAT,
        "observation": observation,
        "observation_sha256": blackbox.CLI_SCORING_OBSERVATION_SHA256,
        "policy_digest": "sha256:" + "2" * 64,
        "policy_file_sha256": "3" * 64,
        "portable_artifact_count": 17,
        "provider_receipt": receipt,
        "provider_receipt_sha256": hashlib.sha256(
            blackbox._canonical_json(receipt)
        ).hexdigest(),
        "schedule_sha256": blackbox.CLI_SCHEDULE_SHA256,
        "verification": {
            "baseline_score": 1.0,
            "regression": 0.0,
            "subject_score": 1.0,
            "verdict": "pass",
        },
    }


def valid_result(*, image_digest: str) -> dict[str, object]:
    return {
        "cli_journey": valid_cli_journey(image_digest=image_digest),
        "fixture": {
            "byte_length": blackbox.FIXTURE_BYTE_LENGTH,
            "repository": blackbox.FIXTURE_REPOSITORY,
            "revision": blackbox.FIXTURE_REVISION,
            "sha256": blackbox.FIXTURE_SHA256,
        },
        "format_version": blackbox.RESULT_FORMAT,
        "image_digest": image_digest,
        "observation": blackbox._expected_observation(
            schedule_sha256=blackbox.SCHEDULE_SHA256
        ),
        "receipt": valid_receipt(image_digest=image_digest),
    }


def exact_schedule() -> dict[str, object]:
    return {
        "dataset_identity": {
            "config_name": None,
            "dataset_name": None,
            "provider": "local_manifest",
            "revision": None,
            "split": "release-canary",
        },
        "format_version": "invarlock/runtime-behavioral-schedule-v1",
        "records": [
            {
                "expected_output": blackbox.EXPECTED_OUTPUT,
                "input_sha256": hashlib.sha256(
                    blackbox.PROMPT.encode("utf-8")
                ).hexdigest(),
                "input_text": blackbox.PROMPT,
                "record_id": blackbox.RECORD_ID,
            }
        ],
    }


def write_json(path: Path, value: object, *, manifest: bool = False) -> None:
    if manifest:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    else:
        payload = blackbox._canonical_json(value)
    path.write_bytes(payload)


def write_side_bundle(side: Path, *, role: str, image_digest: str) -> None:
    side.mkdir()
    write_json(
        side / "evaluation.report.json",
        {
            "role": role,
            "verdict": "observation_verified",
            "score": 1.0,
            "correct_records": 1,
            "total_records": 1,
            "schedule_sha256": blackbox.CLI_SCHEDULE_SHA256,
        },
    )
    write_json(side / "model-artifact.identity.json", blackbox._expected_artifact())
    write_json(side / "runtime-behavior.config.json", {"role": role})
    write_json(
        side / "runtime-provider.receipt.json",
        valid_receipt(
            image_digest=image_digest,
            batch_size=1,
            observation_sha256=blackbox.CLI_SCORING_OBSERVATION_SHA256,
        ),
    )
    write_json(
        side / "runtime-scoring.observation.json",
        blackbox._expected_observation(schedule_sha256=blackbox.CLI_SCHEDULE_SHA256),
    )
    write_json(side / "runtime.manifest.json", {"files": []}, manifest=True)
