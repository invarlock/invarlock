"""Installed-CLI journey support for the pinned GGUF release black-box."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

BlackBoxAPI = Mapping[str, object]


def _value(api: BlackBoxAPI, name: str) -> Any:
    return api[name]


def _call(api: BlackBoxAPI, name: str) -> Callable[..., Any]:
    return cast(Callable[..., Any], api[name])


def _error(api: BlackBoxAPI, message: str) -> RuntimeError:
    error_type = cast(type[RuntimeError], api["GGUFBlackBoxError"])
    return error_type(message)


def write_canonical_new(api: BlackBoxAPI, path: Path, value: object) -> None:
    canonical_json = _call(api, "_canonical_json")
    payload = cast(bytes, canonical_json(value))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise _error(api, "a private CLI input could not be created") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise _error(api, "a private CLI input could not be written") from exc
    finally:
        os.close(descriptor)


def run_installed_cli(
    api: BlackBoxAPI,
    arguments: Sequence[str],
    *,
    expected_format: str,
    expect_success: bool = True,
    timeout_seconds: int = 240,
) -> dict[str, object]:
    run_captured = _call(api, "_run_captured")
    status, stdout, _stderr = cast(
        tuple[int, bytes, bytes],
        run_captured(
            (
                _value(api, "_CONTAINER_CLI"),
                "advanced",
                "runtime-behavior",
                *arguments,
                "--json",
            ),
            timeout_seconds=timeout_seconds,
            stdout_limit=_value(api, "_MAX_CLI_OUTPUT_BYTES"),
            stderr_limit=_value(api, "_MAX_CLI_OUTPUT_BYTES"),
        ),
    )
    if not stdout.endswith(b"\n") or stdout.endswith(b"\n\n"):
        raise _error(api, "the installed CLI result framing is invalid")
    try:
        decoded = json.loads(stdout[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(api, "the installed CLI result is invalid") from exc
    canonical_json = _call(api, "_canonical_json")
    if not isinstance(decoded, dict) or canonical_json(decoded) != stdout[:-1]:
        raise _error(api, "the installed CLI result is not canonical JSON")
    if decoded.get("format_version") != expected_format:
        raise _error(api, "the installed CLI result format does not match")
    expected_status = 0 if expect_success else 2
    if status != expected_status or decoded.get("ok") is not expect_success:
        raise _error(api, "the installed CLI command outcome does not match")
    return cast(dict[str, object], decoded)


def path_free_strings(api: BlackBoxAPI, value: object) -> bool:
    if isinstance(value, str):
        windows_path = cast(Any, _value(api, "_WINDOWS_PATH"))
        return (
            not any(
                marker in value
                for marker in ("/tmp/", "/fixtures/", "/opt/", "/Users/", "/root/")
            )
            and windows_path.search(value) is None
        )
    if isinstance(value, list):
        return all(path_free_strings(api, item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and path_free_strings(api, key)
            and path_free_strings(api, item)
            for key, item in value.items()
        )
    return value is None or isinstance(value, bool | int | float)


def portable_json(
    api: BlackBoxAPI,
    path: Path,
    *,
    manifest: bool = False,
) -> tuple[bytes, dict[str, object]]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _value(
            api, "_MAX_RESULT_BYTES"
        ):
            raise _error(api, "a CLI artifact is not a bounded regular file")
        payload = path.read_bytes()
    except OSError as exc:
        raise _error(api, "a CLI artifact could not be read") from exc

    duplicate = False

    def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                duplicate = True
            result[key] = item
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(api, "a CLI artifact is not valid JSON") from exc
    if duplicate or not isinstance(decoded, dict):
        raise _error(api, "a CLI artifact is not a unique JSON object")
    canonical_json = _call(api, "_canonical_json")
    expected = (
        (json.dumps(decoded, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        if manifest
        else canonical_json(decoded)
    )
    if payload != expected:
        raise _error(api, "a CLI artifact does not use its canonical encoding")
    if not path_free_strings(api, decoded):
        raise _error(api, "a CLI artifact contains a host or runtime path")
    return payload, cast(dict[str, object], decoded)


def native_cli_arguments(
    api: BlackBoxAPI, *, image_digest: str, settings: Path
) -> tuple[str, ...]:
    fixture_sha256 = _value(api, "FIXTURE_SHA256")
    return (
        "--provider",
        "llama_cpp",
        "--model-id",
        f"gguf-sha256-{fixture_sha256}.gguf",
        "--settings",
        str(settings),
        "--artifact",
        _value(api, "_CONTAINER_MODEL"),
        "--backend-executable",
        _value(api, "_CONTAINER_EXECUTABLE"),
        "--backend-source",
        _value(api, "_CONTAINER_SOURCE"),
        "--container-image-digest",
        image_digest,
    )


def expected_side_bindings(api: BlackBoxAPI, side: Path) -> dict[str, str]:
    sha256_file = _call(api, "_sha256_file_unbounded")
    return {
        "artifact_identity_sidecar_sha256": sha256_file(
            side / "model-artifact.identity.json"
        ),
        "evaluation_report_sha256": sha256_file(side / "evaluation.report.json"),
        "provider_receipt_sidecar_sha256": sha256_file(
            side / "runtime-provider.receipt.json"
        ),
        "runtime_manifest_sha256": sha256_file(side / "runtime.manifest.json"),
        "scoring_observation_sidecar_sha256": sha256_file(
            side / "runtime-scoring.observation.json"
        ),
    }


def validate_cli_side(
    api: BlackBoxAPI,
    side: Path,
    *,
    role: str,
    image_digest: str,
) -> dict[str, object]:
    expected_names = {
        "evaluation.report.json",
        "model-artifact.identity.json",
        "runtime-behavior.config.json",
        "runtime-provider.receipt.json",
        "runtime-scoring.observation.json",
        "runtime.manifest.json",
    }
    try:
        if (
            side.is_symlink()
            or {entry.name for entry in side.iterdir()} != expected_names
        ):
            raise _error(api, "a CLI side bundle has an unexpected file set")
    except OSError as exc:
        raise _error(api, "a CLI side bundle cannot be inspected") from exc
    decoded: dict[str, dict[str, object]] = {}
    portable = _call(api, "_portable_json")
    for name in sorted(expected_names):
        _payload, value = portable(
            side / name,
            manifest=name == "runtime.manifest.json",
        )
        decoded[name] = value
    observation = decoded["runtime-scoring.observation.json"]
    expected_observation = _call(api, "_expected_observation")
    schedule_sha256 = _value(api, "CLI_SCHEDULE_SHA256")
    if observation != expected_observation(schedule_sha256=schedule_sha256):
        raise _error(api, "the CLI side observation does not match the pin")
    canonical_json = _call(api, "_canonical_json")
    if hashlib.sha256(canonical_json(observation)).hexdigest() != _value(
        api, "CLI_SCORING_OBSERVATION_SHA256"
    ):
        raise _error(api, "the CLI side observation digest does not match")
    receipt = decoded["runtime-provider.receipt.json"]
    validate_receipt = _call(api, "_validate_provider_receipt")
    validate_receipt(
        receipt,
        image_digest=image_digest,
        batch_size=1,
        observation_sha256=_value(api, "CLI_SCORING_OBSERVATION_SHA256"),
    )
    report = decoded["evaluation.report.json"]
    if (
        report.get("role") != role
        or report.get("verdict") != "observation_verified"
        or report.get("score") != 1.0
        or report.get("correct_records") != 1
        or report.get("total_records") != 1
        or report.get("schedule_sha256") != schedule_sha256
    ):
        raise _error(api, "the CLI side report did not verify the known answer")
    return receipt


def inside_cli_journey(api: BlackBoxAPI, *, image_digest: str) -> dict[str, object]:
    for name in (
        "INVARLOCK_ALLOW_HOST_EXECUTION",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE",
    ):
        if os.environ.get(name) != "0":
            raise _error(api, "the CLI journey requires a fail-closed environment")
    cli = Path(_value(api, "_CONTAINER_CLI"))
    try:
        cli_metadata = cli.lstat()
    except OSError as exc:
        raise _error(api, "the installed InvarLock CLI is unavailable") from exc
    if not stat.S_ISREG(cli_metadata.st_mode) or not os.access(cli, os.X_OK):
        raise _error(api, "the installed InvarLock CLI is not executable")

    root = Path(_value(api, "_CONTAINER_WORK_ROOT"))
    try:
        root.mkdir(mode=0o700)
    except OSError as exc:
        raise _error(api, "the private CLI workspace could not be created") from exc
    records = root / "records.json"
    dataset = root / "dataset.json"
    settings = root / "settings.json"
    schedule = root / "schedule.json"
    baseline_binding = root / "baseline-binding.json"
    subject_binding = root / "subject-binding.json"
    policy = root / "policy.json"
    baseline_side = root / "baseline-side"
    subject_side = root / "subject-side"
    pair_receipt = root / "pair-receipt.json"

    write_new = _call(api, "_write_canonical_new")
    write_new(
        records,
        [
            {
                "expected_output": _value(api, "EXPECTED_OUTPUT"),
                "input_text": _value(api, "PROMPT"),
                "record_id": _value(api, "RECORD_ID"),
            }
        ],
    )
    write_new(
        dataset,
        {
            "config_name": None,
            "dataset_name": None,
            "provider": "local_manifest",
            "revision": None,
            "split": "release-canary",
        },
    )
    sha256_file = _call(api, "_sha256_file_unbounded")
    backend_binary_sha256 = sha256_file(Path(_value(api, "_CONTAINER_EXECUTABLE")))
    normalized_backend_version = _call(api, "_normalized_backend_version")
    write_new(
        settings,
        {
            "artifact_byte_length": _value(api, "FIXTURE_BYTE_LENGTH"),
            "artifact_sha256": _value(api, "FIXTURE_SHA256"),
            "backend_binary_sha256": backend_binary_sha256,
            "backend_source_sha256": _value(api, "LLAMA_CPP_SOURCE_SHA256"),
            "backend_version": normalized_backend_version(),
            "batch_size": 1,
            "context_length": 256,
            "gguf_metadata_sha256": _value(api, "FIXTURE_METADATA_SHA256"),
            "max_output_tokens": 16,
            "seed": 7,
            "tensor_inventory_sha256": _value(api, "FIXTURE_TENSOR_INVENTORY_SHA256"),
            "timeout_seconds": 120,
            "tokenizer_metadata_sha256": _value(
                api, "FIXTURE_TOKENIZER_METADATA_SHA256"
            ),
        },
    )

    run_cli = _call(api, "_run_installed_cli")
    schedule_arguments = (
        "build-schedule",
        "--records",
        str(records),
        "--dataset-identity",
        str(dataset),
        "--out",
        str(schedule),
    )
    schedule_result = run_cli(
        schedule_arguments,
        expected_format="runtime-behavior-build-schedule-cli-v1",
    )
    if schedule_result.get("schedule_sha256") != _value(api, "CLI_SCHEDULE_SHA256"):
        raise _error(api, "the installed CLI schedule does not match the pin")
    schedule_before = sha256_file(schedule)
    run_cli(
        schedule_arguments,
        expected_format="runtime-behavior-build-schedule-cli-v1",
        expect_success=False,
    )
    if sha256_file(schedule) != schedule_before:
        raise _error(api, "the installed CLI clobbered its schedule")

    native_arguments = _call(api, "_native_cli_arguments")(
        image_digest=image_digest,
        settings=settings,
    )
    for output in (baseline_binding, subject_binding):
        binding_result = run_cli(
            ("prepare-binding", *native_arguments, "--out", str(output)),
            expected_format="runtime-behavior-prepare-binding-cli-v1",
        )
        if binding_result.get("artifact_identity_sha256") != _value(
            api, "ARTIFACT_IDENTITY_SHA256"
        ) or binding_result.get("execution_settings_sha256") != _value(
            api, "CLI_EXECUTION_SETTINGS_SHA256"
        ):
            raise _error(api, "the installed CLI binding does not match the pin")
    portable = _call(api, "_portable_json")
    baseline_binding_bytes, baseline_binding_value = portable(baseline_binding)
    subject_binding_bytes, subject_binding_value = portable(subject_binding)
    expected_binding = {
        "artifact_format": "gguf",
        "artifact_identity_sha256": _value(api, "ARTIFACT_IDENTITY_SHA256"),
        "execution_settings_sha256": _value(api, "CLI_EXECUTION_SETTINGS_SHA256"),
        "outer_image_digest": image_digest,
        "provider_name": "llama_cpp",
    }
    if (
        baseline_binding_value != expected_binding
        or subject_binding_value != expected_binding
        or baseline_binding_bytes != subject_binding_bytes
    ):
        raise _error(api, "the directed CLI bindings are not exact")

    policy_result = run_cli(
        (
            "build-policy",
            "--schedule",
            str(schedule),
            "--baseline-binding",
            str(baseline_binding),
            "--subject-binding",
            str(subject_binding),
            "--tier",
            "balanced",
            "--minimum-subject-score",
            "1.0",
            "--maximum-regression",
            "0.0",
            "--evidence-surface",
            "behavior",
            "--evidence-surface",
            "tokenizer",
            "--out",
            str(policy),
        ),
        expected_format="runtime-behavior-build-policy-cli-v1",
    )
    policy_bytes, policy_value = portable(policy)
    policy_digest = policy_result.get("policy_digest")
    claim = policy_value.get("behavioral_claim")
    policy_digest_pattern = cast(Any, _value(api, "_POLICY_DIGEST"))
    if (
        not isinstance(policy_digest, str)
        or policy_digest_pattern.fullmatch(policy_digest) is None
        or policy_value.get("policy_digest") != policy_digest
        or not isinstance(claim, dict)
        or claim.get("schedule_sha256") != _value(api, "CLI_SCHEDULE_SHA256")
        or claim.get("baseline") != expected_binding
        or claim.get("subject") != expected_binding
    ):
        raise _error(api, "the installed CLI policy is not exactly directed")

    validate_side = _call(api, "_validate_cli_side")
    for role, output in (("baseline", baseline_side), ("subject", subject_side)):
        run_cli(
            (
                "run-side",
                "--role",
                role,
                *native_arguments,
                "--schedule",
                str(schedule),
                "--policy-pack",
                str(policy),
                "--out",
                str(output),
            ),
            expected_format="runtime-behavior-run-side-cli-v1",
        )
        validate_side(output, role=role, image_digest=image_digest)

    pair_arguments = (
        "verify-pair",
        "--baseline",
        str(baseline_side),
        "--subject",
        str(subject_side),
        "--schedule",
        str(schedule),
        "--policy-pack",
        str(policy),
        "--receipt",
        str(pair_receipt),
    )
    pair_result = run_cli(
        pair_arguments,
        expected_format="runtime-behavior-verify-pair-cli-v1",
    )
    pair_bytes, pair_value = portable(pair_receipt)
    side_bindings = _call(api, "_expected_side_bindings")
    expected_pair = {
        "baseline": side_bindings(baseline_side),
        "baseline_score": 1.0,
        "claim_set": "invarlock-runtime-behavioral-regression-v1",
        "format_version": "invarlock/runtime-behavioral-claim-receipt-v1",
        "metric": "exact_match",
        "policy_digest": policy_digest,
        "regression": 0.0,
        "schedule_sha256": _value(api, "CLI_SCHEDULE_SHA256"),
        "subject": side_bindings(subject_side),
        "subject_score": 1.0,
        "verdict": "pass",
    }
    if pair_value != expected_pair or any(
        pair_result.get(name) != value
        for name, value in (
            ("baseline_score", 1.0),
            ("subject_score", 1.0),
            ("regression", 0.0),
        )
    ):
        raise _error(api, "the installed CLI paired receipt did not pass")
    pair_before = hashlib.sha256(pair_bytes).hexdigest()
    run_cli(
        pair_arguments,
        expected_format="runtime-behavior-verify-pair-cli-v1",
        expect_success=False,
    )
    if sha256_file(pair_receipt) != pair_before:
        raise _error(api, "the installed CLI clobbered its paired receipt")

    portable_paths = [schedule, baseline_binding, subject_binding, policy, pair_receipt]
    portable_paths.extend(sorted(baseline_side.iterdir()))
    portable_paths.extend(sorted(subject_side.iterdir()))
    if len(portable_paths) != 17:
        raise _error(api, "the CLI portable artifact inventory is incomplete")
    for path in portable_paths:
        portable(path, manifest=path.name == "runtime.manifest.json")

    baseline_receipt = baseline_side / "runtime-provider.receipt.json"
    subject_receipt = subject_side / "runtime-provider.receipt.json"
    baseline_receipt_bytes, baseline_receipt_value = portable(baseline_receipt)
    if baseline_receipt_bytes != subject_receipt.read_bytes():
        raise _error(api, "the same-artifact CLI receipts are not deterministic")
    _observation_bytes, observation_value = portable(
        baseline_side / "runtime-scoring.observation.json"
    )
    return {
        "artifact_identity_sha256": _value(api, "ARTIFACT_IDENTITY_SHA256"),
        "binding_sha256": hashlib.sha256(baseline_binding_bytes).hexdigest(),
        "execution_settings_sha256": _value(api, "CLI_EXECUTION_SETTINGS_SHA256"),
        "format_version": _value(api, "CLI_JOURNEY_FORMAT"),
        "observation": observation_value,
        "observation_sha256": _value(api, "CLI_SCORING_OBSERVATION_SHA256"),
        "policy_digest": policy_digest,
        "policy_file_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "portable_artifact_count": len(portable_paths),
        "provider_receipt": baseline_receipt_value,
        "provider_receipt_sha256": hashlib.sha256(baseline_receipt_bytes).hexdigest(),
        "schedule_sha256": _value(api, "CLI_SCHEDULE_SHA256"),
        "verification": {
            "baseline_score": 1.0,
            "regression": 0.0,
            "subject_score": 1.0,
            "verdict": "pass",
        },
    }
