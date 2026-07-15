#!/usr/bin/env python3
"""Run pinned native and installed-CLI GGUF black-boxes without fetching data."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

try:
    from scripts.release import _gguf_runtime_blackbox_cli as _cli_support
except ModuleNotFoundError as exc:
    if exc.name not in {
        "scripts",
        "scripts.release",
        "scripts.release._gguf_runtime_blackbox_cli",
    }:
        raise
    _cli_support = importlib.import_module("_gguf_runtime_blackbox_cli")

FIXTURE_REPOSITORY: Final = "ggml-org/tiny-llamas"
FIXTURE_REVISION: Final = "99dd1a73db5a37100bd4ae633f4cfce6560e1567"
FIXTURE_SHA256: Final = (
    "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04"
)
FIXTURE_BYTE_LENGTH: Final = 19_077_344
FIXTURE_METADATA_SHA256: Final = (
    "f13ce88664095c2aaf02a9850a824bf1948ddd6107e9b946b0cf667e0c22fc9b"
)
FIXTURE_TENSOR_INVENTORY_SHA256: Final = (
    "8f38cc5b0c8574c5f36b0ef15b894a182001e844b9ea85d2029a0243355481ff"
)
FIXTURE_TOKENIZER_METADATA_SHA256: Final = (
    "2e9c7c4fb5ec6d5e33c8aa63f92d505a77a2bdabe2170e47788883912ee06b25"
)

LLAMA_CPP_SOURCE_COMMIT: Final = "12127defda4f41b7679cb2477a4b0d65ee6a0c8f"
LLAMA_CPP_SOURCE_SHA256: Final = (
    "5ab75e394f4c71425ecce64a213dab3b8e3e9cfe0f19d0dcda4d5a4f7733da83"
)

PROMPT: Final = "Once upon a time"
EXPECTED_OUTPUT: Final = (
    ", there was a little girl named Lily. She loved to play outside and"
)
RECORD_ID: Final = "stories15m-q4-0-release-canary"
SCHEDULE_SHA256: Final = (
    "444ece399992388c55ffd22e33334f5833a6ad267a72249a41fe372adb9057e7"
)
RESULT_FORMAT: Final = "invarlock/gguf-runtime-blackbox-result-v1"
ARTIFACT_IDENTITY_SHA256: Final = (
    "ebcd080c74bb58f8b65cd1bf5bfd126731854814f9609b4ca981b404555b934c"
)
AGGREGATE_SOURCE_SHA256: Final = (
    "eafec7ee38be373016454c51d050a0d7f9ea52e1c38a21f6fa3f6578ccc24014"
)
SCORING_OBSERVATION_SHA256: Final = (
    "6bdbae284bb0db426d7264d7fe3b945df4c1e990675959e55528f18dfb872275"
)
CLI_SCHEDULE_SHA256: Final = (
    "7a7b07b3d944241da5634888d253343bcbbbe66d0a6449fc363d05c0788a5998"
)
CLI_SCORING_OBSERVATION_SHA256: Final = (
    "feae45dc3644c02cf3561041edc5b121cf9bf2b938abd3fdb59dfeb0d2650cd4"
)
CLI_EXECUTION_SETTINGS_SHA256: Final = (
    "09a5104366bad1745d5de2aaad8f2e705f95093258a0606c052e1b6430080adf"
)
CLI_JOURNEY_FORMAT: Final = "invarlock/gguf-runtime-cli-journey-v1"

_CONTAINER_SCRIPT: Final = "/opt/invarlock-blackbox/gguf_runtime_blackbox.py"
_CONTAINER_SUPPORT: Final = "/opt/invarlock-blackbox/_gguf_runtime_blackbox_cli.py"
_CONTAINER_MODEL: Final = "/fixtures/stories15M-q4_0.gguf"
_CONTAINER_EXECUTABLE: Final = "/opt/llama.cpp/llama-completion"
_CONTAINER_SOURCE: Final = "/opt/llama.cpp/source/llama.cpp-b10015.tar.gz"
_CONTAINER_CLI: Final = "/usr/local/bin/invarlock"
_CONTAINER_WORK_ROOT: Final = "/tmp/invarlock-gguf-cli"
_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_POLICY_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_WINDOWS_PATH = re.compile(r"(?i)(?:^|[^A-Za-z0-9])[A-Z]:[\\/]")
_VERSION = re.compile(
    rf"^version: 10015 \({LLAMA_CPP_SOURCE_COMMIT}\) "
    r"built with [ -~]{1,384} for Linux (?:aarch64|x86_64|amd64)$"
)
_MAX_INSPECT_BYTES: Final = 1024 * 1024
_MAX_RESULT_BYTES: Final = 4 * 1024 * 1024
_MAX_CLI_OUTPUT_BYTES: Final = 256 * 1024


class GGUFBlackBoxError(RuntimeError):
    """Raised when the optional GGUF release black-box fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GGUFBlackBoxError("the GGUF fixture cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise GGUFBlackBoxError("the GGUF fixture must be a regular file")
        if opened.st_size != FIXTURE_BYTE_LENGTH:
            raise GGUFBlackBoxError("the GGUF fixture byte length does not match")
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise GGUFBlackBoxError("the GGUF fixture changed while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise GGUFBlackBoxError("the GGUF fixture changed while hashing")
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise GGUFBlackBoxError("the GGUF fixture changed while hashing")
    except OSError as exc:
        raise GGUFBlackBoxError("the GGUF fixture cannot be read safely") from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _validate_fixture(path: Path) -> None:
    try:
        named = path.lstat()
    except OSError as exc:
        raise GGUFBlackBoxError("the pinned GGUF fixture is unavailable") from exc
    if not stat.S_ISREG(named.st_mode):
        raise GGUFBlackBoxError("the GGUF fixture must be a non-symlink regular file")
    if _sha256_file(path) != FIXTURE_SHA256:
        raise GGUFBlackBoxError("the GGUF fixture digest does not match")


def _run_captured(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[int, bytes, bytes]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
            )
        except OSError as exc:
            raise GGUFBlackBoxError(
                "the container engine could not be started"
            ) from exc
        try:
            status = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise GGUFBlackBoxError(
                "the container command exceeded its time limit"
            ) from exc
        stdout.seek(0)
        stderr.seek(0)
        stdout_bytes = stdout.read(stdout_limit + 1)
        stderr_bytes = stderr.read(stderr_limit + 1)
    if len(stdout_bytes) > stdout_limit or len(stderr_bytes) > stderr_limit:
        raise GGUFBlackBoxError("the container command exceeded its output limit")
    return status, stdout_bytes, stderr_bytes


def _inspect_image(engine: str, image: str) -> str:
    status, stdout, _stderr = _run_captured(
        (engine, "image", "inspect", image),
        timeout_seconds=30,
        stdout_limit=_MAX_INSPECT_BYTES,
        stderr_limit=64 * 1024,
    )
    if status != 0:
        raise GGUFBlackBoxError("the GGUF runtime image could not be inspected")
    try:
        decoded = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GGUFBlackBoxError("the GGUF runtime image inspection is invalid") from exc
    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
        or not isinstance(decoded[0], dict)
    ):
        raise GGUFBlackBoxError("the GGUF runtime image inspection is ambiguous")
    inspection = decoded[0]
    image_digest = inspection.get("Id")
    config = inspection.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if (
        not isinstance(image_digest, str)
        or _IMAGE_DIGEST.fullmatch(image_digest) is None
    ):
        raise GGUFBlackBoxError("the GGUF runtime image has no canonical digest")
    expected_labels = {
        "dev.invarlock.runtime-provider": "llama_cpp",
        "dev.invarlock.llama-cpp.source-commit": LLAMA_CPP_SOURCE_COMMIT,
        "dev.invarlock.llama-cpp.source-sha256": LLAMA_CPP_SOURCE_SHA256,
    }
    if not isinstance(labels, dict) or any(
        labels.get(name) != value for name, value in expected_labels.items()
    ):
        raise GGUFBlackBoxError("the GGUF runtime image labels do not match the pin")
    return image_digest


def _container_command(
    *,
    engine: str,
    image_digest: str,
    model_path: Path,
    script_path: Path,
) -> tuple[str, ...]:
    if _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise GGUFBlackBoxError("the GGUF runtime image digest is invalid")
    support_path = script_path.with_name("_gguf_runtime_blackbox_cli.py")
    if any(
        "," in str(path) or any(ord(character) < 32 for character in str(path))
        for path in (model_path, script_path, support_path)
    ):
        raise GGUFBlackBoxError("a container mount source is not safely representable")
    return (
        engine,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777",
        "--mount",
        f"type=bind,src={model_path},dst={_CONTAINER_MODEL},readonly",
        "--mount",
        f"type=bind,src={script_path},dst={_CONTAINER_SCRIPT},readonly",
        "--mount",
        f"type=bind,src={support_path},dst={_CONTAINER_SUPPORT},readonly",
        "-e",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "-e",
        "INVARLOCK_ALLOW_HOST_EXECUTION=0",
        "-e",
        "INVARLOCK_ALLOW_NETWORK=0",
        "-e",
        "INVARLOCK_ALLOW_REMOTE_CODE=0",
        "-e",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0",
        "-e",
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE=0",
        "-e",
        f"INVARLOCK_RUNTIME_IMAGE_DIGEST={image_digest}",
        "-e",
        f"INVARLOCK_RUNTIME_IMAGE=invarlock-runtime@{image_digest}",
        "--entrypoint",
        "python",
        image_digest,
        _CONTAINER_SCRIPT,
        "--inside-container",
        "--image-digest",
        image_digest,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _expected_observation(*, schedule_sha256: str) -> dict[str, object]:
    expected_record = {
        "error_code": None,
        "input_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "logprob_sum": None,
        "output_sha256": hashlib.sha256(EXPECTED_OUTPUT.encode("utf-8")).hexdigest(),
        "output_text": EXPECTED_OUTPUT,
        "record_id": RECORD_ID,
        "status": "ok",
        "token_count": None,
        "utf8_byte_count": None,
    }
    return {
        "aggregate_source_sha256": AGGREGATE_SOURCE_SHA256,
        "artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "format_version": "invarlock/runtime-scoring-observation-v1",
        "provider_name": "llama_cpp",
        "records": [expected_record],
        "schedule_sha256": schedule_sha256,
    }


def _expected_artifact() -> dict[str, object]:
    return {
        "artifact_format": "gguf",
        "artifact_name": f"gguf-sha256-{FIXTURE_SHA256}.gguf",
        "byte_length": FIXTURE_BYTE_LENGTH,
        "format_version": "invarlock/model-artifact-identity-v1",
        "gguf_metadata_sha256": FIXTURE_METADATA_SHA256,
        "sha256": FIXTURE_SHA256,
        "tensor_inventory_sha256": FIXTURE_TENSOR_INVENTORY_SHA256,
        "tokenizer_metadata_sha256": FIXTURE_TOKENIZER_METADATA_SHA256,
    }


def _validate_provider_receipt(
    receipt: dict[str, object],
    *,
    image_digest: str,
    batch_size: int,
    observation_sha256: str,
) -> None:
    backend = receipt.get("backend")
    capabilities = receipt.get("capabilities")
    device = receipt.get("device")
    plugin = receipt.get("plugin")
    if receipt.get("format_version") != "invarlock/runtime-provider-receipt-v1":
        raise GGUFBlackBoxError("the GGUF receipt format does not match")
    if receipt.get("artifact_identity") != _expected_artifact():
        raise GGUFBlackBoxError("the GGUF receipt artifact does not match")
    if receipt.get("outer_image_digest") != image_digest:
        raise GGUFBlackBoxError("the GGUF receipt image digest does not match")
    if receipt.get("scoring_observation_sha256") != observation_sha256:
        raise GGUFBlackBoxError("the GGUF receipt observation binding does not match")
    if receipt.get("execution_settings") != {
        "allow_network": False,
        "batch_size": batch_size,
        "context_length": 256,
        "max_output_tokens": 16,
        "seed": 7,
        "timeout_seconds": 120,
    }:
        raise GGUFBlackBoxError("the GGUF receipt settings do not match")
    if (
        not isinstance(backend, dict)
        or backend.get("name") != "llama.cpp"
        or backend.get("source_sha256") != LLAMA_CPP_SOURCE_SHA256
        or backend.get("build_sha256") is not None
        or not isinstance(backend.get("binary_sha256"), str)
        or _SHA256.fullmatch(backend["binary_sha256"]) is None
        or not isinstance(backend.get("version"), str)
        or _VERSION.fullmatch(backend["version"]) is None
    ):
        raise GGUFBlackBoxError("the GGUF receipt backend does not match")
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("provider_name") != "llama_cpp"
        or capabilities.get("metrics") != ["exact_match"]
        or capabilities.get("tasks") != ["text_causal"]
        or capabilities.get("supported_claim_sets")
        != ["invarlock-runtime-behavioral-regression-v1"]
    ):
        raise GGUFBlackBoxError("the GGUF receipt capabilities do not match")
    if not isinstance(device, dict) or device.get("device_kind") != "cpu":
        raise GGUFBlackBoxError("the GGUF receipt device does not match")
    if (
        not isinstance(plugin, dict)
        or plugin.get("name") != "llama_cpp"
        or plugin.get("distribution") != "invarlock"
        or plugin.get("provider_abi") != "1"
    ):
        raise GGUFBlackBoxError("the GGUF receipt plugin does not match")


def _validate_cli_journey_summary(value: object, *, image_digest: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "artifact_identity_sha256",
        "binding_sha256",
        "execution_settings_sha256",
        "format_version",
        "observation_sha256",
        "observation",
        "policy_digest",
        "policy_file_sha256",
        "portable_artifact_count",
        "provider_receipt_sha256",
        "provider_receipt",
        "schedule_sha256",
        "verification",
    }:
        raise GGUFBlackBoxError("the GGUF CLI journey summary is incomplete")
    if value.get("format_version") != CLI_JOURNEY_FORMAT:
        raise GGUFBlackBoxError("the GGUF CLI journey format does not match")
    if value.get("artifact_identity_sha256") != ARTIFACT_IDENTITY_SHA256:
        raise GGUFBlackBoxError("the GGUF CLI artifact binding does not match")
    if value.get("execution_settings_sha256") != CLI_EXECUTION_SETTINGS_SHA256:
        raise GGUFBlackBoxError("the GGUF CLI settings binding does not match")
    if value.get("schedule_sha256") != CLI_SCHEDULE_SHA256:
        raise GGUFBlackBoxError("the GGUF CLI schedule binding does not match")
    if value.get("observation_sha256") != CLI_SCORING_OBSERVATION_SHA256:
        raise GGUFBlackBoxError("the GGUF CLI observation binding does not match")
    observation = value.get("observation")
    if (
        observation != _expected_observation(schedule_sha256=CLI_SCHEDULE_SHA256)
        or hashlib.sha256(_canonical_json(observation)).hexdigest()
        != CLI_SCORING_OBSERVATION_SHA256
    ):
        raise GGUFBlackBoxError("the GGUF CLI known-answer observation does not match")
    for name in (
        "binding_sha256",
        "policy_file_sha256",
        "provider_receipt_sha256",
    ):
        digest = value.get(name)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise GGUFBlackBoxError(f"the GGUF CLI {name} is invalid")
    policy_digest = value.get("policy_digest")
    if (
        not isinstance(policy_digest, str)
        or _POLICY_DIGEST.fullmatch(policy_digest) is None
    ):
        raise GGUFBlackBoxError("the GGUF CLI policy digest is invalid")
    provider_receipt = value.get("provider_receipt")
    if not isinstance(provider_receipt, dict):
        raise GGUFBlackBoxError("the GGUF CLI provider receipt is incomplete")
    _validate_provider_receipt(
        provider_receipt,
        image_digest=image_digest,
        batch_size=1,
        observation_sha256=CLI_SCORING_OBSERVATION_SHA256,
    )
    if hashlib.sha256(_canonical_json(provider_receipt)).hexdigest() != value.get(
        "provider_receipt_sha256"
    ):
        raise GGUFBlackBoxError("the GGUF CLI provider receipt digest does not match")
    if value.get("portable_artifact_count") != 17:
        raise GGUFBlackBoxError("the GGUF CLI artifact inventory is incomplete")
    if value.get("verification") != {
        "baseline_score": 1.0,
        "regression": 0.0,
        "subject_score": 1.0,
        "verdict": "pass",
    }:
        raise GGUFBlackBoxError("the GGUF CLI paired verification did not pass")


def _validate_result_payload(payload: bytes, *, image_digest: str) -> bytes:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise GGUFBlackBoxError("the GGUF black-box result framing is invalid")
    canonical = payload[:-1]
    try:
        decoded = json.loads(canonical)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GGUFBlackBoxError("the GGUF black-box result is invalid") from exc
    if not isinstance(decoded, dict) or _canonical_json(decoded) != canonical:
        raise GGUFBlackBoxError("the GGUF black-box result is not canonical JSON")
    if set(decoded) != {
        "cli_journey",
        "fixture",
        "format_version",
        "image_digest",
        "observation",
        "receipt",
    }:
        raise GGUFBlackBoxError("the GGUF black-box result field set does not match")
    fixture = decoded.get("fixture")
    observation = decoded.get("observation")
    receipt = decoded.get("receipt")
    journey = decoded.get("cli_journey")
    expected_fixture = {
        "byte_length": FIXTURE_BYTE_LENGTH,
        "repository": FIXTURE_REPOSITORY,
        "revision": FIXTURE_REVISION,
        "sha256": FIXTURE_SHA256,
    }
    if decoded.get("format_version") != RESULT_FORMAT:
        raise GGUFBlackBoxError("the GGUF black-box result format does not match")
    if decoded.get("image_digest") != image_digest:
        raise GGUFBlackBoxError("the GGUF black-box result image digest does not match")
    if fixture != expected_fixture:
        raise GGUFBlackBoxError("the GGUF black-box fixture identity does not match")
    if not isinstance(observation, dict) or not isinstance(receipt, dict):
        raise GGUFBlackBoxError("the GGUF black-box evidence is incomplete")
    expected_observation = _expected_observation(schedule_sha256=SCHEDULE_SHA256)
    if observation != expected_observation:
        raise GGUFBlackBoxError("the GGUF black-box output does not match the pin")
    if hashlib.sha256(_canonical_json(observation)).hexdigest() != (
        SCORING_OBSERVATION_SHA256
    ):
        raise GGUFBlackBoxError("the GGUF scoring observation digest does not match")
    _validate_provider_receipt(
        receipt,
        image_digest=image_digest,
        batch_size=32,
        observation_sha256=SCORING_OBSERVATION_SHA256,
    )
    _validate_cli_journey_summary(journey, image_digest=image_digest)
    return canonical


def _run_container_once(
    *, engine: str, image_digest: str, model_path: Path, script_path: Path
) -> bytes:
    command = _container_command(
        engine=engine,
        image_digest=image_digest,
        model_path=model_path,
        script_path=script_path,
    )
    status, stdout, _stderr = _run_captured(
        command,
        timeout_seconds=600,
        stdout_limit=_MAX_RESULT_BYTES,
        stderr_limit=256 * 1024,
    )
    if status != 0:
        raise GGUFBlackBoxError("the GGUF black-box container run failed")
    return _validate_result_payload(stdout, image_digest=image_digest)


def _run_host(*, engine: str, image: str, model_path: Path) -> dict[str, object]:
    _validate_fixture(model_path)
    image_digest = _inspect_image(engine, image)
    script_path = Path(__file__).resolve(strict=True)
    first = _run_container_once(
        engine=engine,
        image_digest=image_digest,
        model_path=model_path.resolve(strict=True),
        script_path=script_path,
    )
    _validate_fixture(model_path)
    second = _run_container_once(
        engine=engine,
        image_digest=image_digest,
        model_path=model_path.resolve(strict=True),
        script_path=script_path,
    )
    _validate_fixture(model_path)
    if first != second:
        raise GGUFBlackBoxError(
            "the two GGUF observations and receipts are not byte-identical"
        )
    return {
        "evidence_sha256": hashlib.sha256(first).hexdigest(),
        "fixture_revision": FIXTURE_REVISION,
        "format_version": "invarlock/gguf-runtime-blackbox-summary-v1",
        "image_digest": image_digest,
        "runs": 2,
        "status": "ok",
    }


def _normalized_backend_version() -> str:
    status, stdout, stderr = _run_captured(
        (_CONTAINER_EXECUTABLE, "--version"),
        timeout_seconds=10,
        stdout_limit=16 * 1024,
        stderr_limit=16 * 1024,
    )
    if status != 0:
        raise GGUFBlackBoxError("the pinned llama.cpp version probe failed")
    try:
        decoded = (stdout + b"\n" + stderr).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GGUFBlackBoxError("the pinned llama.cpp version is invalid") from exc
    lines = [" ".join(line.split()) for line in decoded.splitlines() if line.strip()]
    versions = [line for line in lines if line.startswith("version: ")]
    builds = [line for line in lines if line.startswith("built with ")]
    if len(versions) != 1 or len(builds) != 1:
        raise GGUFBlackBoxError("the pinned llama.cpp version is ambiguous")
    version = f"{versions[0]} {builds[0]}"
    if _VERSION.fullmatch(version) is None:
        raise GGUFBlackBoxError("the pinned llama.cpp version does not match")
    return version


def _require_installed_wheel() -> None:
    if os.environ.get("PYTHONPATH"):
        raise GGUFBlackBoxError("the container must not set PYTHONPATH")
    try:
        import invarlock

        distribution = importlib.metadata.distribution("invarlock")
        package_path = Path(invarlock.__file__).resolve(strict=True)
        distribution_root = Path(str(distribution.locate_file(""))).resolve(strict=True)
        package_path.relative_to(distribution_root)
    except (
        ImportError,
        importlib.metadata.PackageNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        raise GGUFBlackBoxError("the installed InvarLock wheel is unavailable") from exc
    if str(package_path).startswith("/opt/invarlock-blackbox/"):
        raise GGUFBlackBoxError("the black-box imported InvarLock from a source mount")


def _inside_provider_result(*, image_digest: str) -> dict[str, object]:
    if _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise GGUFBlackBoxError("the supplied image digest is invalid")
    _require_installed_wheel()
    if os.environ.get("INVARLOCK_RUNTIME_IMAGE_DIGEST") != image_digest:
        raise GGUFBlackBoxError("the container image digest binding does not match")

    from invarlock.core.runtime_provider import (
        EvaluationBatch,
        EvaluationRecord,
        ModelRuntimeSpec,
        RuntimeExecutionContext,
        artifact_identity_sha256,
    )
    from invarlock.runtime_provider_evidence import (
        encode_runtime_provider_receipt,
        encode_scoring_observation,
        runtime_provider_evidence_errors,
    )
    from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity
    from invarlock.runtime_providers.llama_cpp import (
        LlamaCppProvider,
        LlamaCppRuntimeBindings,
    )

    model = Path(_CONTAINER_MODEL)
    _validate_fixture(model)
    identity = read_gguf_artifact_identity(model)
    expected_identity = {
        "artifact_name": f"gguf-sha256-{FIXTURE_SHA256}.gguf",
        "sha256": FIXTURE_SHA256,
        "byte_length": FIXTURE_BYTE_LENGTH,
        "gguf_metadata_sha256": FIXTURE_METADATA_SHA256,
        "tensor_inventory_sha256": FIXTURE_TENSOR_INVENTORY_SHA256,
        "tokenizer_metadata_sha256": FIXTURE_TOKENIZER_METADATA_SHA256,
        "format_version": "invarlock/model-artifact-identity-v1",
        "artifact_format": "gguf",
    }
    if asdict(identity) != expected_identity:
        raise GGUFBlackBoxError("the parsed GGUF identity does not match the pin")
    executable = Path(_CONTAINER_EXECUTABLE)
    source = Path(_CONTAINER_SOURCE)
    if _sha256_file_unbounded(source) != LLAMA_CPP_SOURCE_SHA256:
        raise GGUFBlackBoxError("the llama.cpp source archive does not match the pin")
    backend_binary_sha256 = _sha256_file_unbounded(executable)
    backend_version = _normalized_backend_version()
    spec = ModelRuntimeSpec(
        provider_name="llama_cpp",
        model_id=identity.artifact_name,
        settings={
            "artifact_byte_length": FIXTURE_BYTE_LENGTH,
            "artifact_sha256": FIXTURE_SHA256,
            "backend_binary_sha256": backend_binary_sha256,
            "backend_source_sha256": LLAMA_CPP_SOURCE_SHA256,
            "backend_version": backend_version,
            "batch_size": 32,
            "context_length": 256,
            "gguf_metadata_sha256": FIXTURE_METADATA_SHA256,
            "max_output_tokens": 16,
            "seed": 7,
            "tensor_inventory_sha256": FIXTURE_TENSOR_INVENTORY_SHA256,
            "timeout_seconds": 120,
            "tokenizer_metadata_sha256": FIXTURE_TOKENIZER_METADATA_SHA256,
        },
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=image_digest,
        device_kind="cpu",
        artifact_identity_sha256=artifact_identity_sha256(identity),
        native_model=LlamaCppRuntimeBindings(
            gguf_path=model,
            executable_path=executable,
            source_archive_path=source,
        ),
    )
    record = EvaluationRecord(
        record_id=RECORD_ID,
        input_text=PROMPT,
        input_sha256=hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        expected_output=EXPECTED_OUTPUT,
    )
    provider = LlamaCppProvider()
    session = provider.open(spec, context)
    try:
        observation = session.score(
            EvaluationBatch(schedule_sha256=SCHEDULE_SHA256, records=(record,))
        )
        receipt = session.runtime_receipt()
    finally:
        session.close()
    if (
        len(observation.records) != 1
        or observation.records[0].output_text != EXPECTED_OUTPUT
    ):
        raise GGUFBlackBoxError("the GGUF black-box output does not match the pin")
    observation_bytes = encode_scoring_observation(observation)
    errors = runtime_provider_evidence_errors(
        artifact_identity=identity,
        scoring_observation=observation,
        receipt=receipt,
        scoring_observation_bytes=observation_bytes,
        expected_outer_image_digest=image_digest,
    )
    if errors:
        raise GGUFBlackBoxError("the GGUF provider evidence is not cross-bound")
    observation_object = json.loads(observation_bytes)
    receipt_object = json.loads(encode_runtime_provider_receipt(receipt))
    return {
        "fixture": {
            "byte_length": FIXTURE_BYTE_LENGTH,
            "repository": FIXTURE_REPOSITORY,
            "revision": FIXTURE_REVISION,
            "sha256": FIXTURE_SHA256,
        },
        "format_version": RESULT_FORMAT,
        "image_digest": image_digest,
        "observation": observation_object,
        "receipt": receipt_object,
    }


def _write_canonical_new(path: Path, value: object) -> None:
    _cli_support.write_canonical_new(globals(), path, value)


def _run_installed_cli(
    arguments: Sequence[str],
    *,
    expected_format: str,
    expect_success: bool = True,
    timeout_seconds: int = 240,
) -> dict[str, object]:
    return _cli_support.run_installed_cli(
        globals(),
        arguments,
        expected_format=expected_format,
        expect_success=expect_success,
        timeout_seconds=timeout_seconds,
    )


def _path_free_strings(value: object) -> bool:
    return _cli_support.path_free_strings(globals(), value)


def _portable_json(
    path: Path, *, manifest: bool = False
) -> tuple[bytes, dict[str, object]]:
    return _cli_support.portable_json(globals(), path, manifest=manifest)


def _native_cli_arguments(*, image_digest: str, settings: Path) -> tuple[str, ...]:
    return _cli_support.native_cli_arguments(
        globals(), image_digest=image_digest, settings=settings
    )


def _expected_side_bindings(side: Path) -> dict[str, str]:
    return _cli_support.expected_side_bindings(globals(), side)


def _validate_cli_side(
    side: Path,
    *,
    role: str,
    image_digest: str,
) -> dict[str, object]:
    return _cli_support.validate_cli_side(
        globals(),
        side,
        role=role,
        image_digest=image_digest,
    )


def _inside_cli_journey(*, image_digest: str) -> dict[str, object]:
    return _cli_support.inside_cli_journey(globals(), image_digest=image_digest)


def _inside_result(*, image_digest: str) -> dict[str, object]:
    provider_result = _inside_provider_result(image_digest=image_digest)
    provider_result["cli_journey"] = _inside_cli_journey(image_digest=image_digest)
    return provider_result


def _sha256_file_unbounded(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise GGUFBlackBoxError("a pinned runtime file cannot be read") from exc
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the optional pinned GGUF release black-box twice."
    )
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--image", default="invarlock-runtime:gguf-local")
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--inside-container", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--image-digest", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.inside_container:
            if arguments.model is not None or arguments.image_digest is None:
                raise GGUFBlackBoxError("the inside-container invocation is invalid")
            result = _inside_result(image_digest=arguments.image_digest)
        else:
            if arguments.model is None or arguments.image_digest is not None:
                raise GGUFBlackBoxError("--model is required for the host invocation")
            result = _run_host(
                engine=arguments.engine,
                image=arguments.image,
                model_path=arguments.model,
            )
    except GGUFBlackBoxError as exc:
        print(f"GGUF black-box failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
