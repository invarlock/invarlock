"""Candidate-image qualification through the real TensorRT-LLM provider path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProviderReceipt,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.runtime_provider_evidence import (
    encode_runtime_provider_receipt,
    encode_scoring_observation,
    runtime_provider_evidence_errors,
)
from invarlock.runtime_providers.tensorrt_llm import TensorRTLLMProvider
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)
from invarlock.runtime_providers.tensorrt_llm_session import (
    TensorRTLLMRuntimeBindings,
)
from invarlock.runtime_security_helpers import (
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
)

_FORMAT = "invarlock/tensorrt-llm-candidate-qualification-v1"
_RUNNER_INFO_FORMAT = "invarlock/tensorrt-llm-runner-info-v1"
_RUNNER_PROTOCOL = "invarlock/tensorrt-llm-runner-v1"
_BACKEND_VERSION = "1.2.1"
_MAX_RUNNER_BYTES = 16 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 128 * 1024 * 1024
_MAX_INFO_BYTES = 64 * 1024
_INFO_TIMEOUT_SECONDS = 30
_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMPUTE_CAPABILITY = re.compile(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)$")
_INFO_KEYS = frozenset(
    {
        "backend_build_sha256",
        "backend_name",
        "backend_version",
        "cuda_compute_capability",
        "cuda_device_name",
        "cuda_driver_version",
        "cuda_runtime_version",
        "device_kind",
        "format_version",
        "protocol_version",
    }
)


class TensorRTLLMCanaryError(RuntimeError):
    """Raised when a candidate cannot pass the real provider journey."""


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise TensorRTLLMCanaryError(f"runner info {name} is invalid")
    return value


def _required_expected_sha256(value: str, *, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise TensorRTLLMCanaryError(f"{label} must be a lowercase sha256 digest")
    return value


def _validate_runner_info(payload: object) -> dict[str, str]:
    if not isinstance(payload, Mapping) or set(payload) != _INFO_KEYS:
        raise TensorRTLLMCanaryError("runner info has an unexpected schema")
    fixed = {
        "backend_name": "TensorRT-LLM",
        "backend_version": _BACKEND_VERSION,
        "device_kind": "cuda",
        "format_version": _RUNNER_INFO_FORMAT,
        "protocol_version": _RUNNER_PROTOCOL,
    }
    for name, expected in fixed.items():
        if payload.get(name) != expected:
            raise TensorRTLLMCanaryError(
                f"runner info {name} does not match the pinned contract"
            )
    normalized = {name: _required_text(payload, name) for name in _INFO_KEYS}
    if _SHA256.fullmatch(normalized["backend_build_sha256"]) is None:
        raise TensorRTLLMCanaryError("runner info backend build digest is invalid")
    if _COMPUTE_CAPABILITY.fullmatch(normalized["cuda_compute_capability"]) is None:
        raise TensorRTLLMCanaryError("runner info compute capability is invalid")
    return normalized


def _raw_runner_info(runner: Path) -> dict[str, str]:
    """Probe candidate facts before the provider pins them as expected values."""

    path = Path(runner)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TensorRTLLMCanaryError("candidate runner is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_RUNNER_BYTES
        or metadata.st_mode & 0o111 == 0
    ):
        raise TensorRTLLMCanaryError(
            "candidate runner must be a bounded executable regular file"
        )

    with (
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        try:
            completed = subprocess.run(  # noqa: S603 - exact candidate path, no shell
                [str(path), "--invarlock-runtime-info-v1"],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=_INFO_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TensorRTLLMCanaryError("candidate runner info probe failed") from exc
        stdout_size = os.fstat(stdout.fileno()).st_size
        stderr_size = os.fstat(stderr.fileno()).st_size
        if completed.returncode != 0:
            raise TensorRTLLMCanaryError(
                f"candidate runner info probe exited with status {completed.returncode}"
            )
        if stderr_size:
            raise TensorRTLLMCanaryError("candidate runner info probe emitted stderr")
        if stdout_size <= 0 or stdout_size > _MAX_INFO_BYTES:
            raise TensorRTLLMCanaryError("candidate runner info output is invalid")
        stdout.seek(0)
        output = stdout.read(_MAX_INFO_BYTES + 1)
    try:
        decoded = parse_json_bytes(output, label="candidate runner info")
    except StrictJsonError as exc:
        raise TensorRTLLMCanaryError(
            "candidate runner info is not strict JSON"
        ) from exc
    return _validate_runner_info(decoded)


def _read_digest(path: Path, *, label: str, max_bytes: int) -> str:
    try:
        payload = read_regular_file_bytes(path, label=label, max_bytes=max_bytes)
    except StrictJsonError as exc:
        raise TensorRTLLMCanaryError(f"{label} cannot be authenticated") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_image_binding() -> str:
    image_digest = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    if _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise TensorRTLLMCanaryError(
            f"{RUNTIME_IMAGE_DIGEST_ENV} must be a canonical image digest"
        )
    image_ref = os.environ.get(RUNTIME_IMAGE_ENV, "")
    repository, separator, embedded_digest = image_ref.rpartition("@")
    if image_ref != image_digest and not (
        repository and separator and embedded_digest == image_digest
    ):
        raise TensorRTLLMCanaryError(
            f"{RUNTIME_IMAGE_ENV} must embed the exact candidate image digest"
        )
    return image_digest


def _build_spec(
    *,
    identity: TensorRTLLMArtifactIdentity,
    runner_info: Mapping[str, str],
    runner_sha256: str,
) -> ModelRuntimeSpec:
    return ModelRuntimeSpec(
        provider_name="tensorrt_llm",
        model_id=identity.bundle_name,
        settings={
            "backend_build_sha256": runner_info["backend_build_sha256"],
            "backend_version": runner_info["backend_version"],
            "batch_size": 1,
            "builder_config_sha256": identity.builder_config_sha256,
            "context_length": 8,
            "engine_bundle_tree_sha256": identity.engine_bundle_tree_sha256,
            "engine_metadata_sha256": identity.engine_metadata_sha256,
            "file_inventory_sha256": identity.file_inventory_sha256,
            "max_output_tokens": 1,
            "runner_binary_sha256": runner_sha256,
            "seed": 0,
            "target_compute_capability": identity.target_compute_capability,
            "timeout_seconds": 300,
            "tokenizer_metadata_sha256": identity.tokenizer_metadata_sha256,
        },
    )


def qualify_candidate(
    *,
    engine_bundle: Path,
    tokenizer_contract: Path,
    runner: Path,
    expected_engine_tree_sha256: str,
    expected_tokenizer_sha256: str,
    expected_output_sha256: str,
) -> dict[str, object]:
    """Execute two matching scores through fresh real provider sessions."""

    expected_engine_tree_sha256 = _required_expected_sha256(
        expected_engine_tree_sha256, label="expected engine tree digest"
    )
    expected_tokenizer_sha256 = _required_expected_sha256(
        expected_tokenizer_sha256, label="expected tokenizer digest"
    )
    expected_output_sha256 = _required_expected_sha256(
        expected_output_sha256, label="expected output digest"
    )
    image_digest = _require_image_binding()
    runner_info = _raw_runner_info(runner)
    runner_sha256 = _read_digest(
        runner, label="candidate runner", max_bytes=_MAX_RUNNER_BYTES
    )
    tokenizer_sha256 = _read_digest(
        tokenizer_contract,
        label="candidate tokenizer contract",
        max_bytes=_MAX_TOKENIZER_BYTES,
    )
    if tokenizer_sha256 != expected_tokenizer_sha256:
        raise TensorRTLLMCanaryError(
            "candidate tokenizer contract does not match the reviewed digest"
        )
    try:
        identity = read_tensorrt_llm_artifact_identity(
            engine_bundle,
            target_compute_capability=runner_info["cuda_compute_capability"],
            tokenizer_metadata_sha256=tokenizer_sha256,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise TensorRTLLMCanaryError(
            "candidate engine identity cannot be authenticated"
        ) from exc
    if identity.engine_bundle_tree_sha256 != expected_engine_tree_sha256:
        raise TensorRTLLMCanaryError(
            "candidate engine bundle does not match the reviewed tree digest"
        )

    spec = _build_spec(
        identity=identity,
        runner_info=runner_info,
        runner_sha256=runner_sha256,
    )
    provider = TensorRTLLMProvider()
    provider.validate_config(spec)
    artifact_sha256 = artifact_identity_sha256(identity)
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=image_digest,
        device_kind="cuda",
        artifact_identity_sha256=artifact_sha256,
        native_model=TensorRTLLMRuntimeBindings(
            engine_bundle_path=engine_bundle,
            tokenizer_contract_path=tokenizer_contract,
            runner_executable_path=runner,
        ),
    )
    input_text = "InvarLock"
    batch = EvaluationBatch(
        schedule_sha256=hashlib.sha256(
            b"invarlock/tensorrt-llm-candidate-schedule-v1\0InvarLock"
        ).hexdigest(),
        records=(
            EvaluationRecord(
                record_id="candidate-1",
                input_text=input_text,
                input_sha256=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
            ),
        ),
    )
    expected_device = {
        "compute_capability": runner_info["cuda_compute_capability"],
        "cuda_runtime_version": runner_info["cuda_runtime_version"],
        "device_name": runner_info["cuda_device_name"],
        "driver_version": runner_info["cuda_driver_version"],
    }

    def score_once() -> tuple[ScoringObservation, RuntimeProviderReceipt, bytes, bytes]:
        session = None
        try:
            session = provider.open(spec, context)
            observation: ScoringObservation = session.score(batch)
            receipt: RuntimeProviderReceipt = session.runtime_receipt()
        finally:
            if session is not None:
                session.close()

        if len(observation.records) != 1 or observation.records[0].status != "ok":
            raise TensorRTLLMCanaryError("candidate provider score did not complete")
        output_text = observation.records[0].output_text
        if output_text is None:
            raise TensorRTLLMCanaryError("candidate provider score has no output")
        if hashlib.sha256(output_text.encode("utf-8")).hexdigest() != (
            expected_output_sha256
        ):
            raise TensorRTLLMCanaryError(
                "candidate provider output does not match the reviewed digest"
            )
        observed_device = {
            "compute_capability": receipt.device.compute_capability,
            "cuda_runtime_version": receipt.device.cuda_runtime_version,
            "device_name": receipt.device.device_name,
            "driver_version": receipt.device.driver_version,
        }
        if observed_device != expected_device:
            raise TensorRTLLMCanaryError(
                "candidate provider device facts changed under the session boundary"
            )
        observation_bytes = encode_scoring_observation(observation)
        receipt_bytes = encode_runtime_provider_receipt(receipt)
        errors = runtime_provider_evidence_errors(
            artifact_identity=identity,
            scoring_observation=observation,
            receipt=receipt,
            scoring_observation_bytes=observation_bytes,
            expected_outer_image_digest=image_digest,
        )
        if errors:
            raise TensorRTLLMCanaryError(
                "candidate provider evidence is inconsistent: " + "; ".join(errors)
            )
        return observation, receipt, observation_bytes, receipt_bytes

    first_observation, first_receipt, first_observation_bytes, first_receipt_bytes = (
        score_once()
    )
    second_observation, _, second_observation_bytes, second_receipt_bytes = score_once()
    if first_observation.records[0].output_text != (
        second_observation.records[0].output_text
    ):
        raise TensorRTLLMCanaryError(
            "candidate provider output is not byte-identical across sessions"
        )
    if first_observation_bytes != second_observation_bytes:
        raise TensorRTLLMCanaryError(
            "candidate provider observation is not deterministic across sessions"
        )
    if first_receipt_bytes != second_receipt_bytes:
        raise TensorRTLLMCanaryError(
            "candidate provider receipt is not deterministic across sessions"
        )
    return {
        "artifact_identity_sha256": artifact_sha256,
        "engine_bundle_tree_sha256": expected_engine_tree_sha256,
        "format_version": _FORMAT,
        "ok": True,
        "output_sha256": expected_output_sha256,
        "scoring_observation_sha256": first_receipt.scoring_observation_sha256,
        "tokenizer_metadata_sha256": expected_tokenizer_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualify a TensorRT-LLM candidate through the real provider path."
    )
    parser.add_argument("--engine-bundle", type=Path, required=True)
    parser.add_argument("--tokenizer-contract", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--expected-engine-tree-sha256", required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--expected-output-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = qualify_candidate(
            engine_bundle=args.engine_bundle,
            tokenizer_contract=args.tokenizer_contract,
            runner=args.runner,
            expected_engine_tree_sha256=args.expected_engine_tree_sha256,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            expected_output_sha256=args.expected_output_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"TensorRT-LLM candidate qualification failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the runtime image
    raise SystemExit(main())


__all__ = ["TensorRTLLMCanaryError", "main", "qualify_candidate"]
