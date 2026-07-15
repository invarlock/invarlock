"""Process-isolated session support for the TensorRT-LLM runtime provider.

This module deliberately does not import TensorRT-LLM, TensorRT, CUDA, torch, or
transformers.  A digest-pinned runner inside the digest-pinned runtime image owns
those imports and exposes a small versioned JSON protocol.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from invarlock.core.api import ModelAdapter
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.runtime_providers._tensorrt_llm_execution import (
    TensorRTLLMExecutionError,
    _ImmutableExecutionBoundary,
    _pin_official_runner,
    _PinnedFile,
    _resolve_vendor_python,
    _run_bounded_process,
    _RunDirectory,
)
from invarlock.runtime_providers._tensorrt_llm_inspection import (
    _MAX_TOKENIZER_CONTRACT_BYTES,
    _open_validated_tensorrt_llm_static_inputs,
    _strict_json_object,
)
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)

_RUNNER_PROTOCOL = "invarlock/tensorrt-llm-runner-v1"
_RUNNER_INFO_FORMAT = "invarlock/tensorrt-llm-runner-info-v1"
_RUNNER_REQUEST_FORMAT = "invarlock/tensorrt-llm-runner-request-v1"
_RUNNER_RESPONSE_FORMAT = "invarlock/tensorrt-llm-runner-response-v1"
_MAX_INPUT_BYTES = 1024 * 1024
_MAX_BATCH_RECORDS = 1024
_MAX_STDOUT_BYTES = 2 * 1024 * 1024
_MAX_STDERR_BYTES = 256 * 1024
_MAX_INFO_BYTES = 16 * 1024
_INFO_TIMEOUT_SECONDS = 120
_IO_CHUNK_BYTES = 64 * 1024
_FICLONE = 0x40049409
_ENGINE_NAME = re.compile(r"^rank(0|[1-9][0-9]*)\.engine$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMPUTE_CAPABILITY = re.compile(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)$")
_CUDA_RUNTIME_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_fcntl = importlib.import_module("fcntl") if os.name == "posix" else None


@dataclass(frozen=True)
class TensorRTLLMRuntimeBindings:
    """Ephemeral paths excluded from public specifications.

    Device facts are observed by the pinned runner from the live CUDA runtime;
    callers cannot supply or override them.
    """

    engine_bundle_path: Path = field(repr=False, compare=False)
    tokenizer_contract_path: Path = field(repr=False, compare=False)
    runner_executable_path: Path = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_bundle_path", Path(self.engine_bundle_path))
        object.__setattr__(
            self, "tokenizer_contract_path", Path(self.tokenizer_contract_path)
        )
        object.__setattr__(
            self, "runner_executable_path", Path(self.runner_executable_path)
        )


@dataclass(frozen=True)
class TensorRTLLMInputInspection:
    """Path-free identities observed from one pinned native runtime."""

    artifact_identity: TensorRTLLMArtifactIdentity
    backend_build_sha256: str
    backend_version: str
    engine_max_batch_size: int
    engine_max_input_len: int
    engine_max_seq_len: int
    runner_binary_sha256: str


def _probe_runner_info_object(
    runner: _PinnedFile,
    vendor_python: _PinnedFile,
    execution_boundary: _ImmutableExecutionBoundary,
    run_directory: _RunDirectory,
) -> dict[str, object]:
    status, stdout, stderr = _run_bounded_process(
        runner=runner,
        vendor_python=vendor_python,
        execution_boundary=execution_boundary,
        arguments=("--invarlock-runtime-info-v1",),
        input_bytes=b"",
        run_directory=run_directory,
        timeout_seconds=_INFO_TIMEOUT_SECONDS,
        stdout_limit=_MAX_INFO_BYTES,
        stderr_limit=_MAX_INFO_BYTES,
    )
    if status != 0:
        raise TensorRTLLMExecutionError(
            f"TensorRT-LLM runner info probe exited with status {status}"
        )
    if stderr:
        raise TensorRTLLMExecutionError("TensorRT-LLM runner info probe emitted stderr")
    return _strict_json_object(stdout, label="TensorRT-LLM runner info")


def _authenticated_official_runner_info(
    runner_path: Path,
) -> tuple[dict[str, object], str]:
    resources: list[_PinnedFile | _ImmutableExecutionBoundary | _RunDirectory] = []
    try:
        runner = _pin_official_runner(runner_path, expected_sha256=None)
        resources.append(runner)
        vendor_python = _resolve_vendor_python()
        resources.append(vendor_python)
        execution_boundary = _ImmutableExecutionBoundary.create(
            runner,
            vendor_python,
        )
        resources.append(execution_boundary)
        run_directory = _RunDirectory.create()
        resources.append(run_directory)
        _require_isolated_network_namespace()
        return (
            _probe_runner_info_object(
                runner,
                vendor_python,
                execution_boundary,
                run_directory,
            ),
            runner.sha256,
        )
    finally:
        cleanup_errors: list[Exception] = []
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception as exc:  # cleanup must continue across resources
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM runner probe cleanup did not complete"
            ) from cleanup_errors[0]


def _validated_inspection_info(info: dict[str, object]) -> dict[str, str]:
    expected_keys = {
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
    if set(info) != expected_keys:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner info has unexpected fields"
        )
    fixed = {
        "backend_name": "TensorRT-LLM",
        "backend_version": "1.2.1",
        "device_kind": "cuda",
        "format_version": _RUNNER_INFO_FORMAT,
        "protocol_version": _RUNNER_PROTOCOL,
    }
    if any(info.get(name) != value for name, value in fixed.items()):
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner identity does not match the pinned contract"
        )
    normalized: dict[str, str] = {}
    for name in expected_keys:
        value = info.get(name)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise TensorRTLLMExecutionError(
                f"TensorRT-LLM runner info {name} is not canonical"
            )
        normalized[name] = value
    if _SHA256.fullmatch(normalized["backend_build_sha256"]) is None:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner build identity is not canonical"
        )
    if _COMPUTE_CAPABILITY.fullmatch(normalized["cuda_compute_capability"]) is None:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner compute capability is not canonical"
        )
    if _CUDA_RUNTIME_VERSION.fullmatch(normalized["cuda_runtime_version"]) is None:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner CUDA runtime version is not canonical"
        )
    return normalized


def inspect_tensorrt_llm_inputs(
    bindings: TensorRTLLMRuntimeBindings,
) -> TensorRTLLMInputInspection:
    """Derive engine, tokenizer, runner, and backend identities in one probe."""

    if not isinstance(bindings, TensorRTLLMRuntimeBindings):
        raise ValueError("tensorrt_llm inspection requires native runtime bindings")
    static_inputs = _open_validated_tensorrt_llm_static_inputs(
        engine_bundle_path=bindings.engine_bundle_path,
        tokenizer_contract_path=bindings.tokenizer_contract_path,
    )
    try:
        info, runner_sha256 = _authenticated_official_runner_info(
            bindings.runner_executable_path
        )
        normalized = _validated_inspection_info(info)
        identity = read_tensorrt_llm_artifact_identity(
            bindings.engine_bundle_path,
            target_compute_capability=normalized["cuda_compute_capability"],
            tokenizer_metadata_sha256=static_inputs.tokenizer_sha256,
        )
        static_inputs.recheck()
        if (
            read_tensorrt_llm_artifact_identity(
                bindings.engine_bundle_path,
                target_compute_capability=normalized["cuda_compute_capability"],
                tokenizer_metadata_sha256=static_inputs.tokenizer_sha256,
            )
            != identity
        ):
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM engine changed during runtime inspection"
            )
        return TensorRTLLMInputInspection(
            artifact_identity=identity,
            backend_build_sha256=normalized["backend_build_sha256"],
            backend_version=normalized["backend_version"],
            engine_max_batch_size=static_inputs.engine_max_batch_size,
            engine_max_input_len=static_inputs.engine_max_input_len,
            engine_max_seq_len=static_inputs.engine_max_seq_len,
            runner_binary_sha256=runner_sha256,
        )
    finally:
        static_inputs.close()


def _probe_runner(
    runner: _PinnedFile,
    vendor_python: _PinnedFile,
    execution_boundary: _ImmutableExecutionBoundary,
    run_directory: _RunDirectory,
    *,
    expected_version: str,
    expected_build_sha256: str,
    expected_compute_capability: str,
) -> RuntimeDeviceFacts:
    info = _probe_runner_info_object(
        runner,
        vendor_python,
        execution_boundary,
        run_directory,
    )
    expected_keys = {
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
    if set(info) != expected_keys:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner info has unexpected fields"
        )
    expected_identity = {
        "backend_build_sha256": expected_build_sha256,
        "backend_name": "TensorRT-LLM",
        "backend_version": expected_version,
        "device_kind": "cuda",
        "format_version": _RUNNER_INFO_FORMAT,
        "protocol_version": _RUNNER_PROTOCOL,
    }
    if any(info.get(name) != value for name, value in expected_identity.items()):
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner identity does not match the pinned configuration"
        )
    build_sha256 = info["backend_build_sha256"]
    compute_capability = info["cuda_compute_capability"]
    device_name = info["cuda_device_name"]
    driver_version = info["cuda_driver_version"]
    runtime_version = info["cuda_runtime_version"]
    if not isinstance(build_sha256, str) or _SHA256.fullmatch(build_sha256) is None:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner build identity is not canonical"
        )
    if (
        not isinstance(compute_capability, str)
        or _COMPUTE_CAPABILITY.fullmatch(compute_capability) is None
        or compute_capability != expected_compute_capability
    ):
        raise TensorRTLLMExecutionError(
            "observed CUDA compute capability does not match the engine target"
        )
    for label, value in (
        ("device name", device_name),
        ("driver version", driver_version),
    ):
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise TensorRTLLMExecutionError(
                f"TensorRT-LLM runner {label} is not canonical"
            )
    if (
        not isinstance(runtime_version, str)
        or _CUDA_RUNTIME_VERSION.fullmatch(runtime_version) is None
    ):
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM runner CUDA runtime version is not canonical"
        )
    assert isinstance(device_name, str)
    assert isinstance(driver_version, str)
    assert isinstance(compute_capability, str)
    return RuntimeDeviceFacts(
        device_kind="cuda",
        device_name=device_name,
        compute_capability=compute_capability,
        driver_version=driver_version,
        cuda_runtime_version=runtime_version,
    )


def _copy_from_descriptor(source: int, destination: Path, byte_length: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    destination_fd = os.open(destination, flags, 0o400)
    try:
        try:
            if _fcntl is None:
                raise OSError("copy-on-write cloning is unavailable")
            _fcntl.ioctl(destination_fd, _FICLONE, source)
        except OSError:
            os.lseek(source, 0, os.SEEK_SET)
            remaining = byte_length
            while remaining:
                chunk = os.read(source, min(remaining, _IO_CHUNK_BYTES))
                if not chunk:
                    raise TensorRTLLMExecutionError(
                        "engine bundle changed while being snapshotted"
                    ) from None
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
                remaining -= len(chunk)
            if os.read(source, 1):
                raise TensorRTLLMExecutionError(
                    "engine bundle changed while being snapshotted"
                ) from None
        os.fchmod(destination_fd, 0o400)
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)


def _snapshot_bundle(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    try:
        entries = sorted(source.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise TensorRTLLMExecutionError("engine bundle cannot be listed") from exc
    if not entries or len(entries) > 257:
        raise TensorRTLLMExecutionError("engine bundle file count is invalid")
    if {entry.name for entry in entries} != {"config.json", "rank0.engine"}:
        raise TensorRTLLMExecutionError(
            "the current TensorRT-LLM provider requires a single-rank engine"
        )
    for entry in entries:
        if entry.name != "config.json" and _ENGINE_NAME.fullmatch(entry.name) is None:
            raise TensorRTLLMExecutionError("engine bundle layout is not closed")
        try:
            descriptor = os.open(
                entry,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise TensorRTLLMExecutionError(
                "engine bundle entry cannot be opened without following symlinks"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise TensorRTLLMExecutionError(
                    "engine bundle contains a non-regular file"
                )
            _copy_from_descriptor(descriptor, destination / entry.name, opened.st_size)
        finally:
            os.close(descriptor)
    destination.chmod(0o500)


def _snapshot_tokenizer(source: _PinnedFile, destination: Path) -> None:
    _copy_from_descriptor(source.descriptor, destination, source.initial_stat.st_size)


def _require_isolated_network_namespace() -> None:
    try:
        ipv4_lines = (
            Path("/proc/net/route")
            .read_text(encoding="ascii", errors="strict")
            .splitlines()
        )
        ipv6_lines = (
            Path("/proc/net/ipv6_route")
            .read_text(encoding="ascii", errors="strict")
            .splitlines()
        )
    except OSError as exc:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM cannot verify the network namespace"
        ) from exc
    ipv4_interfaces = {fields[0] for line in ipv4_lines[1:] if (fields := line.split())}
    ipv6_interfaces = {fields[-1] for line in ipv6_lines if (fields := line.split())}
    if (ipv4_interfaces | ipv6_interfaces) - {"lo"}:
        raise TensorRTLLMExecutionError(
            "TensorRT-LLM requires a network-disabled container"
        )


def _records_sha256(records: tuple[RuntimeScoringRecord, ...]) -> str:
    encoded = json.dumps(
        [asdict(record) for record in records],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_sha256(observation: ScoringObservation) -> str:
    encoded = json.dumps(
        asdict(observation),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TensorRTLLMSessionConfig:
    artifact_identity: TensorRTLLMArtifactIdentity
    backend_build_sha256: str
    backend_version: str
    runner_binary_sha256: str
    execution_settings: RuntimeExecutionSettings
    capabilities: RuntimeProviderCapabilities
    plugin: RuntimeProviderPluginIdentity
    outer_image_digest: str
    bindings: TensorRTLLMRuntimeBindings = field(repr=False, compare=False)


class TensorRTLLMSession:
    """One authenticated TensorRT-LLM engine session."""

    def __init__(self, config: TensorRTLLMSessionConfig) -> None:
        self._config = config
        self._score_lock = threading.Lock()
        self._closed = False
        self._latest_observation_sha256: str | None = None
        self._run_directory = _RunDirectory.create()
        self._runner: _PinnedFile | None = None
        self._vendor_python: _PinnedFile | None = None
        self._execution_boundary: _ImmutableExecutionBoundary | None = None
        self._tokenizer_source: _PinnedFile | None = None
        self._device: RuntimeDeviceFacts | None = None
        self._engine_snapshot = self._run_directory.path / "engine"
        self._tokenizer_snapshot = self._run_directory.path / "tokenizer.json"
        try:
            self._runner = _pin_official_runner(
                config.bindings.runner_executable_path,
                expected_sha256=config.runner_binary_sha256,
            )
            self._vendor_python = _resolve_vendor_python()
            self._execution_boundary = _ImmutableExecutionBoundary.create(
                self._runner,
                self._vendor_python,
            )
            self._tokenizer_source = _PinnedFile.open(
                config.bindings.tokenizer_contract_path,
                expected_sha256=config.artifact_identity.tokenizer_metadata_sha256,
                require_executable=False,
                max_bytes=_MAX_TOKENIZER_CONTRACT_BYTES,
            )
            _snapshot_bundle(
                config.bindings.engine_bundle_path,
                self._engine_snapshot,
            )
            observed = read_tensorrt_llm_artifact_identity(
                self._engine_snapshot,
                target_compute_capability=(
                    config.artifact_identity.target_compute_capability
                ),
                tokenizer_metadata_sha256=(
                    config.artifact_identity.tokenizer_metadata_sha256
                ),
            )
            if observed != config.artifact_identity:
                raise TensorRTLLMExecutionError(
                    "snapshotted engine identity does not match the configuration"
                )
            _snapshot_tokenizer(self._tokenizer_source, self._tokenizer_snapshot)
            if (
                hashlib.sha256(self._tokenizer_snapshot.read_bytes()).hexdigest()
                != config.artifact_identity.tokenizer_metadata_sha256
            ):
                raise TensorRTLLMExecutionError(
                    "snapshotted tokenizer contract digest does not match"
                )
            _require_isolated_network_namespace()
            self._device = _probe_runner(
                self._runner,
                self._vendor_python,
                self._execution_boundary,
                self._run_directory,
                expected_version=config.backend_version,
                expected_build_sha256=config.backend_build_sha256,
                expected_compute_capability=(
                    config.artifact_identity.target_compute_capability
                ),
            )
        except Exception:
            self.close()
            raise
        self._artifact_identity_sha256 = artifact_identity_sha256(
            config.artifact_identity
        )

    def _require_open(
        self,
    ) -> tuple[_PinnedFile, _PinnedFile, _ImmutableExecutionBoundary]:
        if (
            self._closed
            or self._runner is None
            or self._vendor_python is None
            or self._execution_boundary is None
        ):
            raise RuntimeError("runtime provider session is closed")
        return self._runner, self._vendor_python, self._execution_boundary

    def _recheck_runtime(self) -> None:
        runner, vendor_python, execution_boundary = self._require_open()
        self._run_directory.recheck()
        execution_boundary.recheck(runner, vendor_python)
        self._recheck_artifact_snapshots()

    def _recheck_artifact_snapshots(self) -> None:
        observed = read_tensorrt_llm_artifact_identity(
            self._engine_snapshot,
            target_compute_capability=(
                self._config.artifact_identity.target_compute_capability
            ),
            tokenizer_metadata_sha256=(
                self._config.artifact_identity.tokenizer_metadata_sha256
            ),
        )
        if observed != self._config.artifact_identity:
            raise TensorRTLLMExecutionError("snapshotted engine identity changed")
        if (
            hashlib.sha256(self._tokenizer_snapshot.read_bytes()).hexdigest()
            != self._config.artifact_identity.tokenizer_metadata_sha256
        ):
            raise TensorRTLLMExecutionError("snapshotted tokenizer contract changed")

    def _request(self, record: EvaluationRecord) -> bytes:
        request = {
            "engine_bundle": str(self._engine_snapshot),
            "format_version": _RUNNER_REQUEST_FORMAT,
            "input_text": record.input_text,
            "protocol_version": _RUNNER_PROTOCOL,
            "settings": asdict(self._config.execution_settings),
            "tokenizer_contract": str(self._tokenizer_snapshot),
        }
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > _MAX_INPUT_BYTES:
            raise ValueError("TensorRT-LLM record input exceeds the byte limit")
        return encoded

    def _execute_record(self, record: EvaluationRecord) -> str:
        runner, vendor_python, execution_boundary = self._require_open()
        status, stdout, stderr = _run_bounded_process(
            runner=runner,
            vendor_python=vendor_python,
            execution_boundary=execution_boundary,
            arguments=("--invarlock-score-v1",),
            input_bytes=self._request(record),
            run_directory=self._run_directory,
            timeout_seconds=self._config.execution_settings.timeout_seconds,
            stdout_limit=_MAX_STDOUT_BYTES,
            stderr_limit=_MAX_STDERR_BYTES,
        )
        if status != 0:
            raise TensorRTLLMExecutionError(
                f"TensorRT-LLM runner exited with status {status}"
            )
        if stderr:
            raise TensorRTLLMExecutionError("TensorRT-LLM runner emitted stderr")
        response = _strict_json_object(stdout, label="TensorRT-LLM runner response")
        if set(response) != {"format_version", "output_text"}:
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM runner response has unexpected fields"
            )
        if response.get("format_version") != _RUNNER_RESPONSE_FORMAT:
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM runner response format is unsupported"
            )
        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM runner output_text must be a string"
            )
        return output_text

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        if not isinstance(batch, EvaluationBatch):
            raise TypeError("batch must be an EvaluationBatch")
        if len(batch.records) > _MAX_BATCH_RECORDS:
            raise ValueError("TensorRT-LLM batch exceeds the record limit")
        with self._score_lock:
            self._require_open()
            self._latest_observation_sha256 = None
            for record in batch.records:
                expected_input_sha256 = hashlib.sha256(
                    record.input_text.encode("utf-8")
                ).hexdigest()
                if record.input_sha256 != expected_input_sha256:
                    raise ValueError(
                        f"record {record.record_id!r} input_sha256 does not match input_text"
                    )
            self._recheck_runtime()
            scoring_records: list[RuntimeScoringRecord] = []
            try:
                for record in batch.records:
                    output_text = self._execute_record(record)
                    output_bytes = output_text.encode("utf-8")
                    scoring_records.append(
                        RuntimeScoringRecord(
                            record_id=record.record_id,
                            input_sha256=record.input_sha256,
                            status="ok",
                            output_text=output_text,
                            output_sha256=hashlib.sha256(output_bytes).hexdigest(),
                        )
                    )
            finally:
                self._recheck_artifact_snapshots()
            records = tuple(scoring_records)
            expected_pairing = tuple(
                (record.record_id, record.input_sha256) for record in batch.records
            )
            observed_pairing = tuple(
                (record.record_id, record.input_sha256) for record in records
            )
            if observed_pairing != expected_pairing:
                raise TensorRTLLMExecutionError(
                    "TensorRT-LLM output pairing does not match the batch"
                )
            observation = ScoringObservation(
                provider_name=self._config.capabilities.provider_name,
                artifact_identity_sha256=self._artifact_identity_sha256,
                schedule_sha256=batch.schedule_sha256,
                records=records,
                aggregate_source_sha256=_records_sha256(records),
            )
            self._latest_observation_sha256 = _observation_sha256(observation)
            return observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        with self._score_lock:
            self._require_open()
            if self._latest_observation_sha256 is None:
                raise RuntimeError(
                    "runtime provider receipt is unavailable before scoring"
                )
            if self._device is None:
                raise RuntimeError("runtime provider device facts are unavailable")
            return RuntimeProviderReceipt(
                plugin=self._config.plugin,
                backend=RuntimeBackendIdentity(
                    name="TensorRT-LLM",
                    version=self._config.backend_version,
                    source_sha256=None,
                    binary_sha256=self._config.runner_binary_sha256,
                    build_sha256=self._config.backend_build_sha256,
                ),
                capabilities=self._config.capabilities,
                artifact_identity=self._config.artifact_identity,
                execution_settings=self._config.execution_settings,
                device=self._device,
                outer_image_digest=self._config.outer_image_digest,
                scoring_observation_sha256=self._latest_observation_sha256,
            )

    def model_adapter(self) -> ModelAdapter | None:
        with self._score_lock:
            self._require_open()
            return None

    def native_model(self) -> object | None:
        with self._score_lock:
            self._require_open()
            return None

    def close(self) -> None:
        with self._score_lock:
            if self._closed:
                return
            self._closed = True
            cleanup_errors: list[Exception] = []
            resources = (
                self._tokenizer_source,
                self._execution_boundary,
                self._runner,
                self._vendor_python,
                self._run_directory,
            )
            for resource in resources:
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception as exc:  # cleanup must continue across resources
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise TensorRTLLMExecutionError(
                    "TensorRT-LLM session cleanup did not complete"
                ) from cleanup_errors[0]


__all__ = [
    "inspect_tensorrt_llm_inputs",
    "TensorRTLLMExecutionError",
    "TensorRTLLMInputInspection",
    "TensorRTLLMRuntimeBindings",
    "TensorRTLLMSession",
    "TensorRTLLMSessionConfig",
]
