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
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

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
_MAX_TOKENIZER_CONTRACT_BYTES = 128 * 1024 * 1024
_INFO_TIMEOUT_SECONDS = 120
_IO_CHUNK_BYTES = 64 * 1024
_FICLONE = 0x40049409
_ENGINE_NAME = re.compile(r"^rank(0|[1-9][0-9]*)\.engine$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMPUTE_CAPABILITY = re.compile(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)$")
_CUDA_RUNTIME_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_fcntl = importlib.import_module("fcntl") if os.name == "posix" else None


class TensorRTLLMExecutionError(RuntimeError):
    """Raised when authenticated TensorRT-LLM execution cannot continue."""


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


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _hash_descriptor(descriptor: int, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = expected_size
    digest = hashlib.sha256()
    while remaining:
        chunk = os.read(descriptor, min(remaining, _IO_CHUNK_BYTES))
        if not chunk:
            raise TensorRTLLMExecutionError("pinned file changed while being hashed")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise TensorRTLLMExecutionError("pinned file grew while being hashed")
    return digest.hexdigest()


@dataclass
class _PinnedFile:
    path: Path = field(repr=False)
    descriptor: int = field(repr=False)
    parent_descriptor: int = field(repr=False)
    basename: str
    initial_stat: os.stat_result = field(repr=False)
    sha256: str
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_sha256: str,
        require_executable: bool,
        max_bytes: int | None = None,
    ) -> _PinnedFile:
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise TensorRTLLMExecutionError(
                "secure pinned-file execution requires POSIX nofollow support"
            )
        try:
            absolute = Path(os.path.abspath(os.fspath(path)))
        except (TypeError, ValueError, OSError) as exc:
            raise TensorRTLLMExecutionError("pinned file path is invalid") from exc
        flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            parent_descriptor = os.open(absolute.anchor, flags)
        except OSError as exc:
            raise TensorRTLLMExecutionError(
                "pinned file root cannot be opened"
            ) from exc
        try:
            for component in absolute.parts[1:-1]:
                try:
                    next_descriptor = os.open(
                        component, flags, dir_fd=parent_descriptor
                    )
                except OSError as exc:
                    raise TensorRTLLMExecutionError(
                        "pinned file path contains a symlink or inaccessible directory"
                    ) from exc
                os.close(parent_descriptor)
                parent_descriptor = next_descriptor
            try:
                named = os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                descriptor = os.open(
                    absolute.name,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise TensorRTLLMExecutionError(
                    "pinned file cannot be opened without following symlinks"
                ) from exc
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or _stat_identity(
                    named
                ) != _stat_identity(opened):
                    raise TensorRTLLMExecutionError(
                        "pinned file changed while being opened"
                    )
                if max_bytes is not None and opened.st_size > max_bytes:
                    raise TensorRTLLMExecutionError(
                        "pinned file exceeds the configured size bound"
                    )
                if require_executable and opened.st_mode & 0o111 == 0:
                    raise TensorRTLLMExecutionError(
                        "pinned TensorRT-LLM runner is not executable"
                    )
                observed_sha256 = _hash_descriptor(descriptor, opened.st_size)
                if observed_sha256 != expected_sha256:
                    raise TensorRTLLMExecutionError("pinned file digest does not match")
                return cls(
                    path=absolute,
                    descriptor=descriptor,
                    parent_descriptor=parent_descriptor,
                    basename=absolute.name,
                    initial_stat=opened,
                    sha256=observed_sha256,
                )
            except Exception:
                os.close(descriptor)
                raise
        except Exception:
            os.close(parent_descriptor)
            raise

    @property
    def fd_path(self) -> str:
        self._require_open()
        if os.path.isdir("/proc/self/fd"):
            return f"/proc/self/fd/{self.descriptor}"
        raise TensorRTLLMExecutionError(
            "Linux descriptor-backed runner execution is unavailable"
        )

    def _require_open(self) -> None:
        if self._closed:
            raise TensorRTLLMExecutionError("pinned file is closed")

    def recheck(self) -> None:
        self._require_open()
        expected = _stat_identity(self.initial_stat)
        try:
            opened = os.fstat(self.descriptor)
            named = os.stat(
                self.basename,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise TensorRTLLMExecutionError("pinned file became unavailable") from exc
        if _stat_identity(opened) != expected or _stat_identity(named) != expected:
            raise TensorRTLLMExecutionError("pinned file identity changed")
        if _hash_descriptor(self.descriptor, opened.st_size) != self.sha256:
            raise TensorRTLLMExecutionError("pinned file digest changed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.descriptor)
        os.close(self.parent_descriptor)


@dataclass
class _RunDirectory:
    path: Path = field(repr=False)
    descriptor: int = field(repr=False)
    initial_stat: os.stat_result = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(cls) -> _RunDirectory:
        path = Path(tempfile.mkdtemp(prefix="invarlock-tensorrt-llm-"))
        path.chmod(0o700)
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | os.O_DIRECTORY
                | os.O_NOFOLLOW,
            )
        except OSError:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return cls(path=path, descriptor=descriptor, initial_stat=os.fstat(descriptor))

    def recheck(self) -> None:
        if self._closed:
            raise TensorRTLLMExecutionError("isolated runtime directory is closed")
        try:
            opened = os.fstat(self.descriptor)
            named = self.path.lstat()
        except OSError as exc:
            raise TensorRTLLMExecutionError(
                "isolated runtime directory became unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(opened) != _directory_identity(self.initial_stat)
            or _directory_identity(named) != _directory_identity(self.initial_stat)
        ):
            raise TensorRTLLMExecutionError("isolated runtime directory changed")

    def environment(self) -> dict[str, str]:
        self.recheck()
        rendered = str(self.path)
        return {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "DO_NOT_TRACK": "1",
            "FORCE_DETERMINISTIC": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": rendered,
            "INVARLOCK_CONTAINER_EXECUTION": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "NO_COLOR": "1",
            "NO_PROXY": "*",
            "TELEMETRY_DISABLED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "TRTLLM_NO_USAGE_STATS": "1",
            "TMPDIR": rendered,
            "XDG_CACHE_HOME": rendered,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.descriptor)
        shutil.rmtree(self.path, ignore_errors=True)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _close_selector_stream(selector: selectors.BaseSelector, stream: BinaryIO) -> None:
    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    stream.close()


def _run_bounded_process(
    *,
    executable: _PinnedFile,
    arguments: Sequence[str],
    input_bytes: bytes,
    run_directory: _RunDirectory,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[int, bytes, bytes]:
    run_directory.recheck()
    executable.recheck()
    executable_path = executable.fd_path
    try:
        process = subprocess.Popen(
            [executable_path, *arguments],
            executable=executable_path,
            stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=run_directory.path,
            env=run_directory.environment(),
            shell=False,
            close_fds=True,
            pass_fds=(executable.descriptor,),
            start_new_session=True,
            bufsize=0,
        )
    except OSError as exc:
        raise TensorRTLLMExecutionError(
            "descriptor-backed TensorRT-LLM runner execution is unavailable"
        ) from exc
    if (
        (input_bytes and process.stdin is None)
        or process.stdout is None
        or process.stderr is None
    ):
        _kill_process_group(process)
        raise TensorRTLLMExecutionError("TensorRT-LLM runner pipes are unavailable")

    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    input_offset = 0
    try:
        streams = tuple(
            stream
            for stream in (process.stdin, process.stdout, process.stderr)
            if stream is not None
        )
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        if input_bytes:
            assert process.stdin is not None
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TensorRTLLMExecutionError("TensorRT-LLM record timed out")
            events = selector.select(remaining)
            if not events:
                raise TensorRTLLMExecutionError("TensorRT-LLM record timed out")
            for key, _mask in events:
                stream = cast(BinaryIO, key.fileobj)
                if key.data == "stdin":
                    try:
                        written = os.write(
                            stream.fileno(),
                            input_bytes[input_offset : input_offset + _IO_CHUNK_BYTES],
                        )
                    except BrokenPipeError:
                        _close_selector_stream(selector, stream)
                        continue
                    input_offset += written
                    if input_offset == len(input_bytes):
                        _close_selector_stream(selector, stream)
                    continue
                try:
                    chunk = os.read(stream.fileno(), _IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    _close_selector_stream(selector, stream)
                    continue
                target = stdout if key.data == "stdout" else stderr
                target.extend(chunk)
                limit = stdout_limit if key.data == "stdout" else stderr_limit
                if len(target) > limit:
                    raise TensorRTLLMExecutionError(
                        f"TensorRT-LLM runner {key.data} limit exceeded"
                    )
        try:
            status = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise TensorRTLLMExecutionError("TensorRT-LLM record timed out") from exc
        return status, bytes(stdout), bytes(stderr)
    except BaseException:
        _kill_process_group(process)
        raise
    finally:
        selector.close()
        for final_stream in (process.stdin, process.stdout, process.stderr):
            if final_stream is not None and not final_stream.closed:
                final_stream.close()


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TensorRTLLMExecutionError(f"{label} is not UTF-8") from exc

    def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise TensorRTLLMExecutionError(f"{label} contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise TensorRTLLMExecutionError(
            f"{label} contains non-finite JSON number {value!r}"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except TensorRTLLMExecutionError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise TensorRTLLMExecutionError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise TensorRTLLMExecutionError(f"{label} must be a JSON object")
    return value


def _probe_runner(
    executable: _PinnedFile,
    run_directory: _RunDirectory,
    *,
    expected_version: str,
    expected_build_sha256: str,
    expected_compute_capability: str,
) -> RuntimeDeviceFacts:
    status, stdout, stderr = _run_bounded_process(
        executable=executable,
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
    info = _strict_json_object(stdout, label="TensorRT-LLM runner info")
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
        self._tokenizer_source: _PinnedFile | None = None
        self._device: RuntimeDeviceFacts | None = None
        self._engine_snapshot = self._run_directory.path / "engine"
        self._tokenizer_snapshot = self._run_directory.path / "tokenizer.json"
        try:
            self._runner = _PinnedFile.open(
                config.bindings.runner_executable_path,
                expected_sha256=config.runner_binary_sha256,
                require_executable=True,
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

    def _require_open(self) -> _PinnedFile:
        if self._closed or self._runner is None:
            raise RuntimeError("runtime provider session is closed")
        return self._runner

    def _recheck_runtime(self) -> None:
        runner = self._require_open()
        runner.recheck()
        self._run_directory.recheck()
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
        runner = self._require_open()
        status, stdout, stderr = _run_bounded_process(
            executable=runner,
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
                self._recheck_runtime()
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
        self._require_open()
        if self._latest_observation_sha256 is None:
            raise RuntimeError("runtime provider receipt is unavailable before scoring")
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
        self._require_open()
        return None

    def native_model(self) -> object | None:
        self._require_open()
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._tokenizer_source is not None:
            self._tokenizer_source.close()
        if self._runner is not None:
            self._runner.close()
        self._run_directory.close()


__all__ = [
    "TensorRTLLMExecutionError",
    "TensorRTLLMRuntimeBindings",
    "TensorRTLLMSession",
    "TensorRTLLMSessionConfig",
]
