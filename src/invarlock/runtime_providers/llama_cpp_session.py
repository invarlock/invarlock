"""Bounded process-isolated session support for the llama.cpp provider."""

from __future__ import annotations

import hashlib
import json
import os
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
    GGUFArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    artifact_identity_sha256,
)
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity

_MAX_INPUT_BYTES = 1024 * 1024
_MAX_BATCH_RECORDS = 1024
_MAX_STDOUT_BYTES = 1024 * 1024
_MAX_STDERR_BYTES = 256 * 1024
_MAX_VERSION_BYTES = 16 * 1024
_VERSION_TIMEOUT_SECONDS = 5
_IO_CHUNK_BYTES = 64 * 1024


class LlamaCppExecutionError(RuntimeError):
    """Raised when a pinned llama.cpp execution cannot be authenticated."""


@dataclass(frozen=True)
class LlamaCppRuntimeBindings:
    """Ephemeral host bindings excluded from public specs and object reprs."""

    gguf_path: Path = field(repr=False, compare=False)
    executable_path: Path = field(repr=False, compare=False)
    source_archive_path: Path = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gguf_path", Path(self.gguf_path))
        object.__setattr__(self, "executable_path", Path(self.executable_path))
        object.__setattr__(self, "source_archive_path", Path(self.source_archive_path))


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
            raise LlamaCppExecutionError("pinned file changed while being hashed")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise LlamaCppExecutionError("pinned file grew while being hashed")
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
    ) -> _PinnedFile:
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise LlamaCppExecutionError(
                "secure pinned-file execution requires POSIX nofollow support"
            )
        try:
            absolute = Path(os.path.abspath(os.fspath(path)))
        except (TypeError, ValueError, OSError) as exc:
            raise LlamaCppExecutionError("pinned file path is invalid") from exc
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            parent_descriptor = os.open(absolute.anchor, directory_flags)
        except OSError as exc:
            raise LlamaCppExecutionError("pinned file root cannot be opened") from exc
        try:
            for component in absolute.parts[1:-1]:
                try:
                    next_descriptor = os.open(
                        component,
                        directory_flags,
                        dir_fd=parent_descriptor,
                    )
                except OSError as exc:
                    raise LlamaCppExecutionError(
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
                raise LlamaCppExecutionError(
                    "pinned file cannot be opened without following symlinks"
                ) from exc
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or _stat_identity(
                    named
                ) != _stat_identity(opened):
                    raise LlamaCppExecutionError(
                        "pinned file changed while being opened"
                    )
                if require_executable and opened.st_mode & 0o111 == 0:
                    raise LlamaCppExecutionError(
                        "pinned llama.cpp binary is not executable"
                    )
                observed_sha256 = _hash_descriptor(descriptor, opened.st_size)
                if observed_sha256 != expected_sha256:
                    raise LlamaCppExecutionError("pinned file digest does not match")
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
        for root in ("/proc/self/fd", "/dev/fd"):
            if os.path.isdir(root):
                return f"{root}/{self.descriptor}"
        raise LlamaCppExecutionError("descriptor-backed execution path is unavailable")

    def _require_open(self) -> None:
        if self._closed:
            raise LlamaCppExecutionError("pinned file is closed")

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
            raise LlamaCppExecutionError("pinned file became unavailable") from exc
        if _stat_identity(opened) != expected or _stat_identity(named) != expected:
            raise LlamaCppExecutionError("pinned file identity changed")
        if _hash_descriptor(self.descriptor, opened.st_size) != self.sha256:
            raise LlamaCppExecutionError("pinned file digest changed")

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
        path = Path(tempfile.mkdtemp(prefix="invarlock-llama-cpp-"))
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
            raise LlamaCppExecutionError("isolated runtime directory is closed")
        try:
            opened = os.fstat(self.descriptor)
            named = self.path.lstat()
        except OSError as exc:
            raise LlamaCppExecutionError(
                "isolated runtime directory became unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(opened) != _directory_identity(self.initial_stat)
            or _directory_identity(named) != _directory_identity(self.initial_stat)
        ):
            raise LlamaCppExecutionError("isolated runtime directory changed")

    def environment(self) -> dict[str, str]:
        self.recheck()
        rendered = str(self.path)
        return {
            "HOME": rendered,
            "LANG": "C",
            "LC_ALL": "C",
            "NO_COLOR": "1",
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
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def _run_bounded_process(
    *,
    executable: _PinnedFile,
    arguments: Sequence[str],
    input_bytes: bytes,
    pass_fds: tuple[int, ...],
    run_directory: _RunDirectory,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[int, bytes, bytes]:
    run_directory.recheck()
    executable.recheck()
    executable_path = executable.fd_path
    inherited_fds = tuple(sorted({executable.descriptor, *pass_fds}))
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
            pass_fds=inherited_fds,
            start_new_session=True,
            bufsize=0,
        )
    except OSError as exc:
        raise LlamaCppExecutionError(
            "descriptor-backed llama.cpp execution is unavailable"
        ) from exc
    if (
        (input_bytes and process.stdin is None)
        or process.stdout is None
        or process.stderr is None
    ):
        _kill_process_group(process)
        raise LlamaCppExecutionError("llama.cpp pipes could not be established")

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
                raise LlamaCppExecutionError("llama.cpp record timed out")
            events = selector.select(remaining)
            if not events:
                raise LlamaCppExecutionError("llama.cpp record timed out")
            for key, _mask in events:
                selected_stream = cast(BinaryIO, key.fileobj)
                if key.data == "stdin":
                    try:
                        written = os.write(
                            selected_stream.fileno(),
                            input_bytes[input_offset : input_offset + _IO_CHUNK_BYTES],
                        )
                    except BrokenPipeError:
                        _close_selector_stream(selector, selected_stream)
                        continue
                    input_offset += written
                    if input_offset == len(input_bytes):
                        _close_selector_stream(selector, selected_stream)
                    continue

                try:
                    chunk = os.read(selected_stream.fileno(), _IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    _close_selector_stream(selector, selected_stream)
                    continue
                target = stdout if key.data == "stdout" else stderr
                target.extend(chunk)
                limit = stdout_limit if key.data == "stdout" else stderr_limit
                if len(target) > limit:
                    raise LlamaCppExecutionError(f"llama.cpp {key.data} limit exceeded")

        try:
            return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise LlamaCppExecutionError("llama.cpp record timed out") from exc
        return return_code, bytes(stdout), bytes(stderr)
    except BaseException:
        _kill_process_group(process)
        raise
    finally:
        selector.close()
        for final_stream in (process.stdin, process.stdout, process.stderr):
            if final_stream is not None and not final_stream.closed:
                final_stream.close()


def _normalize_version_output(stdout: bytes, stderr: bytes) -> str:
    try:
        decoded = (stdout + b"\n" + stderr).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LlamaCppExecutionError("llama.cpp version output is not UTF-8") from exc
    lines = [" ".join(line.split()) for line in decoded.splitlines() if line.strip()]
    version_lines = [line for line in lines if line.startswith("version: ")]
    compiler_lines = [line for line in lines if line.startswith("built with ")]
    if len(version_lines) != 1 or len(compiler_lines) != 1:
        raise LlamaCppExecutionError(
            "llama.cpp version output lacks exact version/build lines"
        )
    return f"{version_lines[0]} {compiler_lines[0]}"


def probe_llama_cpp_version(
    executable: _PinnedFile,
    run_directory: _RunDirectory,
) -> str:
    status, stdout, stderr = _run_bounded_process(
        executable=executable,
        arguments=("--version",),
        input_bytes=b"",
        pass_fds=(executable.descriptor,),
        run_directory=run_directory,
        timeout_seconds=_VERSION_TIMEOUT_SECONDS,
        stdout_limit=_MAX_VERSION_BYTES,
        stderr_limit=_MAX_VERSION_BYTES,
    )
    if status != 0:
        raise LlamaCppExecutionError(
            f"llama.cpp version probe exited with status {status}"
        )
    return _normalize_version_output(stdout, stderr)


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


def _extract_generated_output(output: bytes) -> bytes:
    """Remove only llama-completion's pinned non-interactive framing.

    b10015 writes the generated token stream directly, followed by two newlines.
    EOG is disabled so its human-readable marker cannot be confused with model
    output. The parser removes one terminal framing instance and otherwise
    leaves generated bytes untouched; prompt bytes are never echoed because
    ``--no-display-prompt`` is mandatory below.
    """

    final_newlines = b"\n\n"
    if not output.endswith(final_newlines):
        raise LlamaCppExecutionError(
            "llama-completion output lacks the pinned final framing"
        )
    return output[: -len(final_newlines)]


@dataclass(frozen=True)
class LlamaCppSessionConfig:
    artifact_identity: GGUFArtifactIdentity
    backend_binary_sha256: str
    backend_source_sha256: str
    backend_version: str
    execution_settings: RuntimeExecutionSettings
    capabilities: RuntimeProviderCapabilities
    plugin: RuntimeProviderPluginIdentity
    device: RuntimeDeviceFacts
    outer_image_digest: str | None
    bindings: LlamaCppRuntimeBindings = field(repr=False, compare=False)


class LlamaCppSession:
    """One authenticated GGUF/llama-completion execution session."""

    def __init__(self, config: LlamaCppSessionConfig) -> None:
        self._config = config
        self._artifact_identity_sha256 = artifact_identity_sha256(
            config.artifact_identity
        )
        self._run_directory = _RunDirectory.create()
        self._executable: _PinnedFile | None = None
        self._model: _PinnedFile | None = None
        self._source_archive: _PinnedFile | None = None
        self._closed = False
        self._latest_observation_sha256: str | None = None
        self._score_lock = threading.Lock()
        try:
            self._executable = _PinnedFile.open(
                config.bindings.executable_path,
                expected_sha256=config.backend_binary_sha256,
                require_executable=True,
            )
            self._model = _PinnedFile.open(
                config.bindings.gguf_path,
                expected_sha256=config.artifact_identity.sha256,
                require_executable=False,
            )
            self._source_archive = _PinnedFile.open(
                config.bindings.source_archive_path,
                expected_sha256=config.backend_source_sha256,
                require_executable=False,
            )
            observed_version = probe_llama_cpp_version(
                self._executable, self._run_directory
            )
            if observed_version != config.backend_version:
                raise LlamaCppExecutionError(
                    "llama.cpp observed version does not match the pinned version"
                )
            self._recheck_runtime()
        except Exception:
            self.close()
            raise

    def _require_open(self) -> tuple[_PinnedFile, _PinnedFile, _PinnedFile]:
        if (
            self._closed
            or self._executable is None
            or self._model is None
            or self._source_archive is None
        ):
            raise RuntimeError("llama.cpp runtime provider session is closed")
        return self._executable, self._model, self._source_archive

    def _recheck_runtime(self) -> None:
        executable, model, source_archive = self._require_open()
        observed_identity = read_gguf_artifact_identity(self._config.bindings.gguf_path)
        if observed_identity != self._config.artifact_identity:
            raise LlamaCppExecutionError(
                "GGUF artifact identity does not match the authenticated identity"
            )
        model.recheck()
        executable.recheck()
        source_archive.recheck()
        self._run_directory.recheck()

    def _arguments(self, model_fd_path: str) -> tuple[str, ...]:
        settings = self._config.execution_settings
        return (
            "--model",
            model_fd_path,
            "--file",
            "/dev/stdin",
            "--seed",
            str(settings.seed),
            "--ctx-size",
            str(settings.context_length),
            "--batch-size",
            str(settings.batch_size),
            "--ubatch-size",
            str(min(settings.batch_size, 512)),
            "--n-predict",
            str(settings.max_output_tokens),
            "--threads",
            "1",
            "--threads-batch",
            "1",
            "--temp",
            "0",
            "--device",
            "none",
            "--fit",
            "off",
            "--no-conversation",
            "--no-display-prompt",
            "--ignore-eos",
            "--no-warmup",
            "--no-context-shift",
            "--no-perf",
            "--no-escape",
            "--verbosity",
            "0",
            "--offline",
        )

    def _execute_record(self, record: EvaluationRecord) -> bytes:
        executable, model, _source_archive = self._require_open()
        input_bytes = record.input_text.encode("utf-8")
        if len(input_bytes) > _MAX_INPUT_BYTES:
            raise ValueError("llama.cpp record input exceeds the byte limit")
        status, stdout, stderr = _run_bounded_process(
            executable=executable,
            arguments=self._arguments(model.fd_path),
            input_bytes=input_bytes,
            pass_fds=(executable.descriptor, model.descriptor),
            run_directory=self._run_directory,
            timeout_seconds=self._config.execution_settings.timeout_seconds,
            stdout_limit=_MAX_STDOUT_BYTES,
            stderr_limit=_MAX_STDERR_BYTES,
        )
        if status != 0:
            raise LlamaCppExecutionError(
                f"llama-completion record exited with status {status}"
            )
        if stderr:
            raise LlamaCppExecutionError(
                "llama-completion emitted unexpected stderr output"
            )
        return _extract_generated_output(stdout)

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        """Score an exact ordered batch using one bounded child per record."""

        if not isinstance(batch, EvaluationBatch):
            raise TypeError("batch must be an EvaluationBatch")
        if len(batch.records) > _MAX_BATCH_RECORDS:
            raise ValueError("llama.cpp batch exceeds the record limit")
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
                    output_bytes = self._execute_record(record)
                    try:
                        output_text = output_bytes.decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        raise LlamaCppExecutionError(
                            "llama.cpp output is not UTF-8"
                        ) from exc
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
                raise LlamaCppExecutionError(
                    "llama.cpp scoring output pairing does not match the batch"
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
        """Return provenance bound to the latest complete scoring observation."""

        self._require_open()
        if self._latest_observation_sha256 is None:
            raise RuntimeError("runtime provider receipt is unavailable before scoring")
        return RuntimeProviderReceipt(
            plugin=self._config.plugin,
            backend=RuntimeBackendIdentity(
                name="llama.cpp",
                version=self._config.backend_version,
                source_sha256=self._config.backend_source_sha256,
                binary_sha256=self._config.backend_binary_sha256,
                build_sha256=None,
            ),
            capabilities=self._config.capabilities,
            artifact_identity=self._config.artifact_identity,
            execution_settings=self._config.execution_settings,
            device=self._config.device,
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
        if self._model is not None:
            self._model.close()
        if self._source_archive is not None:
            self._source_archive.close()
        if self._executable is not None:
            self._executable.close()
        self._run_directory.close()


__all__ = [
    "LlamaCppExecutionError",
    "LlamaCppRuntimeBindings",
    "LlamaCppSession",
    "LlamaCppSessionConfig",
]
