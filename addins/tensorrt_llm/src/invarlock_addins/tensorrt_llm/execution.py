"""Immutable process-execution boundary for the TensorRT-LLM add-in.

This internal module owns file pinning, read-only image-root validation, process
privilege checks, private run-directory lifecycle, and bounded subprocess I/O.
It deliberately does not import TensorRT-LLM, TensorRT, CUDA, torch, or
transformers.
"""

from __future__ import annotations

import hashlib
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

_MAX_RUNNER_BYTES = 16 * 1024 * 1024
_MAX_PROC_FACT_BYTES = 128 * 1024
_IO_CHUNK_BYTES = 64 * 1024
_TENSORRT_LLM_LD_LIBRARY_PATH = "/usr/local/tensorrt/lib"
_TENSORRT_LLM_OPAL_PREFIX = "/opt/hpcx/ompi"
_TENSORRT_LLM_PATH = "/opt/hpcx/ompi/bin:/usr/bin:/bin"
_OFFICIAL_RUNNER_PATH = Path("/opt/invarlock/bin/tensorrt-llm-runner")
_VENDOR_PYTHON = Path("/opt/invarlock/bin/vendor-python")
_REQUIRED_EXECUTABLE_OWNER = (0, 0)


def official_tensorrt_llm_runner_path() -> Path:
    """Return the image-owned runner path used by execution transactions."""

    return Path(_OFFICIAL_RUNNER_PATH)


class TensorRTLLMExecutionError(RuntimeError):
    """Raised when authenticated TensorRT-LLM execution cannot continue."""


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


def _parent_entry_is_protected(
    parent: os.stat_result,
    entry: os.stat_result,
) -> bool:
    if parent.st_mode & 0o022 == 0:
        return True
    trusted_uids = {0, os.geteuid()}
    return (
        stat.S_ISDIR(parent.st_mode)
        and parent.st_mode & stat.S_ISVTX != 0
        and parent.st_uid in trusted_uids
        and entry.st_uid in trusted_uids
    )


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
        expected_sha256: str | None,
        require_executable: bool,
        require_secure_parents: bool = False,
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
            parent_stat = os.fstat(parent_descriptor)
            for component in absolute.parts[1:-1]:
                try:
                    next_descriptor = os.open(
                        component, flags, dir_fd=parent_descriptor
                    )
                except OSError as exc:
                    raise TensorRTLLMExecutionError(
                        "pinned file path contains a symlink or inaccessible directory"
                    ) from exc
                try:
                    next_stat = os.fstat(next_descriptor)
                    if require_secure_parents and not _parent_entry_is_protected(
                        parent_stat, next_stat
                    ):
                        raise TensorRTLLMExecutionError(
                            "pinned file path has a group- or other-writable parent"
                        )
                except Exception:
                    os.close(next_descriptor)
                    raise
                os.close(parent_descriptor)
                parent_descriptor = next_descriptor
                parent_stat = next_stat
            try:
                named = os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if require_secure_parents and not _parent_entry_is_protected(
                    parent_stat, named
                ):
                    raise TensorRTLLMExecutionError(
                        "pinned file path has a group- or other-writable parent"
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
                if expected_sha256 is not None and observed_sha256 != expected_sha256:
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


def _pin_trusted_executable(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
    max_bytes: int | None = None,
) -> _PinnedFile:
    pinned = _PinnedFile.open(
        path,
        expected_sha256=expected_sha256,
        require_executable=True,
        require_secure_parents=True,
        max_bytes=max_bytes,
    )
    if (pinned.initial_stat.st_uid, pinned.initial_stat.st_gid) != (
        _REQUIRED_EXECUTABLE_OWNER
    ):
        pinned.close()
        raise TensorRTLLMExecutionError(f"the {label} ownership is invalid")
    if pinned.initial_stat.st_mode & 0o022:
        pinned.close()
        raise TensorRTLLMExecutionError(f"the {label} is group- or other-writable")
    return pinned


def _pin_official_runner(
    path: Path,
    *,
    expected_sha256: str | None,
) -> _PinnedFile:
    if str(path) != str(_OFFICIAL_RUNNER_PATH):
        raise TensorRTLLMExecutionError(
            "the TensorRT-LLM runner must use the official installed path"
        )
    return _pin_trusted_executable(
        _OFFICIAL_RUNNER_PATH,
        expected_sha256=expected_sha256,
        label="official TensorRT-LLM runner",
        max_bytes=_MAX_RUNNER_BYTES,
    )


def _resolve_vendor_python() -> _PinnedFile:
    try:
        resolved = Path(_VENDOR_PYTHON).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TensorRTLLMExecutionError(
            "the fixed TensorRT-LLM Python interpreter cannot be resolved"
        ) from exc
    if not resolved.is_absolute():
        raise TensorRTLLMExecutionError(
            "the fixed TensorRT-LLM Python interpreter is not absolute"
        )
    return _pin_trusted_executable(
        resolved,
        expected_sha256=None,
        label="fixed TensorRT-LLM Python interpreter",
    )


def _parse_mount_id(payload: str) -> int:
    values = [
        line.partition(":")[2].strip()
        for line in payload.splitlines()
        if line.partition(":")[0] == "mnt_id"
    ]
    if len(values) != 1 or re.fullmatch(r"[1-9][0-9]*", values[0]) is None:
        raise TensorRTLLMExecutionError("descriptor mount identity is not canonical")
    return int(values[0])


def _read_bounded_proc_text(path: Path, *, label: str) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise TensorRTLLMExecutionError(f"{label} is unavailable") from exc
    try:
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_PROC_FACT_BYTES:
            chunk = os.read(
                descriptor,
                min(_IO_CHUNK_BYTES, _MAX_PROC_FACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
    except OSError as exc:
        raise TensorRTLLMExecutionError(f"{label} cannot be read") from exc
    finally:
        os.close(descriptor)
    if len(payload) > _MAX_PROC_FACT_BYTES:
        raise TensorRTLLMExecutionError(f"{label} exceeds the size limit")
    try:
        return payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise TensorRTLLMExecutionError(f"{label} is not canonical ASCII") from exc


def _descriptor_mount_id(descriptor: int) -> int:
    payload = _read_bounded_proc_text(
        Path(f"/proc/self/fdinfo/{descriptor}"),
        label="descriptor mount identity",
    )
    return _parse_mount_id(payload)


def _require_readonly_descriptor(descriptor: int, *, label: str) -> None:
    try:
        flags = os.fstatvfs(descriptor).f_flag
        readonly_flag = os.ST_RDONLY
    except (AttributeError, OSError) as exc:
        raise TensorRTLLMExecutionError(
            f"{label} filesystem facts are unavailable"
        ) from exc
    if flags & readonly_flag == 0:
        raise TensorRTLLMExecutionError(f"{label} requires a read-only filesystem")


_RESTRICTED_STATUS_FIELDS = (
    "CapInh",
    "CapPrm",
    "CapEff",
    "CapBnd",
    "CapAmb",
    "NoNewPrivs",
)


def _parse_restricted_process_status(payload: str) -> None:
    observed: dict[str, str] = {}
    for line in payload.splitlines():
        name, separator, value = line.partition(":")
        if not separator or name not in _RESTRICTED_STATUS_FIELDS:
            continue
        if name in observed:
            raise TensorRTLLMExecutionError("process security status is not canonical")
        observed[name] = value.strip()
    if set(observed) != set(_RESTRICTED_STATUS_FIELDS):
        raise TensorRTLLMExecutionError("process security status is incomplete")
    if observed["NoNewPrivs"] != "1" or any(
        observed[name] != "0000000000000000"
        for name in _RESTRICTED_STATUS_FIELDS
        if name != "NoNewPrivs"
    ):
        raise TensorRTLLMExecutionError(
            "process security status permits privilege acquisition"
        )


def _require_restricted_process_status() -> None:
    payload = _read_bounded_proc_text(
        Path("/proc/thread-self/status"),
        label="process security status",
    )
    _parse_restricted_process_status(payload)


@dataclass
class _ImmutableExecutionBoundary:
    root_descriptor: int = field(repr=False)
    root_initial_stat: os.stat_result = field(repr=False)
    mount_id: int
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        runner: _PinnedFile,
        vendor_python: _PinnedFile,
    ) -> _ImmutableExecutionBoundary:
        flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            root_descriptor = os.open("/", flags)
        except OSError as exc:
            raise TensorRTLLMExecutionError(
                "the immutable root filesystem cannot be opened"
            ) from exc
        try:
            boundary = cls(
                root_descriptor=root_descriptor,
                root_initial_stat=os.fstat(root_descriptor),
                mount_id=_descriptor_mount_id(root_descriptor),
            )
            boundary.recheck(runner, vendor_python)
            return boundary
        except Exception:
            os.close(root_descriptor)
            raise

    def recheck(
        self,
        runner: _PinnedFile,
        vendor_python: _PinnedFile,
    ) -> None:
        if self._closed:
            raise TensorRTLLMExecutionError("immutable execution boundary is closed")
        try:
            root_opened = os.fstat(self.root_descriptor)
            root_named = Path("/").lstat()
        except OSError as exc:
            raise TensorRTLLMExecutionError(
                "immutable root filesystem became unavailable"
            ) from exc
        expected_root = _directory_identity(self.root_initial_stat)
        if (
            not stat.S_ISDIR(root_opened.st_mode)
            or _directory_identity(root_opened) != expected_root
            or _directory_identity(root_named) != expected_root
        ):
            raise TensorRTLLMExecutionError("immutable root filesystem changed")
        runner.recheck()
        vendor_python.recheck()
        _require_readonly_descriptor(self.root_descriptor, label="root")
        _require_readonly_descriptor(runner.descriptor, label="runner")
        _require_readonly_descriptor(
            vendor_python.descriptor,
            label="interpreter",
        )
        mount_ids = {
            _descriptor_mount_id(self.root_descriptor),
            _descriptor_mount_id(runner.descriptor),
            _descriptor_mount_id(vendor_python.descriptor),
        }
        if mount_ids != {self.mount_id}:
            raise TensorRTLLMExecutionError(
                "runner and interpreter must remain on the same root mount"
            )
        _require_restricted_process_status()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.root_descriptor)


@dataclass
class _RunDirectory:
    path: Path = field(repr=False)
    descriptor: int = field(repr=False)
    initial_stat: os.stat_result = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(cls) -> _RunDirectory:
        path = Path(
            tempfile.mkdtemp(prefix="invarlock-tensorrt-llm-", dir="/tmp")
        ).resolve(strict=True)
        path.chmod(0o700)
        flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            descriptor = os.open(path, flags)
        except OSError:
            shutil.rmtree(path, ignore_errors=True)
            raise
        try:
            initial_stat = os.fstat(descriptor)
        except OSError:
            os.close(descriptor)
            shutil.rmtree(path, ignore_errors=True)
            raise
        return cls(
            path=path,
            descriptor=descriptor,
            initial_stat=initial_stat,
        )

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
        numeric_uid = str(os.getuid())
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
            "LD_LIBRARY_PATH": _TENSORRT_LLM_LD_LIBRARY_PATH,
            "LOGNAME": numeric_uid,
            "NO_COLOR": "1",
            "NO_PROXY": "*",
            "OPAL_PREFIX": _TENSORRT_LLM_OPAL_PREFIX,
            "PATH": _TENSORRT_LLM_PATH,
            "TELEMETRY_DISABLED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCHINDUCTOR_CACHE_DIR": f"{rendered}/torchinductor",
            "TRANSFORMERS_OFFLINE": "1",
            "TRITON_CACHE_DIR": f"{rendered}/triton",
            "TRTLLM_NO_USAGE_STATS": "1",
            "TMPDIR": rendered,
            "USER": numeric_uid,
            "XDG_CACHE_HOME": rendered,
        }

    def close(self) -> None:
        if self._closed:
            return
        cleanup_errors: list[Exception] = []
        safe_to_remove = False
        try:
            opened = os.fstat(self.descriptor)
            named = self.path.lstat()
            expected_object = (
                self.initial_stat.st_dev,
                self.initial_stat.st_ino,
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != expected_object
                or (named.st_dev, named.st_ino) != expected_object
            ):
                raise TensorRTLLMExecutionError(
                    "isolated runtime directory cannot be safely removed"
                )
            safe_to_remove = True
            os.fchmod(self.descriptor, 0o700)

            def fail_walk(error: OSError) -> None:
                raise TensorRTLLMExecutionError(
                    "isolated runtime directory cleanup preparation failed"
                ) from error

            for directory, child_directories, _files in os.walk(
                self.path,
                topdown=True,
                onerror=fail_walk,
                followlinks=False,
            ):
                for child in child_directories:
                    child_path = Path(directory, child)
                    if not child_path.is_symlink():
                        child_path.chmod(0o700)
                Path(directory).chmod(0o700)
        except Exception as exc:
            cleanup_errors.append(exc)
        finally:
            self._closed = True
            try:
                os.close(self.descriptor)
            except OSError as exc:
                cleanup_errors.append(exc)
        if safe_to_remove:
            try:
                shutil.rmtree(self.path)
            except OSError as exc:
                cleanup_errors.append(exc)
            if os.path.lexists(self.path):
                cleanup_errors.append(
                    TensorRTLLMExecutionError(
                        "isolated runtime directory cleanup left files behind"
                    )
                )
        if cleanup_errors:
            raise TensorRTLLMExecutionError(
                "isolated runtime directory cleanup failed"
            ) from cleanup_errors[0]


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
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
    runner: _PinnedFile,
    vendor_python: _PinnedFile,
    execution_boundary: _ImmutableExecutionBoundary,
    arguments: Sequence[str],
    input_bytes: bytes,
    run_directory: _RunDirectory,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[int, bytes, bytes]:
    def recheck_bindings() -> None:
        run_directory.recheck()
        execution_boundary.recheck(runner, vendor_python)

    recheck_bindings()
    vendor_python_path = str(vendor_python.path)
    runner_path = str(runner.path)
    try:
        try:
            process = subprocess.Popen(
                [vendor_python_path, runner_path, *arguments],
                executable=vendor_python_path,
                stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=run_directory.path,
                env=run_directory.environment(),
                shell=False,
                close_fds=True,
                pass_fds=(),
                start_new_session=True,
                bufsize=0,
            )
        except OSError as exc:
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM runner execution is unavailable"
            ) from exc
        selector: selectors.BaseSelector | None = None
        try:
            if (
                (input_bytes and process.stdin is None)
                or process.stdout is None
                or process.stderr is None
            ):
                raise TensorRTLLMExecutionError(
                    "TensorRT-LLM runner pipes are unavailable"
                )

            selector = selectors.DefaultSelector()
            stdout = bytearray()
            stderr = bytearray()
            input_offset = 0
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
                                input_bytes[
                                    input_offset : input_offset + _IO_CHUNK_BYTES
                                ],
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
                raise TensorRTLLMExecutionError(
                    "TensorRT-LLM record timed out"
                ) from exc
            return status, bytes(stdout), bytes(stderr)
        finally:
            if selector is not None:
                selector.close()
            for final_stream in (process.stdin, process.stdout, process.stderr):
                if final_stream is not None and not final_stream.closed:
                    final_stream.close()
            _kill_process_group(process)
    finally:
        recheck_bindings()
