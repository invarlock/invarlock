from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from io import BytesIO
from typing import Any

import pytest

from invarlock._bounded_subprocess import close_selector_stream, communicate_bounded


class BoundedProcessError(RuntimeError):
    pass


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait(timeout=2)


def _process(
    code: str, *, stdout: int | None = subprocess.PIPE
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _communicate(
    process: subprocess.Popen[bytes],
    *,
    payload: bytes = b"request",
    timeout_seconds: int = 2,
    stdout_limit: int = 1024,
    terminate: Callable[[subprocess.Popen[bytes]], None] = _terminate,
    terminate_after: bool = False,
) -> tuple[int, bytes, bytes]:
    return communicate_bounded(
        process,
        input_bytes=payload,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=1024,
        error_type=BoundedProcessError,
        timeout_label="test process",
        output_label="test process",
        pipes_message="test pipes unavailable",
        terminate=terminate,
        terminate_after=terminate_after,
    )


def test_communicate_bounded_exchanges_input_and_captures_both_streams() -> None:
    process = _process(
        "import sys; data=sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(data.upper()); sys.stderr.buffer.write(b'notice')"
    )

    status, stdout, stderr = _communicate(process)

    assert status == 0
    assert stdout == b"REQUEST"
    assert stderr == b"notice"


def test_communicate_bounded_closes_empty_input_stream() -> None:
    process = _process(
        "import sys; data=sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(b'closed' if not data else b'open')"
    )

    status, stdout, stderr = _communicate(process, payload=b"")

    assert (status, stdout, stderr) == (0, b"closed", b"")


def test_communicate_bounded_rejects_output_over_limit() -> None:
    process = _process("import sys; sys.stdout.buffer.write(b'x' * 32)")

    with pytest.raises(BoundedProcessError, match="stdout limit exceeded"):
        _communicate(process, stdout_limit=8)

    assert process.poll() is not None


def test_communicate_bounded_rejects_stderr_over_limit() -> None:
    process = _process("import sys; sys.stderr.buffer.write(b'x' * 2048)")

    with pytest.raises(BoundedProcessError, match="stderr limit exceeded"):
        communicate_bounded(
            process,
            input_bytes=b"request",
            timeout_seconds=2,
            stdout_limit=1024,
            stderr_limit=8,
            error_type=BoundedProcessError,
            timeout_label="test process",
            output_label="test process",
            pipes_message="test pipes unavailable",
            terminate=_terminate,
        )

    assert process.poll() is not None


def test_communicate_bounded_times_out_and_terminates() -> None:
    process = _process("import time; time.sleep(10)")

    with pytest.raises(BoundedProcessError, match="timed out"):
        _communicate(process, timeout_seconds=0)

    assert process.poll() is not None


def test_communicate_bounded_times_out_after_child_closes_output() -> None:
    process = _process(
        "import os, time; os.close(1); os.close(2); os.read(0, 1024); time.sleep(10)"
    )

    with pytest.raises(BoundedProcessError, match="timed out"):
        _communicate(process, timeout_seconds=1)

    assert process.poll() is not None


def test_communicate_bounded_requires_capture_pipes() -> None:
    process = _process("pass", stdout=None)

    with pytest.raises(BoundedProcessError, match="pipes unavailable"):
        _communicate(process)

    assert process.poll() is not None


def test_communicate_bounded_can_terminate_after_success() -> None:
    process = _process("print('done')")
    calls: list[int | None] = []

    def record_termination(observed: subprocess.Popen[bytes]) -> None:
        calls.append(observed.poll())
        _terminate(observed)

    status, stdout, _stderr = _communicate(
        process,
        terminate=record_termination,
        terminate_after=True,
    )

    assert status == 0
    assert stdout == b"done\n"
    assert len(calls) == 1


def test_close_selector_stream_closes_after_missing_registration() -> None:
    class MissingSelector:
        def unregister(self, _stream: object) -> None:
            raise KeyError("missing")

    stream = BytesIO(b"payload")

    close_selector_stream(MissingSelector(), stream)  # type: ignore[arg-type]

    assert stream.closed


def test_communicate_bounded_handles_child_closing_stdin() -> None:
    process = _process("import os; os.close(0); print('closed')")

    status, stdout, _stderr = _communicate(process, payload=b"x" * (1024 * 1024))

    assert status == 0
    assert stdout == b"closed\n"


def test_communicate_bounded_retries_nonblocking_read(monkeypatch: Any) -> None:
    process = _process("print('done')")
    original_read = __import__("os").read
    attempts = 0

    def flaky_read(fd: int, size: int) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BlockingIOError
        return original_read(fd, size)

    monkeypatch.setattr("invarlock._bounded_subprocess.os.read", flaky_read)

    status, stdout, _stderr = _communicate(process)

    assert status == 0
    assert stdout == b"done\n"
    assert attempts > 1
