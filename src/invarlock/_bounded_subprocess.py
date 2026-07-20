"""Shared bounded subprocess communication for native runtime add-ins."""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from collections.abc import Callable
from typing import BinaryIO, cast

_IO_CHUNK_BYTES = 64 * 1024


def close_selector_stream(
    selector: selectors.BaseSelector,
    stream: BinaryIO,
) -> None:
    """Unregister and close one nonblocking subprocess stream."""

    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    stream.close()


def communicate_bounded(
    process: subprocess.Popen[bytes],
    *,
    input_bytes: bytes,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
    error_type: type[Exception],
    timeout_label: str,
    output_label: str,
    pipes_message: str,
    terminate: Callable[[subprocess.Popen[bytes]], None],
    terminate_after: bool = False,
) -> tuple[int, bytes, bytes]:
    """Exchange bytes with a subprocess under time and output bounds."""

    selector: selectors.BaseSelector | None = None
    failed = True
    try:
        if (
            (input_bytes and process.stdin is None)
            or process.stdout is None
            or process.stderr is None
        ):
            raise error_type(pipes_message)

        selector = selectors.DefaultSelector()
        stdout = bytearray()
        stderr = bytearray()
        input_offset = 0
        if not input_bytes and process.stdin is not None:
            process.stdin.close()
        streams = tuple(
            stream
            for stream in (process.stdin, process.stdout, process.stderr)
            if stream is not None and not stream.closed
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
                raise error_type(f"{timeout_label} timed out")
            events = selector.select(remaining)
            if not events:
                raise error_type(f"{timeout_label} timed out")
            for key, _mask in events:
                stream = cast(BinaryIO, key.fileobj)
                if key.data == "stdin":
                    try:
                        written = os.write(
                            stream.fileno(),
                            input_bytes[input_offset : input_offset + _IO_CHUNK_BYTES],
                        )
                    except BrokenPipeError:
                        close_selector_stream(selector, stream)
                        continue
                    input_offset += written
                    if input_offset == len(input_bytes):
                        close_selector_stream(selector, stream)
                    continue

                try:
                    chunk = os.read(stream.fileno(), _IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    close_selector_stream(selector, stream)
                    continue
                target = stdout if key.data == "stdout" else stderr
                target.extend(chunk)
                limit = stdout_limit if key.data == "stdout" else stderr_limit
                if len(target) > limit:
                    raise error_type(f"{output_label} {key.data} limit exceeded")

        try:
            status = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise error_type(f"{timeout_label} timed out") from exc
        failed = False
        return status, bytes(stdout), bytes(stderr)
    finally:
        if failed or terminate_after:
            terminate(process)
        if selector is not None:
            selector.close()
        for final_stream in (process.stdin, process.stdout, process.stderr):
            if final_stream is not None and not final_stream.closed:
                final_stream.close()
