"""Bounded JSON subprocess transport for an explicitly trusted local pipeline."""

from __future__ import annotations

import asyncio
import os
import selectors
import signal
import subprocess
import threading
import time
from pathlib import Path

from examples.answer_capture import digest, exact, positive, read
from invarlock.evaluation_records.io import physical_file_digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes


class ProcessAdapter:
    """One process per operation, with no shell, retries, or inherited environment."""

    def __init__(self, path: Path, limits: dict):
        self.config = read(path)
        exact(self.config, {"argv", "assets", "environment"}, "process transport")
        argv = self.config["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(
                isinstance(item, str) and item and "\0" not in item for item in argv
            )
        ):
            raise ValueError("argv must be a nonempty string array")
        executable = Path(argv[0])
        if not executable.is_absolute() or executable.is_symlink():
            raise ValueError("argv executable must be an absolute non-symlink path")
        assets = self.config["assets"]
        if not isinstance(assets, list) or not all(
            isinstance(item, str) and Path(item).is_absolute() for item in assets
        ):
            raise ValueError(
                "assets must list absolute paths for scripts and local config"
            )
        names = self.config["environment"]
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) and name.isidentifier() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("environment must list unique variable names")
        self.env = {name: os.environ[name] for name in names if name in os.environ}
        self.files = {name: physical_file_digest(name) for name in [argv[0], *assets]}
        self.identity = {
            "transport": "local-json-process-v1",
            "transport_sha256": physical_file_digest(__file__),
            "config": self.config,
            "files": self.files,
        }
        self.sha256 = digest(self.identity)
        self.timeout = positive(limits["call_timeout_seconds"], "call timeout")
        self.deadline = positive(limits["deadline_unix_seconds"], "deadline")
        self.output_limit = (
            6 * positive(limits["max_output_bytes_per_call"], "output bytes") + 4096
        )

    def _call(self, operation: str, request: dict, cancelled: threading.Event) -> dict:
        for name, expected in self.files.items():
            if physical_file_digest(name) != expected:
                raise ValueError("pipeline executable or retained asset changed")
        payload = (
            canonical_json_bytes({"operation": operation, "request": request}) + b"\n"
        )
        if len(payload) > 1024 * 1024:
            raise ValueError("pipeline request exceeds the 1 MiB transport cap")
        deadline = min(
            time.monotonic() + self.timeout,
            time.monotonic() + self.deadline - time.time(),
        )
        if cancelled.is_set() or deadline <= time.monotonic():
            raise ValueError(
                "pipeline operation cancelled or fixed campaign deadline elapsed"
            )
        process = subprocess.Popen(
            self.config["argv"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            cwd="/",
            start_new_session=True,
        )
        assert process.stdin and process.stdout and process.stderr
        output, errors = bytearray(), bytearray()
        try:
            with selectors.DefaultSelector() as streams:
                for stream in (process.stdin, process.stdout, process.stderr):
                    os.set_blocking(stream.fileno(), False)
                    streams.register(
                        stream,
                        selectors.EVENT_WRITE
                        if stream is process.stdin
                        else selectors.EVENT_READ,
                    )
                offset = 0
                while streams.get_map() or process.poll() is None:
                    if cancelled.is_set() or time.monotonic() >= deadline:
                        raise ValueError("pipeline operation cancelled or timed out")
                    for key, _ in streams.select(
                        min(0.05, max(0, deadline - time.monotonic()))
                    ):
                        stream = key.fileobj
                        if stream is process.stdin:
                            try:
                                offset += os.write(
                                    stream.fileno(), payload[offset : offset + 65536]
                                )
                            except BrokenPipeError:
                                offset = len(payload)
                            if offset == len(payload):
                                streams.unregister(stream)
                                stream.close()
                        else:
                            chunk = os.read(stream.fileno(), 65536)
                            if not chunk:
                                streams.unregister(stream)
                                stream.close()
                                continue
                            target = output if stream is process.stdout else errors
                            limit = (
                                self.output_limit if stream is process.stdout else 65536
                            )
                            if len(target) + len(chunk) > limit:
                                raise ValueError(
                                    "pipeline stdout or stderr exceeded its byte cap"
                                )
                            target.extend(chunk)
                if process.returncode:
                    # Stderr can contain credentials; do not echo it in diagnostics.
                    raise ValueError(
                        f"pipeline process exited with status {process.returncode}"
                    )
            value = parse_json_bytes(bytes(output), label="pipeline response")
            exact(
                value,
                {"input_tokens"}
                if operation == "count_input_tokens"
                else {"output", "input_tokens", "output_tokens"},
                "pipeline response",
            )
            return value
        finally:
            # Kill the process group even if its leader exited but left children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def count_input_tokens(self, request: dict) -> int:
        return positive(
            self._call("count_input_tokens", request, threading.Event())[
                "input_tokens"
            ],
            "input tokens",
        )

    async def generate(self, request: dict) -> dict:
        cancelled = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(self._call, "generate", request, cancelled)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await task
            except ValueError:
                pass
            raise
