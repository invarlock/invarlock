"""Bounded subprocess execution shared by maintained integration examples."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from invarlock._bounded_subprocess import communicate_bounded

DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_STDOUT_LIMIT = 4 * 1024 * 1024
DEFAULT_STDERR_LIMIT = 4 * 1024 * 1024
_STOP_SECONDS = 5


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Stop one example subprocess and escalate if it ignores termination."""

    running = process.poll() is None
    process_id = getattr(process, "pid", None)
    if not running and not isinstance(process_id, int):
        return
    group_signaled = False
    if os.name == "posix" and isinstance(process_id, int) and process_id > 0:
        try:
            os.killpg(process_id, signal.SIGTERM)
            group_signaled = True
        except ProcessLookupError:
            if not running:
                return
        except OSError:
            pass
    if not group_signaled and running:
        process.terminate()
    try:
        process.wait(timeout=_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name == "posix" and isinstance(process_id, int) and process_id > 0:
            try:
                os.killpg(process_id, signal.SIGKILL)
            except ProcessLookupError:
                if process.poll() is None:
                    process.kill()
            except OSError:
                process.kill()
        else:
            process.kill()
        process.wait(timeout=_STOP_SECONDS)
    else:
        # The session leader can exit before descendants that inherited its
        # pipes. Escalate the isolated group so timeout/output-limit cleanup
        # cannot leave those descendants running.
        if group_signaled:
            try:
                assert isinstance(process_id, int)
                os.killpg(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    stdin_path: Path | None = None,
    stdout_path: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    stdout_limit: int = DEFAULT_STDOUT_LIMIT,
    stderr_limit: int = DEFAULT_STDERR_LIMIT,
    capture_output: bool = False,
    check: bool = False,
    label: str = "integration command",
) -> subprocess.CompletedProcess[str]:
    """Run an example command with bounded time, diagnostics, and cleanup."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("bounded command timeout must be a positive integer")
    if (
        isinstance(stdout_limit, bool)
        or not isinstance(stdout_limit, int)
        or stdout_limit <= 0
        or isinstance(stderr_limit, bool)
        or not isinstance(stderr_limit, int)
        or stderr_limit <= 0
    ):
        raise ValueError("bounded command output limits must be positive integers")

    argv = list(command)
    source = None
    destination = None
    completed = False
    try:
        if stdin_path is not None:
            source = stdin_path.open("rb")
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            destination = stdout_path.open("xb")
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment) if environment is not None else os.environ.copy(),
            stdin=source if source is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=os.name == "posix",
        )
        returncode, stdout, stderr = communicate_bounded(
            process,
            input_bytes=b"",
            timeout_seconds=timeout_seconds,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            stdout_destination=destination,
            error_type=RuntimeError,
            timeout_label=label,
            output_label=label,
            pipes_message=f"{label} did not expose pipes",
            terminate=_terminate,
        )
        completed = True
    except OSError as exc:
        raise RuntimeError(f"could not start command: {' '.join(argv)}") from exc
    finally:
        if source is not None:
            source.close()
        if destination is not None:
            destination.close()
        if not completed and stdout_path is not None:
            stdout_path.unlink(missing_ok=True)

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if not capture_output:
        if stdout_text:
            print(stdout_text, end="")
        if stderr_text:
            print(stderr_text, file=sys.stderr, end="")
    result = subprocess.CompletedProcess(
        argv,
        returncode,
        stdout_text if capture_output else None,
        stderr_text if capture_output else None,
    )
    if check and returncode != 0:
        if stdout_path is not None:
            stdout_path.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(
            returncode,
            argv,
            output=stdout_text,
            stderr=stderr_text,
        )
    return result


__all__ = [
    "DEFAULT_STDERR_LIMIT",
    "DEFAULT_STDOUT_LIMIT",
    "DEFAULT_TIMEOUT_SECONDS",
    "run_bounded_command",
]
