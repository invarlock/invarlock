"""Reusable execution helpers for evidence workflow scripts."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class WorkflowCommandRun:
    name: str
    command: tuple[str, ...]
    returncode: int
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def workflow_return_code(results: Sequence[object]) -> int:
    """Return process exit code for a sequence with result.ok semantics."""
    return 0 if results and all(bool(getattr(result, "ok", False)) for result in results) else 1


def write_status_event(
    handle,
    event: str,
    *,
    slug: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> None:
    """Write a stable status.log event line."""
    parts = [f"[{datetime.now(UTC).isoformat()}]", event]
    if slug:
        parts.append(slug)
    for key, value in (fields or {}).items():
        rendered = "-" if value is None else str(value)
        parts.append(f"{key}={rendered}")
    handle.write(" ".join(parts) + "\n")
    handle.flush()


def run_logged_command(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    log_mode: str = "a",
    output_path: Path | None = None,
) -> WorkflowCommandRun:
    """Run a command while recording the command line and output sidecars."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            output_path.open("w", encoding="utf-8") as output_file,
            log_path.open(log_mode, encoding="utf-8") as log_file,
        ):
            log_file.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=output_file,
                stderr=log_file,
                text=True,
                check=False,
            )
    else:
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    return WorkflowCommandRun(
        name=name,
        command=tuple(command),
        returncode=proc.returncode,
    )


def run_logged_command_with_retry(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    log_mode: str = "a",
    output_path: Path | None = None,
    retry_returncodes: Sequence[int] = (),
    retry_message: str | None = None,
) -> WorkflowCommandRun:
    """Run a logged command and retry once for configured transient exits."""
    first = run_logged_command(
        name=name,
        command=command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        log_mode=log_mode,
        output_path=output_path,
    )
    if first.returncode not in set(retry_returncodes):
        return first
    with log_path.open("a", encoding="utf-8") as log_file:
        message = retry_message or (
            f"{name} exited with {first.returncode}; retrying once."
        )
        message = message.format(returncode=first.returncode, name=name)
        log_file.write(f"\n[WARN] {message}\n")
    second = run_logged_command(
        name=name,
        command=command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        log_mode="a",
        output_path=output_path,
    )
    return WorkflowCommandRun(
        name=name,
        command=tuple(command),
        returncode=second.returncode,
        attempts=2,
    )


__all__ = [
    "WorkflowCommandRun",
    "run_logged_command",
    "run_logged_command_with_retry",
    "workflow_return_code",
    "write_status_event",
]
