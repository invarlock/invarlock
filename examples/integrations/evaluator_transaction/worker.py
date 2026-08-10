"""Example-owned OCI worker and bounded result-transfer support.

The production OCI executor owns InvarLock's paired runtime workers. This
module owns only the extra result-transfer protocol required by the maintained
external-evaluator demonstrations. It reuses the production worker process
deadline/log handling, but is deliberately not installed as an InvarLock API.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]
from invarlock.evaluation_oci import (
    OciEvaluationError,
    OciWorkerLimits,
    run_side_worker,
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")
_CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_DEFAULT_WORKER_CPUS = "4"
_DEFAULT_WORKER_MEMORY_MIB = 65536
_DEFAULT_WORKER_USER = "65532:65532"
_MAX_WORKER_DIAGNOSTIC_BYTES = 64 * 1024
_MAX_OUTER_WORKER_TIMEOUT_SECONDS = 24 * 60 * 60
_CONTAINER_STOP_SECONDS = 5
_CONTAINER_CONTROL_TIMEOUT_SECONDS = 10
_LEVEL3_DEFAULT_OUTPUT_BYTES = 64 * 1024 * 1024
_LEVEL3_MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
_LEVEL3_STATUS_RESERVE_BYTES = 64 * 1024
_LEVEL3_STATUS_PATH = "/outputs/.invarlock-level3-status"


def _artifact_mount_source(path: Path, *, label: str) -> Path:
    try:
        entry = path.lstat()
    except OSError as exc:
        raise OciEvaluationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(entry.st_mode) or not (
        stat.S_ISREG(entry.st_mode) or stat.S_ISDIR(entry.st_mode)
    ):
        raise OciEvaluationError(
            f"{label} must be a regular file or directory, not a symlink"
        )
    resolved = path.resolve(strict=True)
    if "," in str(resolved) or any(ord(character) < 32 for character in str(resolved)):
        raise OciEvaluationError(f"{label} path cannot be represented as an OCI mount")
    return resolved


def _worker_grants_read(
    entry: os.stat_result, *, uid: int, gid: int, directory: bool
) -> bool:
    if entry.st_uid == uid:
        read, search = stat.S_IRUSR, stat.S_IXUSR
    elif entry.st_gid == gid:
        read, search = stat.S_IRGRP, stat.S_IXGRP
    else:
        read, search = stat.S_IROTH, stat.S_IXOTH
    if not entry.st_mode & read:
        return False
    return not directory or bool(entry.st_mode & search)


def _assert_worker_readable(source: Path, *, user: str, label: str) -> None:
    """Reject mount sources the non-root worker user cannot read."""

    uid, _, gid = user.partition(":")
    worker_uid, worker_gid = int(uid), int(gid)
    unreadable: list[str] = []

    def visit(path: Path, *, directory: bool) -> None:
        try:
            entry = path.lstat()
        except OSError:
            unreadable.append(str(path.relative_to(source) if path != source else "."))
            return
        if stat.S_ISLNK(entry.st_mode):
            unreadable.append(str(path.relative_to(source) if path != source else "."))
            return
        if not _worker_grants_read(
            entry, uid=worker_uid, gid=worker_gid, directory=directory
        ):
            unreadable.append(str(path.relative_to(source) if path != source else "."))

    if source.is_dir():
        visit(source, directory=True)
        for root, directories, files in os.walk(source):
            for name in directories:
                visit(Path(root) / name, directory=True)
            for name in files:
                visit(Path(root) / name, directory=False)
    else:
        visit(source, directory=False)

    if unreadable:
        shown = ", ".join(sorted(unreadable)[:3])
        remainder = len(unreadable) - min(len(unreadable), 3)
        if remainder:
            shown = f"{shown} and {remainder} more"
        raise OciEvaluationError(
            f"{label} is not readable by the runtime worker user {user}: {shown}; "
            "grant world-readable permissions (for example "
            f"chmod -R a+rX {source}) or select a runtime user able to read it"
        )


def _worker_readable_artifact_mount_source(
    path: Path, *, user: str, label: str
) -> Path:
    resolved = _artifact_mount_source(path, label=label)
    _assert_worker_readable(resolved, user=user, label=label)
    return resolved


def _mount(source: Path, target: str, *, read_only: bool) -> list[str]:
    fields = ["type=bind", f"source={source}", f"target={target}"]
    if read_only:
        fields.append("readonly")
    return ["--mount", ",".join(fields)]


def _run_bounded_command(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int = _MAX_WORKER_DIAGNOSTIC_BYTES,
    stdout_path: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = run_bounded_command(
            list(command),
            capture_output=True,
            check=False,
            timeout_seconds=timeout_seconds,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            stdout_path=stdout_path,
            label="bounded evaluator-worker command",
        )
    except RuntimeError as exc:
        raise OciEvaluationError(str(exc)) from exc
    return subprocess.CompletedProcess(
        list(command),
        completed.returncode,
        (completed.stdout or "").encode("utf-8"),
        (completed.stderr or "").encode("utf-8"),
    )


def _level3_output_name(output: Path) -> str:
    name = output.name
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise OciEvaluationError("evaluator worker output name is invalid")
    return name


def compose_evaluator_worker_command(
    *,
    engine: str,
    image: str,
    entrypoint: Sequence[str],
    worker_arguments: Sequence[str],
    model_source: Path,
    dataset_source: Path,
    output: Path,
    control_root: Path,
    environment: Mapping[str, str] = MappingProxyType({}),
    timeout_seconds: int,
    output_limit_bytes: int = _LEVEL3_DEFAULT_OUTPUT_BYTES,
    cpus: str = _DEFAULT_WORKER_CPUS,
    memory_mib: int = _DEFAULT_WORKER_MEMORY_MIB,
    user: str = _DEFAULT_WORKER_USER,
) -> list[str]:
    """Compose one example worker without writable host output mounts."""

    if engine not in {"docker", "podman"}:
        raise OciEvaluationError(
            "evaluator worker container engine must be docker or podman"
        )
    if _DIGEST_RE.fullmatch(image) is None:
        raise OciEvaluationError("evaluator worker image must be an immutable digest")
    if not entrypoint or any(
        not isinstance(value, str) or not value for value in entrypoint
    ):
        raise OciEvaluationError("evaluator worker entrypoint is invalid")
    if any(not isinstance(value, str) or "\x00" in value for value in worker_arguments):
        raise OciEvaluationError("evaluator worker arguments are invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_OUTER_WORKER_TIMEOUT_SECONDS
    ):
        raise OciEvaluationError("evaluator worker timeout is invalid")
    if (
        isinstance(output_limit_bytes, bool)
        or not isinstance(output_limit_bytes, int)
        or output_limit_bytes <= 0
        or output_limit_bytes > _LEVEL3_MAX_OUTPUT_BYTES
    ):
        raise OciEvaluationError("evaluator worker output limit is invalid")
    limits = OciWorkerLimits(cpus=cpus, memory_mib=memory_mib, user=user)
    cpus, memory_mib, user = limits.cpus, limits.memory_mib, limits.user
    _level3_output_name(output)
    try:
        control_stat = control_root.lstat()
    except OSError as exc:
        raise OciEvaluationError(
            "evaluator worker control directory is unavailable"
        ) from exc
    if control_root.is_symlink() or not stat.S_ISDIR(control_stat.st_mode):
        raise OciEvaluationError("evaluator worker control directory is unsafe")
    control_root = control_root.resolve(strict=True)
    cidfile = control_root / "container.cid"
    if cidfile.exists() or cidfile.is_symlink():
        raise OciEvaluationError(
            "evaluator worker container ID destination already exists"
        )
    container_name = re.sub(r"[^A-Za-z0-9_.-]", "-", control_root.name)
    container_name = f"invarlock-evaluator-{container_name}"[:63]
    if _CONTAINER_NAME_RE.fullmatch(container_name) is None:
        raise OciEvaluationError("evaluator worker container name is invalid")
    if output.exists() or output.is_symlink():
        raise OciEvaluationError("evaluator worker output destination must be new")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise OciEvaluationError("evaluator worker output parent is unsafe")
    worker_uid = user.partition(":")[0]
    artifact = _worker_readable_artifact_mount_source(
        model_source, user=user, label="evaluator model"
    )
    dataset = _worker_readable_artifact_mount_source(
        dataset_source, user=user, label="evaluator dataset"
    )
    command = [
        engine,
        "run",
        "--init",
        "--pull=never",
        "--cidfile",
        str(cidfile),
        "--name",
        container_name,
        "--stop-timeout",
        str(_CONTAINER_STOP_SECONDS),
        "--network",
        "none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "1024",
        "--cpus",
        cpus,
        "--memory",
        f"{memory_mib}m",
        "--user",
        user,
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=2g",
        "--tmpfs",
        (
            "/outputs:rw,noexec,nosuid,nodev,size="
            f"{output_limit_bytes + _LEVEL3_STATUS_RESERVE_BYTES}"
        ),
        *_mount(artifact, "/model", read_only=True),
        *_mount(dataset, "/records.jsonl", read_only=True),
        "--entrypoint",
        "/bin/sh",
    ]
    for key, value in sorted(environment.items()):
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise OciEvaluationError("evaluator worker environment is invalid")
        command.extend(("--env", f"{key}={value}"))
    command.extend(
        [
            "--env",
            "HOME=/tmp",
            "--env",
            f"LOGNAME={worker_uid}",
            "--env",
            "USER=invarlock-evaluator",
            image,
            "-c",
            (
                "set -eu; "
                f"rm -f {_LEVEL3_STATUS_PATH}; "
                '"$@" & worker_pid=$!; '
                'set +e; wait "$worker_pid"; worker_status=$?; set -e; '
                f"printf '%s' \"$worker_status\" > {_LEVEL3_STATUS_PATH}; "
                "while :; do sleep 1; done"
            ),
            "--",
            *entrypoint,
            *worker_arguments,
        ]
    )
    return command


def _extract_output_archive(
    payload: bytes | Path,
    *,
    staging_root: Path,
    output_name: str,
    max_bytes: int,
) -> Path:
    """Extract one worker-owned tar stream without accepting links or devices."""

    if isinstance(payload, bytes):
        if len(payload) > max_bytes + _LEVEL3_STATUS_RESERVE_BYTES:
            raise OciEvaluationError("evaluator worker output exceeds its size limit")
        payload_stream: io.BufferedIOBase = io.BytesIO(payload)
    else:
        try:
            facts = payload.lstat()
            if payload.is_symlink() or not stat.S_ISREG(facts.st_mode):
                raise OciEvaluationError("evaluator worker output archive is unsafe")
            if facts.st_size > max_bytes + _LEVEL3_STATUS_RESERVE_BYTES:
                raise OciEvaluationError(
                    "evaluator worker output exceeds its size limit"
                )
            payload_stream = payload.open("rb")
        except OSError as exc:
            raise OciEvaluationError(
                "evaluator worker output archive is unavailable"
            ) from exc
    staged_output = staging_root / output_name
    total_output_bytes = 0
    try:
        with payload_stream, tarfile.open(fileobj=payload_stream, mode="r:") as archive:
            members = archive.getmembers()
            if not members:
                raise OciEvaluationError("evaluator worker output archive is empty")
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or not path.parts
                    or path.parts[0] != output_name
                    or ".." in path.parts
                ):
                    raise OciEvaluationError(
                        "evaluator worker output archive contains an unsafe path"
                    )
                destination = staging_root.joinpath(*path.parts)
                if (
                    member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise OciEvaluationError(
                        "evaluator worker output archive contains an unsafe entry"
                    )
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=False)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    raise OciEvaluationError(
                        "evaluator worker output archive contains duplicate paths"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise OciEvaluationError(
                        "evaluator worker output archive contains an unreadable file"
                    )
                with source, destination.open("xb") as handle:
                    while chunk := source.read(1024 * 1024):
                        total_output_bytes += len(chunk)
                        if total_output_bytes > max_bytes:
                            raise OciEvaluationError(
                                "evaluator worker output exceeds its size limit"
                            )
                        handle.write(chunk)
    except (OSError, tarfile.TarError) as exc:
        raise OciEvaluationError(
            "evaluator worker output archive could not be extracted"
        ) from exc
    if not staged_output.is_dir() or staged_output.is_symlink():
        raise OciEvaluationError("evaluator worker output archive lacks its result dir")
    return staged_output


def _worker_cidfile(command: Sequence[str]) -> Path | None:
    indexes = [index for index, value in enumerate(command) if value == "--cidfile"]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        return None
    return Path(command[indexes[0] + 1])


def _read_worker_container_id(cidfile: Path | None) -> str | None:
    if cidfile is None or not cidfile.is_file() or cidfile.is_symlink():
        return None
    try:
        value = cidfile.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value if _CONTAINER_ID_RE.fullmatch(value) is not None else None


def _worker_container_name(command: Sequence[str]) -> str | None:
    indexes = [index for index, value in enumerate(command) if value == "--name"]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        return None
    value = command[indexes[0] + 1]
    return value if _CONTAINER_NAME_RE.fullmatch(value) is not None else None


def _worker_container_handle(
    command: Sequence[str], cidfile: Path | None
) -> str | None:
    return _read_worker_container_id(cidfile) or _worker_container_name(command)


def _container_control(
    engine_path: str,
    action: Literal["stop", "kill"],
    container_handle: str,
) -> None:
    command = [engine_path, action]
    if action == "stop":
        command.extend(["--time", str(_CONTAINER_STOP_SECONDS)])
    command.append(container_handle)
    try:
        _run_bounded_command(
            command,
            timeout_seconds=_CONTAINER_CONTROL_TIMEOUT_SECONDS,
            stdout_limit=_MAX_WORKER_DIAGNOSTIC_BYTES,
        )
    except OciEvaluationError:
        return


def _remove_worker_container(engine_path: str, container_handle: str) -> None:
    command = [engine_path, "rm", "--force", "--volumes", container_handle]
    try:
        completed = _run_bounded_command(
            command,
            timeout_seconds=_CONTAINER_CONTROL_TIMEOUT_SECONDS,
            stdout_limit=_MAX_WORKER_DIAGNOSTIC_BYTES,
        )
    except OciEvaluationError as exc:
        raise OciEvaluationError("evaluator worker cleanup did not complete") from exc
    if completed.returncode:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OciEvaluationError(
            diagnostic or "evaluator worker cleanup returned a failure"
        )


def run_evaluator_worker(
    *,
    engine: str,
    image: str,
    entrypoint: Sequence[str],
    worker_arguments: Sequence[str],
    model_source: Path,
    dataset_source: Path,
    output: Path,
    environment: Mapping[str, str] = MappingProxyType({}),
    timeout_seconds: int,
    output_limit_bytes: int = _LEVEL3_DEFAULT_OUTPUT_BYTES,
    cpus: str = _DEFAULT_WORKER_CPUS,
    memory_mib: int = _DEFAULT_WORKER_MEMORY_MIB,
    user: str = _DEFAULT_WORKER_USER,
) -> subprocess.CompletedProcess[str]:
    """Run an evaluator worker and safely transfer its bounded tmpfs result."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="invarlock-evaluator-control-"
    ) as raw_control:
        control_root = Path(raw_control)
        control_root.chmod(0o700)
        staging_root = Path(
            tempfile.mkdtemp(prefix=".invarlock-evaluator-transfer-", dir=output.parent)
        )
        command = compose_evaluator_worker_command(
            engine=engine,
            image=image,
            entrypoint=entrypoint,
            worker_arguments=worker_arguments,
            model_source=model_source,
            dataset_source=dataset_source,
            output=output,
            control_root=control_root,
            environment=environment,
            timeout_seconds=timeout_seconds,
            output_limit_bytes=output_limit_bytes,
            cpus=cpus,
            memory_mib=memory_mib,
            user=user,
        )
        cidfile = _worker_cidfile(command)
        detached_command = list(command)
        detached_command.insert(2, "--detach")
        completed: subprocess.CompletedProcess[str] | None = None
        container_handle: str | None = None
        try:
            completed = run_side_worker(
                detached_command,
                timeout_seconds=timeout_seconds,
            )
            container_handle = _worker_container_handle(command, cidfile)
            if container_handle is None:
                raise OciEvaluationError(
                    "evaluator worker did not publish an engine-recognized container handle"
                )
            if completed.returncode:
                return completed
            deadline = time.monotonic() + timeout_seconds
            worker_status: int | None = None
            while time.monotonic() < deadline:
                status = _run_bounded_command(
                    [
                        command[0],
                        "exec",
                        container_handle,
                        "cat",
                        _LEVEL3_STATUS_PATH,
                    ],
                    timeout_seconds=_CONTAINER_CONTROL_TIMEOUT_SECONDS,
                    stdout_limit=_LEVEL3_STATUS_RESERVE_BYTES,
                )
                if status.returncode == 0:
                    try:
                        rendered_status = status.stdout.decode("ascii")
                    except UnicodeDecodeError as exc:
                        raise OciEvaluationError(
                            "evaluator worker completion status is unreadable"
                        ) from exc
                    if rendered_status not in {str(value) for value in range(256)}:
                        raise OciEvaluationError(
                            "evaluator worker completion status is invalid"
                        )
                    worker_status = int(rendered_status)
                    break
                time.sleep(0.05)
            if worker_status is None:
                raise OciEvaluationError(
                    f"evaluator worker exceeded its {timeout_seconds}-second outer deadline"
                )
            completed = subprocess.CompletedProcess(
                list(command), worker_status, completed.stdout, completed.stderr
            )
            if worker_status:
                logs = _run_bounded_command(
                    [command[0], "logs", container_handle],
                    timeout_seconds=_CONTAINER_CONTROL_TIMEOUT_SECONDS,
                    stdout_limit=_MAX_WORKER_DIAGNOSTIC_BYTES,
                )
                diagnostic = logs.stdout
                if logs.stderr:
                    diagnostic += b"\n" + logs.stderr
                completed = subprocess.CompletedProcess(
                    list(command),
                    worker_status,
                    "",
                    diagnostic.decode("utf-8", errors="replace"),
                )
                return completed
            output_name = _level3_output_name(output)
            archive_path = staging_root / ".evaluator-output.tar"
            copied = _run_bounded_command(
                [
                    command[0],
                    "exec",
                    container_handle,
                    "tar",
                    "-C",
                    "/outputs",
                    "-cf",
                    "-",
                    "--",
                    output_name,
                ],
                timeout_seconds=_CONTAINER_CONTROL_TIMEOUT_SECONDS,
                stdout_limit=output_limit_bytes + _LEVEL3_STATUS_RESERVE_BYTES,
                stdout_path=archive_path,
            )
            if copied.returncode:
                raise OciEvaluationError(
                    copied.stderr.decode("utf-8", errors="replace").strip()
                    or "evaluator worker output transfer failed"
                )
            staged_output = _extract_output_archive(
                archive_path,
                staging_root=staging_root,
                output_name=output_name,
                max_bytes=output_limit_bytes,
            )
            staged_output.rename(output)
            if output.is_symlink() or not output.is_dir():
                raise OciEvaluationError("evaluator worker output transfer was unsafe")
            return completed
        finally:
            if container_handle is None:
                container_handle = _worker_container_handle(command, cidfile)
            if container_handle is not None:
                try:
                    _remove_worker_container(command[0], container_handle)
                finally:
                    if cidfile is not None:
                        cidfile.unlink(missing_ok=True)
            elif cidfile is not None:
                cidfile.unlink(missing_ok=True)
            shutil.rmtree(staging_root, ignore_errors=True)


__all__ = [
    "compose_evaluator_worker_command",
    "run_evaluator_worker",
]
