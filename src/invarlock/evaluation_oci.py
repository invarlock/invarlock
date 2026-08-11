"""Digest-pinned OCI execution for independently isolated evaluation sides."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from invarlock._bounded_subprocess import communicate_bounded
from invarlock._optional_runtime_profiles import OPTIONAL_RUNTIME_PROVIDER_PROFILES
from invarlock.core.evaluation_request import (
    MAX_EVALUATION_REQUEST_BYTES,
    ComparisonSideRequest,
    EvaluationRequest,
    _load_yaml,
    _reject_include_directives,
    _validate_schema,
)
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import (
    RuntimeArtifactResources,
    RuntimeProvider,
    parse_runtime_behavioral_schedule_json,
)
from invarlock.evaluation_run import EvaluationRunResult, load_runtime_side_evidence
from invarlock.evaluation_runtime import (
    CallerRuntimeResources,
    ProviderResourceBinding,
    RuntimeSideRole,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.runtime_provider_evidence import decode_runtime_provider_receipt
from invarlock.runtime_security_helpers import (
    ALLOW_NETWORK_ENV,
    ALLOW_REMOTE_CODE_ENV,
    ALLOW_THIRD_PARTY_PLUGINS_ENV,
    CONTAINER_EXECUTION_ENV,
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
)

type ContainerEngine = Literal["docker", "podman"]
type EntrypointProfile = Literal["auto", "python", "nvidia"]

CONTAINER_ENGINE_ENV = "INVARLOCK_CONTAINER_ENGINE"
RUNTIME_DEVICE_ENV = "INVARLOCK_RUNTIME_DEVICE"
BASELINE_RUNTIME_DEVICE_ENV = "INVARLOCK_BASELINE_RUNTIME_DEVICE"
SUBJECT_RUNTIME_DEVICE_ENV = "INVARLOCK_SUBJECT_RUNTIME_DEVICE"
BASELINE_RUNTIME_IMAGE_ENV = "INVARLOCK_BASELINE_RUNTIME_IMAGE"
SUBJECT_RUNTIME_IMAGE_ENV = "INVARLOCK_SUBJECT_RUNTIME_IMAGE"
BASELINE_RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_BASELINE_RUNTIME_IMAGE_DIGEST"
SUBJECT_RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_SUBJECT_RUNTIME_IMAGE_DIGEST"
RUNTIME_ENTRYPOINT_ENV = "INVARLOCK_RUNTIME_ENTRYPOINT"
BASELINE_RUNTIME_ENTRYPOINT_ENV = "INVARLOCK_BASELINE_RUNTIME_ENTRYPOINT"
SUBJECT_RUNTIME_ENTRYPOINT_ENV = "INVARLOCK_SUBJECT_RUNTIME_ENTRYPOINT"
RUNTIME_CPUS_ENV = "INVARLOCK_RUNTIME_CPUS"
RUNTIME_MEMORY_MIB_ENV = "INVARLOCK_RUNTIME_MEMORY_MIB"
RUNTIME_USER_ENV = "INVARLOCK_RUNTIME_USER"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(0|[1-9][0-9]*))?$")
_CPU_LIMIT_RE = re.compile(r"^(?:0|[1-9][0-9]{0,3})(?:\.[0-9]{1,3})?$")
_NUMERIC_USER_RE = re.compile(r"^(?P<uid>[1-9][0-9]{0,9}):(?P<gid>[1-9][0-9]{0,9})$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")
_BARE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_IMAGE_INSPECT_BYTES = 1024 * 1024
_MAX_WORKER_DIAGNOSTIC_BYTES = 64 * 1024
_GIBIBYTE = 1024**3
_DEFAULT_WORKER_CPUS = "4"
_DEFAULT_WORKER_MEMORY_MIB = 65536
_DEFAULT_WORKER_USER = "65532:65532"
_MAX_WORKER_CPUS = Decimal("4096")
_MIN_WORKER_MEMORY_MIB = 128
_MAX_WORKER_MEMORY_MIB = 4 * 1024 * 1024
_MAX_OUTER_WORKER_TIMEOUT_SECONDS = 24 * 60 * 60
_WORKER_TIMEOUT_ALLOWANCE_CALLS = 2
_CONTAINER_STOP_SECONDS = 5
_CONTAINER_CONTROL_TIMEOUT_SECONDS = 10
_WORKER_DRAIN_JOIN_SECONDS = 0.5
_WORKER_DRAIN_POLL_SECONDS = 0.01
_DEFAULT_WORKER_TMPFS_GIB = 4
_MAX_WORKER_TMPFS_GIB = 64
_TENSORRT_ENGINE_COPY_FACTOR = 2
_TENSORRT_SCRATCH_RESERVE_BYTES = _GIBIBYTE
_ROLES: tuple[RuntimeSideRole, RuntimeSideRole] = ("baseline", "subject")


class OciEvaluationError(ValueError):
    """Raised before or after an unsafe, ambiguous, or failed side launch."""


@dataclass(frozen=True)
class _LocalImageInspection:
    """Local config identity and immutable repository-manifest identities."""

    config_id: str
    repo_digests: tuple[str, ...]


def evaluation_request_execution_mode(path: Path) -> str | None:
    """Read only the schema-validated execution discriminator for CLI dispatch."""

    try:
        payload = read_regular_file_bytes(
            Path(path),
            label="evaluation request",
            max_bytes=MAX_EVALUATION_REQUEST_BYTES,
        )
        value = _load_yaml(payload)
        _reject_include_directives(value)
        value = _validate_schema(value)
    except (OSError, StrictJsonError, ValueError):
        return None
    execution = value.get("execution")
    if not isinstance(execution, dict):
        return None
    mode = execution.get("mode")
    return mode if mode in {"run", "import"} else None


@dataclass(frozen=True)
class OciSideLaunch:
    """Caller-owned OCI binding for exactly one comparison side."""

    image_ref: str
    image_digest: str
    device: str
    entrypoint: EntrypointProfile = "auto"


def _cpu_limit(value: object) -> str:
    if not isinstance(value, str) or _CPU_LIMIT_RE.fullmatch(value) is None:
        raise OciEvaluationError(
            "runtime CPU limit must be a positive decimal with at most three places"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex excludes this
        raise OciEvaluationError("runtime CPU limit is invalid") from exc
    if parsed <= 0 or parsed > _MAX_WORKER_CPUS:
        raise OciEvaluationError(
            "runtime CPU limit must be greater than 0 and at most 4096"
        )
    return value


def _memory_limit_mib(value: object) -> int:
    if isinstance(value, bool):
        raise OciEvaluationError("runtime memory limit must be an integer MiB value")
    if isinstance(value, str):
        if re.fullmatch(r"[1-9][0-9]{0,7}", value) is None:
            raise OciEvaluationError(
                "runtime memory limit must be a canonical integer MiB value"
            )
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    else:
        raise OciEvaluationError("runtime memory limit must be an integer MiB value")
    if not _MIN_WORKER_MEMORY_MIB <= parsed <= _MAX_WORKER_MEMORY_MIB:
        raise OciEvaluationError(
            "runtime memory limit must be between 128 and 4194304 MiB"
        )
    return parsed


def _runtime_user(value: object) -> str:
    if (
        not isinstance(value, str)
        or (match := _NUMERIC_USER_RE.fullmatch(value)) is None
    ):
        raise OciEvaluationError("runtime user must be a non-root numeric UID:GID pair")
    if (
        int(match.group("uid")) > 4_294_967_294
        or int(match.group("gid")) > 4_294_967_294
    ):
        raise OciEvaluationError(
            "runtime UID and GID must fit the portable 32-bit range"
        )
    return value


@dataclass(frozen=True)
class OciWorkerLimits:
    """Caller-owned hard limits applied independently to every side worker."""

    cpus: str = _DEFAULT_WORKER_CPUS
    memory_mib: int = _DEFAULT_WORKER_MEMORY_MIB
    user: str = _DEFAULT_WORKER_USER

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpus", _cpu_limit(self.cpus))
        object.__setattr__(self, "memory_mib", _memory_limit_mib(self.memory_mib))
        object.__setattr__(self, "user", _runtime_user(self.user))


@dataclass(frozen=True)
class OciEvaluationLaunch:
    """Closed launch policy for two independently pinned runtime images."""

    engine: ContainerEngine
    baseline: OciSideLaunch
    subject: OciSideLaunch
    worker_limits: OciWorkerLimits = OciWorkerLimits()
    engine_path: str | None = None


def _inspect_output_bytes(value: object, *, label: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise OciEvaluationError(f"runtime image inspect {label} is invalid")


def _terminate_bounded_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate one bounded control subprocess without leaving it behind."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_CONTAINER_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_CONTAINER_STOP_SECONDS)


def _run_bounded_command(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int = _MAX_WORKER_DIAGNOSTIC_BYTES,
    stdout_path: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one engine control command with bounded pipes and optional streaming."""

    destination = None
    process: subprocess.Popen[bytes] | None = None
    completed = False
    try:
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            destination = stdout_path.open("xb")
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        returncode, stdout, stderr = communicate_bounded(
            process,
            input_bytes=b"",
            timeout_seconds=timeout_seconds,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            stdout_destination=destination,
            error_type=OciEvaluationError,
            timeout_label="bounded engine command",
            output_label="bounded engine command",
            pipes_message="bounded engine command did not expose pipes",
            terminate=_terminate_bounded_process,
        )
        completed = True
        return subprocess.CompletedProcess(list(command), returncode, stdout, stderr)
    except OciEvaluationError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OciEvaluationError("bounded engine command could not complete") from exc
    finally:
        if destination is not None:
            destination.close()
        if not completed and stdout_path is not None:
            stdout_path.unlink(missing_ok=True)


def _normalized_config_id(value: object) -> str:
    if not isinstance(value, str):
        raise OciEvaluationError("runtime image inspect config ID is invalid")
    if _DIGEST_RE.fullmatch(value) is not None:
        return value
    if _BARE_SHA256_RE.fullmatch(value) is not None:
        return f"sha256:{value}"
    raise OciEvaluationError("runtime image inspect config ID is invalid")


def _repository_digest(value: object) -> tuple[str, str, str]:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or value.count("@") != 1
        or any(ord(character) < 32 or character.isspace() for character in value)
    ):
        raise OciEvaluationError("runtime image inspect repository digest is invalid")
    repository, digest = value.rsplit("@", 1)
    if not repository or repository.startswith(("-", "/", "\\")):
        raise OciEvaluationError("runtime image inspect repository digest is invalid")
    if _DIGEST_RE.fullmatch(digest) is None:
        raise OciEvaluationError("runtime image inspect repository digest is invalid")
    return repository, digest, value


def _inspect_local_image(engine_path: str, image: str) -> _LocalImageInspection:
    """Read one bounded Docker- or Podman-shaped local image inspection."""

    try:
        completed = _run_bounded_command(
            [engine_path, "image", "inspect", image],
            timeout_seconds=30,
            stdout_limit=_MAX_IMAGE_INSPECT_BYTES,
        )
    except OciEvaluationError as exc:
        raise OciEvaluationError(
            "runtime image could not be inspected locally"
        ) from exc
    stdout = _inspect_output_bytes(completed.stdout, label="stdout")
    stderr = _inspect_output_bytes(completed.stderr, label="stderr")
    if completed.returncode != 0:
        diagnostic = stderr.decode("utf-8", errors="replace").strip()[:512]
        diagnostic = diagnostic or "image is unavailable"
        raise OciEvaluationError(f"runtime image could not be inspected: {diagnostic}")
    if len(stdout) > _MAX_IMAGE_INSPECT_BYTES:
        raise OciEvaluationError(
            "runtime image inspect output exceeds the bounded size limit"
        )
    try:
        decoded = parse_json_bytes(stdout, label="runtime image inspect output")
    except StrictJsonError as exc:
        raise OciEvaluationError(str(exc)) from exc
    if isinstance(decoded, list):
        if len(decoded) != 1 or not isinstance(decoded[0], dict):
            raise OciEvaluationError(
                "runtime image inspect must return exactly one image object"
            )
        payload = decoded[0]
    elif isinstance(decoded, dict):
        # Podman JSON format can expose the single inspection object directly.
        payload = decoded
    else:
        raise OciEvaluationError(
            "runtime image inspect must return exactly one image object"
        )
    config_id = _normalized_config_id(payload.get("Id"))
    raw_repo_digests = payload.get("RepoDigests")
    if raw_repo_digests is None:
        raw_repo_digests = []
    if not isinstance(raw_repo_digests, list):
        raise OciEvaluationError("runtime image inspect RepoDigests are invalid")
    repo_digests = tuple(
        sorted({_repository_digest(value)[2] for value in raw_repo_digests})
    )
    return _LocalImageInspection(config_id=config_id, repo_digests=repo_digests)


def _tag_repository(image: str) -> str:
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    repository = image[:last_colon] if last_colon > last_slash else image
    if not repository:
        raise OciEvaluationError("runtime image must include a repository name")
    return repository


def _resolve_inspected_image(
    image: str,
    digest: str,
    inspection: _LocalImageInspection,
    *,
    allow_tag: bool,
) -> str:
    if image == digest:
        if inspection.config_id != digest:
            raise OciEvaluationError(
                "runtime local config ID and supplied digest do not agree"
            )
        return digest
    if "@" in image:
        repository, embedded = image.rsplit("@", 1)
        if embedded != digest or image not in inspection.repo_digests:
            raise OciEvaluationError(
                "runtime repository manifest and supplied digest do not agree"
            )
        return f"{repository}@{digest}"
    if not allow_tag:
        raise OciEvaluationError(
            "runtime image execution reference must be an immutable config ID or "
            "repository manifest"
        )
    repository = _tag_repository(image)
    selected = f"{repository}@{digest}"
    if selected in inspection.repo_digests:
        return selected
    if inspection.config_id == digest:
        raise OciEvaluationError(
            "tagged runtime images cannot declare a local config ID; use the "
            "config ID as both the image reference and digest"
        )
    raise OciEvaluationError(
        "runtime image tag does not resolve to the supplied repository manifest"
    )


def _portable_image_ref(image_ref: str, image_digest: str, *, engine_path: str) -> str:
    image = image_ref.strip()
    digest = image_digest.strip()
    if _DIGEST_RE.fullmatch(digest) is None:
        raise OciEvaluationError(
            "runtime image digest: INVARLOCK_RUNTIME_IMAGE_DIGEST or its per-side "
            "override must use lowercase sha256:<64 hex>"
        )
    if (
        not image
        or image != image_ref
        or image.startswith(("-", "/", "\\"))
        or any(ord(character) < 32 or character.isspace() for character in image)
        or image.count("@") > 1
    ):
        raise OciEvaluationError("runtime image must be a portable OCI reference")
    if image == digest:
        return image
    if _DIGEST_RE.fullmatch(image) is not None:
        raise OciEvaluationError(
            "runtime image reference and supplied digest do not agree"
        )
    if "@" in image:
        repository, embedded = image.rsplit("@", 1)
        if not repository or embedded != digest:
            raise OciEvaluationError(
                "runtime image reference and supplied digest do not agree"
            )
    inspection = _inspect_local_image(engine_path, image)
    return _resolve_inspected_image(image, digest, inspection, allow_tag=True)


def _embedded_digest(image: str) -> str:
    if _DIGEST_RE.fullmatch(image) is not None:
        return image
    return image.rsplit("@", 1)[1] if "@" in image else ""


def _engine_binary(engine: str) -> tuple[ContainerEngine, str]:
    if engine not in {"docker", "podman"}:
        raise OciEvaluationError("container engine must be docker or podman")
    executable = shutil.which(engine)
    if executable is None:
        raise OciEvaluationError(f"container engine {engine!r} is not available")
    return cast(ContainerEngine, engine), str(Path(executable).resolve())


def _device(value: str, *, label: str) -> str:
    device = value.strip().lower()
    if device == "cpu" or _CUDA_DEVICE_RE.fullmatch(device) is not None:
        return device
    raise OciEvaluationError(f"{label} must be cpu, cuda, or cuda:<index>")


def _entrypoint(value: str, *, label: str) -> EntrypointProfile:
    profile = value.strip().lower()
    if profile not in {"auto", "python", "nvidia"}:
        raise OciEvaluationError(f"{label} must be auto, python, or nvidia")
    return cast(EntrypointProfile, profile)


def launch_from_environment(
    *,
    engine: str | None = None,
    image_ref: str | None = None,
    image_digest: str | None = None,
    baseline_image_ref: str | None = None,
    baseline_image_digest: str | None = None,
    subject_image_ref: str | None = None,
    subject_image_digest: str | None = None,
    default_device: str | None = None,
    baseline_device: str | None = None,
    subject_device: str | None = None,
    runtime_entrypoint: str | None = None,
    baseline_entrypoint: str | None = None,
    subject_entrypoint: str | None = None,
    runtime_cpus: str | None = None,
    runtime_memory_mib: int | str | None = None,
    runtime_user: str | None = None,
) -> OciEvaluationLaunch:
    """Resolve common defaults and optional per-side OCI bindings."""

    common_image = image_ref or os.environ.get(RUNTIME_IMAGE_ENV, "")
    common_digest = image_digest or os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    common_device = _device(
        default_device or os.environ.get(RUNTIME_DEVICE_ENV, "cpu"),
        label="runtime device",
    )
    common_entrypoint = runtime_entrypoint or os.environ.get(
        RUNTIME_ENTRYPOINT_ENV, "auto"
    )

    for selected_image, selected_digest, digest_variable in (
        (
            baseline_image_ref
            or os.environ.get(BASELINE_RUNTIME_IMAGE_ENV)
            or common_image,
            baseline_image_digest
            or os.environ.get(BASELINE_RUNTIME_IMAGE_DIGEST_ENV)
            or common_digest,
            BASELINE_RUNTIME_IMAGE_DIGEST_ENV,
        ),
        (
            subject_image_ref
            or os.environ.get(SUBJECT_RUNTIME_IMAGE_ENV)
            or common_image,
            subject_image_digest
            or os.environ.get(SUBJECT_RUNTIME_IMAGE_DIGEST_ENV)
            or common_digest,
            SUBJECT_RUNTIME_IMAGE_DIGEST_ENV,
        ),
    ):
        if not selected_digest and not _embedded_digest(selected_image):
            raise OciEvaluationError(
                "runtime image digest is required through "
                f"{digest_variable} or {RUNTIME_IMAGE_DIGEST_ENV}"
            )

    selected_engine, engine_path = _engine_binary(
        engine or os.environ.get(CONTAINER_ENGINE_ENV, "docker")
    )

    def side_launch(
        *,
        role: RuntimeSideRole,
        explicit_image: str | None,
        image_variable: str,
        explicit_digest: str | None,
        digest_variable: str,
        explicit_device: str | None,
        device_variable: str,
        explicit_entrypoint: str | None,
        entrypoint_variable: str,
    ) -> OciSideLaunch:
        selected_image = (
            explicit_image or os.environ.get(image_variable) or common_image
        )
        selected_digest = (
            explicit_digest
            or os.environ.get(digest_variable)
            or _embedded_digest(selected_image)
            or common_digest
        )
        return OciSideLaunch(
            image_ref=_portable_image_ref(
                selected_image, selected_digest, engine_path=engine_path
            ),
            image_digest=selected_digest,
            device=_device(
                explicit_device or os.environ.get(device_variable) or common_device,
                label=f"{role} runtime device",
            ),
            entrypoint=_entrypoint(
                explicit_entrypoint
                or os.environ.get(entrypoint_variable)
                or common_entrypoint,
                label=f"{role} runtime entrypoint",
            ),
        )

    return OciEvaluationLaunch(
        engine=selected_engine,
        engine_path=engine_path,
        worker_limits=OciWorkerLimits(
            cpus=(
                runtime_cpus
                if runtime_cpus is not None
                else os.environ.get(RUNTIME_CPUS_ENV, _DEFAULT_WORKER_CPUS)
            ),
            memory_mib=_memory_limit_mib(
                runtime_memory_mib
                if runtime_memory_mib is not None
                else os.environ.get(
                    RUNTIME_MEMORY_MIB_ENV, str(_DEFAULT_WORKER_MEMORY_MIB)
                )
            ),
            user=(
                runtime_user
                if runtime_user is not None
                else os.environ.get(RUNTIME_USER_ENV, _DEFAULT_WORKER_USER)
            ),
        ),
        baseline=side_launch(
            role="baseline",
            explicit_image=baseline_image_ref,
            image_variable=BASELINE_RUNTIME_IMAGE_ENV,
            explicit_digest=baseline_image_digest,
            digest_variable=BASELINE_RUNTIME_IMAGE_DIGEST_ENV,
            explicit_device=baseline_device,
            device_variable=BASELINE_RUNTIME_DEVICE_ENV,
            explicit_entrypoint=baseline_entrypoint,
            entrypoint_variable=BASELINE_RUNTIME_ENTRYPOINT_ENV,
        ),
        subject=side_launch(
            role="subject",
            explicit_image=subject_image_ref,
            image_variable=SUBJECT_RUNTIME_IMAGE_ENV,
            explicit_digest=subject_image_digest,
            digest_variable=SUBJECT_RUNTIME_IMAGE_DIGEST_ENV,
            explicit_device=subject_device,
            device_variable=SUBJECT_RUNTIME_DEVICE_ENV,
            explicit_entrypoint=subject_entrypoint,
            entrypoint_variable=SUBJECT_RUNTIME_ENTRYPOINT_ENV,
        ),
    )


def _directory_mount_source(path: Path, *, label: str) -> Path:
    try:
        entry = path.lstat()
    except OSError as exc:
        raise OciEvaluationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise OciEvaluationError(f"{label} must be a directory, not a symlink")
    resolved = path.resolve(strict=True)
    if "," in str(resolved) or any(ord(character) < 32 for character in str(resolved)):
        raise OciEvaluationError(f"{label} path cannot be represented as an OCI mount")
    return resolved


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


def _mount(source: Path, target: str, *, read_only: bool) -> list[str]:
    fields = ["type=bind", f"source={source}", f"target={target}"]
    if read_only:
        fields.append("readonly")
    return ["--mount", ",".join(fields)]


def _worker_readable_artifact_mount_source(
    path: Path, *, user: str, label: str
) -> Path:
    resolved = _artifact_mount_source(path, label=label)
    _assert_worker_readable(resolved, user=user, label=label)
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


def _provider_bindings(
    environment: Mapping[str, str],
) -> dict[str, ProviderResourceBinding]:
    bindings: dict[str, ProviderResourceBinding] = {}
    for profile in OPTIONAL_RUNTIME_PROVIDER_PROFILES.values():
        provider_name = profile.provider_name
        root_variable = profile.resource_root_environment
        support_variables = dict(profile.support_resource_environment)
        root_value = environment.get(root_variable)
        support = {
            name: value
            for name, variable in support_variables.items()
            if (value := environment.get(variable)) is not None
        }
        if root_value is None:
            if support:
                raise OciEvaluationError(
                    f"{root_variable} is required when provider support is configured"
                )
            continue
        missing = sorted(set(support_variables) - set(support))
        if missing:
            raise OciEvaluationError(
                f"{support_variables[missing[0]]} is required for the configured provider"
            )
        root = _directory_mount_source(
            Path(root_value), label=f"{provider_name} resource root"
        )
        bindings[provider_name] = ProviderResourceBinding(
            root=root, support_resources=support
        )
    return bindings


def _provider_binding_environment_snapshot(
    environment: Mapping[str, str],
) -> MappingProxyType[str, str]:
    names: set[str] = set()
    for profile in OPTIONAL_RUNTIME_PROVIDER_PROFILES.values():
        names.add(profile.resource_root_environment)
        names.update(dict(profile.support_resource_environment).values())
    return MappingProxyType(
        {name: environment[name] for name in sorted(names) if name in environment}
    )


def _gpu_arguments(engine: ContainerEngine, device: str) -> list[str]:
    if device == "cpu":
        return []
    index = device.split(":", 1)[1] if ":" in device else None
    if engine == "docker":
        return ["--gpus", "all" if index is None else f"device={index}"]
    return [
        "--device",
        "nvidia.com/gpu=all" if index is None else f"nvidia.com/gpu={index}",
    ]


def _inner_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


def _workers_may_run_parallel(launch: OciEvaluationLaunch) -> bool:
    """Allow concurrency only when the two workers cannot contend for one GPU."""

    baseline = launch.baseline.device
    subject = launch.subject.device
    if baseline == subject == "cpu":
        return True
    if baseline.startswith("cuda:") and subject.startswith("cuda:"):
        return baseline.split(":", 1)[1] != subject.split(":", 1)[1]
    return False


def preflight_oci_launch(launch: OciEvaluationLaunch) -> dict[str, str]:
    """Confirm both pinned images are local without starting a container."""

    engine_path = launch.engine_path
    if engine_path is None:
        _engine, engine_path = _engine_binary(launch.engine)
    observed: dict[str, str] = {}
    inspected: dict[str, _LocalImageInspection] = {}
    for role in _ROLES:
        side = getattr(launch, role)
        image = side.image_ref
        inspection = inspected.get(image)
        if inspection is None:
            inspection = _inspect_local_image(engine_path, image)
            inspected[image] = inspection
        try:
            selected = _resolve_inspected_image(
                image,
                side.image_digest,
                inspection,
                allow_tag=False,
            )
        except OciEvaluationError as exc:
            raise OciEvaluationError(
                f"{role} runtime image is unavailable at its pinned digest"
            ) from exc
        if selected != image:
            raise OciEvaluationError(
                f"{role} runtime image execution reference is not immutable"
            )
        # Return the caller-declared identity. Repository manifests and local
        # config IDs are distinct namespaces even when they identify one image.
        observed[role] = side.image_digest
    return observed


def _worker_tmpfs_size_gib(*, provider_name: str, artifact_source: Path) -> int:
    """Size bounded worker scratch, including TensorRT engine snapshots."""

    profile = OPTIONAL_RUNTIME_PROVIDER_PROFILES.get(provider_name)
    if profile is None or profile.scratch_profile == "default":
        return _DEFAULT_WORKER_TMPFS_GIB
    source = _directory_mount_source(artifact_source, label="TensorRT engine bundle")
    total = 0
    try:
        for root, directories, files in os.walk(source, followlinks=False):
            root_path = Path(root)
            for name in (*directories, *files):
                facts = (root_path / name).lstat()
                if stat.S_ISLNK(facts.st_mode):
                    raise OciEvaluationError(
                        "TensorRT engine bundle must not contain symbolic links"
                    )
                if stat.S_ISREG(facts.st_mode):
                    total += facts.st_size
                elif not stat.S_ISDIR(facts.st_mode):
                    raise OciEvaluationError(
                        "TensorRT engine bundle contains an unsupported entry"
                    )
    except OSError as exc:
        raise OciEvaluationError(
            "TensorRT engine bundle size could not be determined"
        ) from exc
    required = total * _TENSORRT_ENGINE_COPY_FACTOR + _TENSORRT_SCRATCH_RESERVE_BYTES
    size_gib = max(
        _DEFAULT_WORKER_TMPFS_GIB,
        (required + _GIBIBYTE - 1) // _GIBIBYTE,
    )
    if size_gib > _MAX_WORKER_TMPFS_GIB:
        raise OciEvaluationError(
            "TensorRT engine bundle exceeds the bounded worker scratch capacity"
        )
    return size_gib


def compose_side_worker_command(
    *,
    launch: OciEvaluationLaunch,
    side_launch: OciSideLaunch,
    provider_name: str,
    artifact_source: Path,
    support_sources: Mapping[str, Path],
    job_root: Path,
    output_root: Path | None = None,
) -> list[str]:
    """Compose one argv-only model-worker launch with no signing-key mount."""

    profile = side_launch.entrypoint
    if profile == "auto":
        provider_profile = OPTIONAL_RUNTIME_PROVIDER_PROFILES.get(provider_name)
        profile = (
            provider_profile.automatic_entrypoint
            if provider_profile is not None
            else "python"
        )
    output_parent = job_root / "output-root" if output_root is None else output_root
    cidfile = job_root / "container.cid"
    if cidfile.exists() or cidfile.is_symlink():
        raise OciEvaluationError("worker container ID destination already exists")
    tmpfs_size_gib = _worker_tmpfs_size_gib(
        provider_name=provider_name,
        artifact_source=artifact_source,
    )
    worker_uid = launch.worker_limits.user.partition(":")[0]
    command = [
        launch.engine_path or launch.engine,
        "run",
        "--rm",
        "--init",
        "--pull=never",
        "--cidfile",
        str(cidfile),
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
        launch.worker_limits.cpus,
        "--memory",
        f"{launch.worker_limits.memory_mib}m",
        "--user",
        launch.worker_limits.user,
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={tmpfs_size_gib}g",
        *_gpu_arguments(launch.engine, side_launch.device),
        *_mount(
            _worker_readable_artifact_mount_source(
                artifact_source,
                user=launch.worker_limits.user,
                label="side artifact",
            ),
            "/invarlock-resources/artifact",
            read_only=True,
        ),
        *_mount(job_root, "/invarlock/job", read_only=True),
        *_mount(output_parent, "/invarlock/output-root", read_only=False),
        "-e",
        f"{CONTAINER_EXECUTION_ENV}=1",
        "-e",
        f"{RUNTIME_IMAGE_ENV}={side_launch.image_ref}",
        "-e",
        f"{RUNTIME_IMAGE_DIGEST_ENV}={side_launch.image_digest}",
        "-e",
        f"{ALLOW_NETWORK_ENV}=0",
        "-e",
        f"{ALLOW_REMOTE_CODE_ENV}=0",
        "-e",
        f"{ALLOW_THIRD_PARTY_PLUGINS_ENV}=0",
        "-e",
        "HOME=/tmp",
        "-e",
        "HF_HOME=/tmp/huggingface",
        "-e",
        f"LOGNAME={worker_uid}",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "-e",
        "TRITON_CACHE_DIR=/tmp/triton",
        "-e",
        f"USER={worker_uid}",
    ]
    for index, (_name, source) in enumerate(sorted(support_sources.items())):
        command.extend(
            _mount(
                _worker_readable_artifact_mount_source(
                    source,
                    user=launch.worker_limits.user,
                    label="side support resource",
                ),
                f"/invarlock-resources/support-{index}",
                read_only=True,
            )
        )
    if profile == "nvidia":
        command.extend(
            [
                "--entrypoint",
                "/opt/nvidia/nvidia_entrypoint.sh",
                side_launch.image_ref,
                "/opt/invarlock/cli-venv/bin/python",
            ]
        )
    else:
        command.extend(["--entrypoint", "python", side_launch.image_ref])
    command.extend(
        ["-m", "invarlock.evaluation_side_worker", "/invarlock/job/job.json"]
    )
    return command


def _read_bounded_stream(
    stream: object, destination: bytearray, stop: threading.Event
) -> None:
    """Drain one worker pipe while retaining only bounded diagnostics."""

    reader = getattr(stream, "read", None)
    if not callable(reader):
        return
    fileno = getattr(stream, "fileno", None)
    if callable(fileno):
        try:
            os.set_blocking(fileno(), False)
        except (OSError, ValueError):
            pass
    while not stop.is_set():
        try:
            chunk = reader(8192)
        except BlockingIOError:
            stop.wait(_WORKER_DRAIN_POLL_SECONDS)
            continue
        except (OSError, ValueError):
            return
        if chunk is None:
            stop.wait(_WORKER_DRAIN_POLL_SECONDS)
            continue
        if not chunk:
            return
        if len(destination) < _MAX_WORKER_DIAGNOSTIC_BYTES:
            destination.extend(chunk[: _MAX_WORKER_DIAGNOSTIC_BYTES - len(destination)])


def _join_worker_drains(drains: Sequence[threading.Thread]) -> None:
    deadline = time.monotonic() + _WORKER_DRAIN_JOIN_SECONDS
    for drain in drains:
        drain.join(timeout=max(0.0, deadline - time.monotonic()))


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


def _worker_container_handle(
    command: Sequence[str], cidfile: Path | None
) -> str | None:
    del command
    return _read_worker_container_id(cidfile)


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


def _terminate_worker(
    process: subprocess.Popen[bytes],
    command: Sequence[str],
    cidfile: Path | None,
) -> None:
    container_handle = _worker_container_handle(command, cidfile)
    if container_handle is not None and command:
        _container_control(command[0], "stop", container_handle)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_CONTAINER_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_CONTAINER_STOP_SECONDS)
    # The engine may create its cidfile immediately before it exits in response
    # to termination. Re-read only after reaping the launcher so cancellation
    # cannot discard the sole handle to a still-running container.
    late_container_handle = _worker_container_handle(command, cidfile)
    if late_container_handle is not None and command:
        if late_container_handle != container_handle:
            _container_control(command[0], "stop", late_container_handle)
        _container_control(command[0], "kill", late_container_handle)


def run_side_worker(
    command: Sequence[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run one precomposed worker with bounded logs and a hard outer deadline."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_OUTER_WORKER_TIMEOUT_SECONDS
    ):
        raise OciEvaluationError("worker outer timeout is invalid")

    cidfile = _worker_cidfile(command)
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        raise OciEvaluationError(
            f"container engine could not be started: {exc}"
        ) from exc
    stdout = bytearray()
    stderr = bytearray()
    stop_drains = threading.Event()
    assert process.stdout is not None
    assert process.stderr is not None
    drains = (
        threading.Thread(
            target=_read_bounded_stream,
            args=(process.stdout, stdout, stop_drains),
            name="invarlock-worker-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded_stream,
            args=(process.stderr, stderr, stop_drains),
            name="invarlock-worker-stderr",
            daemon=True,
        ),
    )
    for drain in drains:
        drain.start()
    timed_out = False
    try:
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_worker(process, command, cidfile)
            returncode = 124
        _join_worker_drains(drains)
    except BaseException:
        _terminate_worker(process, command, cidfile)
        raise
    finally:
        stop_drains.set()
        process.stdout.close()
        process.stderr.close()
        _join_worker_drains(drains)
        if cidfile is not None:
            cidfile.unlink(missing_ok=True)
    if timed_out:
        diagnostic = f"worker exceeded its {timeout_seconds}-second outer deadline"
        if stderr:
            stderr.extend(b"\n")
        stderr.extend(diagnostic.encode("utf-8"))
    return subprocess.CompletedProcess(
        list(command),
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _worker_outer_timeout_seconds(
    settings: Mapping[str, object],
    *,
    record_count: int,
) -> int:
    """Bound one worker from its validated per-record timeout and schedule size."""

    timeout = settings.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise OciEvaluationError(
            "runtime timeout_seconds must be a positive integer before OCI launch"
        )
    if record_count <= 0:
        raise OciEvaluationError("canonical schedule must contain at least one record")
    calculated = timeout * (record_count + _WORKER_TIMEOUT_ALLOWANCE_CALLS)
    return min(calculated, _MAX_OUTER_WORKER_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class OciRuntimeExecutor:
    """Host-side coordinator for two isolated model workers."""

    launch: OciEvaluationLaunch
    environment: Mapping[str, str] | None = None
    _provider_bindings_snapshot: Mapping[str, ProviderResourceBinding] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        observed = os.environ if self.environment is None else self.environment
        object.__setattr__(
            self,
            "_provider_bindings_snapshot",
            MappingProxyType(
                dict(
                    _provider_bindings(_provider_binding_environment_snapshot(observed))
                )
            ),
        )

    def resolve(
        self,
        *,
        request_root: Path,
        role: RuntimeSideRole,
        side: ComparisonSideRequest,
        provider: RuntimeProvider,
    ) -> RuntimeArtifactResources:
        """Resolve the exact side resources used by preflight and execution."""

        side_launch = getattr(self.launch, role)
        resolver = CallerRuntimeResources(
            container_image_digest=side_launch.image_digest,
            default_device=_inner_device(side_launch.device),
            provider_bindings=self._provider_bindings_snapshot,
        )
        return resolver.resolve(
            request_root=request_root,
            role=role,
            side=side,
            provider=provider,
        )

    def execute(
        self,
        request: EvaluationRequest,
        *,
        registry: CoreRegistry,
        schedule_bytes: bytes,
        policy_digest: str,
    ) -> EvaluationRunResult:
        try:
            schedule = parse_runtime_behavioral_schedule_json(
                schedule_bytes.decode("utf-8")
            )
        except (UnicodeError, ValueError) as exc:
            raise OciEvaluationError(
                "canonical schedule is invalid before OCI launch"
            ) from exc
        record_count = len(schedule.records)
        sides = {
            "baseline": (request.comparison.baseline, self.launch.baseline),
            "subject": (request.comparison.subject, self.launch.subject),
        }
        with tempfile.TemporaryDirectory(prefix="invarlock-side-workers-") as temporary:
            root = Path(temporary)
            commands: dict[RuntimeSideRole, list[str]] = {}
            outputs: dict[RuntimeSideRole, Path] = {}
            timeouts: dict[RuntimeSideRole, int] = {}
            for role, (side, side_launch) in sides.items():
                provider: RuntimeProvider = registry.get_runtime_provider(
                    side.runtime.provider
                )
                resources: RuntimeArtifactResources = self.resolve(
                    request_root=request.root,
                    role=cast(RuntimeSideRole, role),
                    side=side,
                    provider=provider,
                )
                job_root = root / role
                job_root.mkdir()
                job_root.joinpath("schedule.json").write_bytes(schedule_bytes)
                output_parent = root / f"{role}-output"
                output_parent.mkdir()
                output_parent.chmod(0o733)
                output = output_parent / "side"
                projected_support = {
                    name: f"support-{index}"
                    for index, name in enumerate(sorted(resources.support_resources))
                }
                support_sources = {
                    name: resources.support_path(name)
                    for name in sorted(resources.support_resources)
                }
                job = {
                    "format_version": "invarlock/runtime-side-job-v1",
                    "role": role,
                    "provider": provider.name,
                    "model_id": side.artifact.model_id,
                    "settings": dict(side.runtime.settings),
                    "metric": request.comparison.collection_metric,
                    "policy_digest": policy_digest,
                    "resource_root": "/invarlock-resources",
                    "primary_artifact": "artifact",
                    "support_resources": projected_support,
                    "device_kind": _inner_device(side_launch.device),
                    "image_digest": side_launch.image_digest,
                    "schedule": "/invarlock/job/schedule.json",
                    "output": "/invarlock/output-root/side",
                }
                job_root.joinpath("job.json").write_bytes(canonical_json_bytes(job))
                commands[cast(RuntimeSideRole, role)] = compose_side_worker_command(
                    launch=self.launch,
                    side_launch=side_launch,
                    provider_name=provider.name,
                    artifact_source=resources.primary_path(),
                    support_sources=support_sources,
                    job_root=job_root,
                    output_root=output_parent,
                )
                outputs[cast(RuntimeSideRole, role)] = output
                timeouts[cast(RuntimeSideRole, role)] = _worker_outer_timeout_seconds(
                    cast(Mapping[str, object], side.runtime.settings),
                    record_count=record_count,
                )

            completed: dict[RuntimeSideRole, subprocess.CompletedProcess[str]] = {}
            if _workers_may_run_parallel(self.launch):
                with ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="invarlock-side"
                ) as pool:
                    futures = {
                        role: pool.submit(
                            run_side_worker,
                            command,
                            timeout_seconds=timeouts[role],
                        )
                        for role, command in commands.items()
                    }
                    for role in _ROLES:
                        completed[role] = futures[role].result()
            else:
                for role in _ROLES:
                    completed[role] = run_side_worker(
                        commands[role], timeout_seconds=timeouts[role]
                    )
            failures = [
                f"{role} worker exited with status {completed[role].returncode}: "
                + (completed[role].stderr.strip() or "no diagnostic")
                for role in _ROLES
                if completed[role].returncode != 0
            ]
            if failures:
                raise OciEvaluationError("; ".join(failures))
            evidence = {
                role: load_runtime_side_evidence(outputs[role]) for role in _ROLES
            }
            digests = {
                role: decode_runtime_provider_receipt(
                    evidence[role].provider_receipt
                ).outer_image_digest
                for role in _ROLES
            }
            for role in _ROLES:
                expected = getattr(self.launch, role).image_digest
                if digests[role] != expected:
                    raise OciEvaluationError(
                        f"{role} worker receipt does not bind its selected runtime image"
                    )
            return EvaluationRunResult(
                baseline=evidence["baseline"],
                subject=evidence["subject"],
                baseline_runtime_digest=cast(str, digests["baseline"]),
                subject_runtime_digest=cast(str, digests["subject"]),
            )


__all__ = [
    "BASELINE_RUNTIME_DEVICE_ENV",
    "BASELINE_RUNTIME_ENTRYPOINT_ENV",
    "BASELINE_RUNTIME_IMAGE_DIGEST_ENV",
    "BASELINE_RUNTIME_IMAGE_ENV",
    "CONTAINER_ENGINE_ENV",
    "OciEvaluationError",
    "OciEvaluationLaunch",
    "OciRuntimeExecutor",
    "OciSideLaunch",
    "OciWorkerLimits",
    "RUNTIME_CPUS_ENV",
    "RUNTIME_DEVICE_ENV",
    "RUNTIME_ENTRYPOINT_ENV",
    "RUNTIME_MEMORY_MIB_ENV",
    "RUNTIME_USER_ENV",
    "SUBJECT_RUNTIME_DEVICE_ENV",
    "SUBJECT_RUNTIME_ENTRYPOINT_ENV",
    "SUBJECT_RUNTIME_IMAGE_DIGEST_ENV",
    "SUBJECT_RUNTIME_IMAGE_ENV",
    "compose_side_worker_command",
    "evaluation_request_execution_mode",
    "launch_from_environment",
    "preflight_oci_launch",
    "run_side_worker",
]
