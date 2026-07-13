from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from invarlock import runtime_security_paths as path_helpers
from invarlock.core import config_loader as _config_loader
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.public_contracts import RUNTIME_MANIFEST_CONTRACT_VERSION

inspect_config_dependencies = _config_loader.inspect_config_dependencies

ALLOW_HOST_EXECUTION_ENV = "INVARLOCK_ALLOW_HOST_EXECUTION"
ALLOW_NETWORK_ENV = "INVARLOCK_ALLOW_NETWORK"
ALLOW_REMOTE_CODE_ENV = "INVARLOCK_ALLOW_REMOTE_CODE"
ALLOW_THIRD_PARTY_PLUGINS_ENV = "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"
ALLOW_UNVERIFIED_PROVENANCE_ENV = "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE"
CONTAINER_EXECUTION_ENV = "INVARLOCK_CONTAINER_EXECUTION"
CONTAINER_ENGINE_ENV = "INVARLOCK_CONTAINER_ENGINE"
RUNTIME_IMAGE_ENV = "INVARLOCK_RUNTIME_IMAGE"
RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_RUNTIME_IMAGE_DIGEST"
SOURCE_BUNDLE_DIGEST_ENV = "INVARLOCK_SOURCE_BUNDLE_SHA256"
SOURCE_BUNDLE_READ_ONLY_ENV = "INVARLOCK_SOURCE_BUNDLE_READ_ONLY"
RUNTIME_MANIFEST_FILENAME = "runtime.manifest.json"
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_VERIFIER_CONTRACT_VERSION = RUNTIME_MANIFEST_CONTRACT_VERSION
RUNTIME_IMAGE_LOCAL_DEFAULT = "invarlock-runtime:local"
RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT = "invarlock-runtime:cuda-local"
RUNTIME_IMAGE_DEFAULT = "ghcr.io/invarlock/invarlock-runtime:latest"
_CONTAINER_INSPECT_TIMEOUT_SECONDS = 30
_CONTAINER_EXECUTION_TIMEOUT_SECONDS = 24 * 60 * 60
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
__all__ = [
    "ALLOW_HOST_EXECUTION_ENV",
    "ALLOW_NETWORK_ENV",
    "ALLOW_REMOTE_CODE_ENV",
    "ALLOW_THIRD_PARTY_PLUGINS_ENV",
    "ALLOW_UNVERIFIED_PROVENANCE_ENV",
    "ContainerLaunchPlan",
    "RuntimeManifestExecution",
    "RuntimeSecurityPolicy",
    "CONTAINER_EXECUTION_ENV",
    "CONTAINER_ENGINE_ENV",
    "RUNTIME_IMAGE_ENV",
    "RUNTIME_IMAGE_DIGEST_ENV",
    "SOURCE_BUNDLE_DIGEST_ENV",
    "SOURCE_BUNDLE_READ_ONLY_ENV",
    "RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT",
    "RUNTIME_MANIFEST_FILENAME",
    "RUNTIME_MANIFEST_VERSION",
    "RUNTIME_VERIFIER_CONTRACT_VERSION",
    "apply_runtime_allowances",
    "build_container_command",
    "build_container_python_command",
    "build_runtime_security_policy",
    "container_image_available_locally",
    "current_execution_mode",
    "current_runtime_security_policy",
    "delegate_container_command",
    "delegate_python_script_to_container",
    "network_allowed",
    "host_execution_allowed",
    "load_runtime_manifest",
    "remote_code_allowed",
    "resolve_container_engine",
    "resolve_runtime_image",
    "resolve_runtime_image_digest",
    "reset_runtime_allowances",
    "runtime_allowances_scope",
    "RuntimeManifestLoadIssueCode",
    "RuntimeManifestLoadResult",
    "running_inside_container",
    "third_party_plugins_allowed",
    "unverified_provenance_allowed",
    "write_runtime_manifest",
]


@dataclass(frozen=True)
class ContainerLaunchPlan:
    """Typed launch plan for delegated container execution."""

    argv: tuple[str, ...]
    argv_mounts: tuple[Path, ...]
    needs_cwd_host_mirror: bool
    gpu_passthrough: bool
    workspace_read_only: bool = False


@dataclass(frozen=True)
class RuntimeSecurityPolicy:
    """Typed runtime-policy snapshot for request-scoped application."""

    allow_network: bool = False
    allow_host_execution: bool = False
    allow_third_party_plugins: bool = False
    allow_remote_code: bool = False
    allow_unverified_provenance: bool = False


@dataclass(frozen=True)
class RuntimeManifestExecution:
    """Execution provenance recorded into a runtime manifest."""

    execution_mode: str
    container_execution: bool
    image_ref: str
    image_digest: str | None
    allow_network: bool
    allow_remote_code: bool
    allow_third_party_plugins: bool


class RuntimeManifestLoadIssueCode(StrEnum):
    MISSING = "missing"
    READ_FAILED = "read_failed"
    INVALID_JSON = "invalid_json"
    INVALID_PAYLOAD = "invalid_payload"


@dataclass(frozen=True)
class RuntimeManifestLoadResult:
    path: Path
    payload: dict[str, Any] | None
    issue_code: RuntimeManifestLoadIssueCode | None = None
    issue_message: str | None = None


@dataclass(frozen=True)
class _ContainerLaunchContext:
    engine: str
    cwd: Path
    image: str
    env_pairs: dict[str, str]
    base_mounts: tuple[Path, ...]


@dataclass(frozen=True)
class _ContainerImageInspection:
    """Identity fields observed from one local container-engine inspection."""

    image_id: str
    repo_digests: tuple[str, ...]


@dataclass(frozen=True)
class _ObservedContainerImage:
    """Immutable local identity selected for one container launch."""

    immutable_ref: str
    image_digest: str
    image_id: str
    repo_digests: tuple[str, ...]


_RUNTIME_SECURITY_POLICY: ContextVar[RuntimeSecurityPolicy | None] = ContextVar(
    "invarlock_runtime_security_policy",
    default=None,
)


def _coerce_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return None


def _runtime_flag_value(name: str) -> str | None:
    policy = _RUNTIME_SECURITY_POLICY.get()
    if policy is None:
        return os.environ.get(name)
    flag_map = {
        ALLOW_NETWORK_ENV: policy.allow_network,
        ALLOW_HOST_EXECUTION_ENV: policy.allow_host_execution,
        ALLOW_THIRD_PARTY_PLUGINS_ENV: policy.allow_third_party_plugins,
        ALLOW_REMOTE_CODE_ENV: policy.allow_remote_code,
        ALLOW_UNVERIFIED_PROVENANCE_ENV: policy.allow_unverified_provenance,
    }
    if name not in flag_map:
        return os.environ.get(name)
    return "1" if flag_map[name] else "0"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        normalized_items = [_json_safe(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def serialize_canonical_json(payload: Any) -> str:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(read_regular_file_bytes(path, label="runtime manifest input"))


def network_allowed() -> bool:
    return _coerce_bool(_runtime_flag_value(ALLOW_NETWORK_ENV)) is True


def host_execution_allowed() -> bool:
    return _coerce_bool(_runtime_flag_value(ALLOW_HOST_EXECUTION_ENV)) is True


def remote_code_allowed() -> bool:
    return _coerce_bool(_runtime_flag_value(ALLOW_REMOTE_CODE_ENV)) is True


def unverified_provenance_allowed() -> bool:
    return _coerce_bool(_runtime_flag_value(ALLOW_UNVERIFIED_PROVENANCE_ENV)) is True


def third_party_plugins_allowed() -> bool:
    explicit = _coerce_bool(_runtime_flag_value(ALLOW_THIRD_PARTY_PLUGINS_ENV))
    return explicit is True


def running_inside_container() -> bool:
    return _coerce_bool(os.environ.get(CONTAINER_EXECUTION_ENV)) is True


def current_execution_mode() -> str:
    return "container" if running_inside_container() else "host-bypass"


def resolve_runtime_image() -> str:
    image = os.environ.get(RUNTIME_IMAGE_ENV, "").strip()
    if image:
        return image
    engine = resolve_container_engine()
    if (
        engine is not None
        and _host_nvidia_visible()
        and container_image_available_locally(
            RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT, engine=engine
        )
    ):
        return RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT
    if engine is not None and container_image_available_locally(
        RUNTIME_IMAGE_LOCAL_DEFAULT, engine=engine
    ):
        return RUNTIME_IMAGE_LOCAL_DEFAULT
    return RUNTIME_IMAGE_DEFAULT


def resolve_runtime_image_digest() -> str | None:
    explicit = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "").strip()
    if explicit:
        return explicit
    image = resolve_runtime_image()
    if "@sha256:" in image:
        return image.split("@", 1)[1]
    engine = resolve_container_engine()
    if engine is None:
        return None
    _, digest = _inspect_container_image(engine, image)
    return digest


def _declared_runtime_image_digest(image: str) -> str | None:
    explicit_raw = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "").strip()
    embedded_raw = image.split("@", 1)[1] if "@" in image else ""
    explicit = explicit_raw.lower()
    embedded = embedded_raw.lower()
    if explicit and (
        explicit != explicit_raw or not _SHA256_DIGEST_RE.fullmatch(explicit)
    ):
        raise RuntimeError(
            f"{RUNTIME_IMAGE_DIGEST_ENV} must be lowercase sha256:<64 hex>."
        )
    if embedded and (
        embedded != embedded_raw or not _SHA256_DIGEST_RE.fullmatch(embedded)
    ):
        raise RuntimeError(
            "The runtime image digest must be lowercase sha256:<64 hex>."
        )
    if explicit and embedded and explicit != embedded:
        raise RuntimeError(
            "The declared runtime image digest does not match the image reference."
        )
    return explicit or embedded or None


def _runtime_provenance_image_ref(image_ref: str, image_digest: str | None) -> str:
    if image_ref in {
        RUNTIME_IMAGE_LOCAL_DEFAULT,
        RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT,
    }:
        return image_ref
    if "@sha256:" in image_ref:
        return image_ref
    if _SHA256_DIGEST_RE.fullmatch(image_ref):
        return image_ref
    if image_digest:
        return f"{image_ref}@{image_digest}"
    if unverified_provenance_allowed():
        return image_ref
    raise RuntimeError(
        "Container-backed runtime manifests require a digest-pinned runtime image; "
        f"set {RUNTIME_IMAGE_DIGEST_ENV}, use {RUNTIME_IMAGE_LOCAL_DEFAULT!r}, "
        "or allow unverified provenance explicitly."
    )


def _inspect_container_image_identity(
    engine: str, image: str
) -> _ContainerImageInspection | None:
    try:
        completed = subprocess.run(
            [
                engine,
                "image",
                "inspect",
                image,
                "--format",
                "{{json .RepoDigests}}\n{{.Id}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_CONTAINER_INSPECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    repo_digests: list[str] = []
    if lines:
        try:
            payload = parse_json_bytes(
                lines[0].encode("utf-8"), label="container inspect output"
            )
        except StrictJsonError:
            payload = None
        if isinstance(payload, list):
            repo_digests = sorted(
                {
                    str(item).strip().lower()
                    for item in payload
                    if isinstance(item, str)
                    and "@" in item
                    and _SHA256_DIGEST_RE.fullmatch(
                        str(item).rsplit("@", 1)[1].strip().lower()
                    )
                }
            )
    image_id = lines[1].lower() if len(lines) >= 2 else ""
    if not _SHA256_DIGEST_RE.fullmatch(image_id):
        return None
    return _ContainerImageInspection(
        image_id=image_id,
        repo_digests=tuple(repo_digests),
    )


def _resolve_observed_container_image(
    engine: str, image: str
) -> _ObservedContainerImage:
    inspection = _inspect_container_image_identity(engine, image)
    if inspection is None:
        raise RuntimeError(
            "The selected runtime image is not available for immutable local "
            "identity inspection. Pull or build it before delegated execution."
        )
    declared = _declared_runtime_image_digest(image)
    repo_refs_by_digest = {
        reference.rsplit("@", 1)[1]: reference for reference in inspection.repo_digests
    }
    if declared is not None:
        if declared == inspection.image_id:
            immutable_ref = inspection.image_id
        elif declared in repo_refs_by_digest:
            immutable_ref = repo_refs_by_digest[declared]
        else:
            observed = sorted({inspection.image_id, *repo_refs_by_digest})
            raise RuntimeError(
                "The declared runtime image digest does not match the observed "
                f"local image identity (observed={observed!r})."
            )
        selected_digest = declared
    elif repo_refs_by_digest:
        selected_digest = sorted(repo_refs_by_digest)[0]
        immutable_ref = repo_refs_by_digest[selected_digest]
    else:
        selected_digest = inspection.image_id
        immutable_ref = inspection.image_id
    return _ObservedContainerImage(
        immutable_ref=immutable_ref,
        image_digest=selected_digest,
        image_id=inspection.image_id,
        repo_digests=inspection.repo_digests,
    )


def _inspect_container_image(engine: str, image: str) -> tuple[bool, str | None]:
    inspection = _inspect_container_image_identity(engine, image)
    if inspection is None:
        return False, None
    if inspection.repo_digests:
        return True, inspection.repo_digests[0].rsplit("@", 1)[1]
    return True, inspection.image_id


def _runtime_image_build_command(image: str) -> str:
    if image == RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT:
        return "make runtime-image-cuda"
    return "make runtime-image"


def resolve_container_engine() -> str | None:
    explicit = os.environ.get(CONTAINER_ENGINE_ENV, "").strip().lower()
    if explicit:
        if explicit in {"docker", "podman"} and shutil.which(explicit):
            return explicit
        return None
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    return None


_CONFIG_SCAN_ARG_FLAGS = path_helpers._CONFIG_SCAN_ARG_FLAGS
_CONFIG_PATH_ARG_FLAGS = path_helpers._CONFIG_PATH_ARG_FLAGS
_OUTPUT_PATH_ARG_FLAGS = path_helpers._OUTPUT_PATH_ARG_FLAGS
_LOCAL_MODEL_ARG_FLAGS = path_helpers._LOCAL_MODEL_ARG_FLAGS
_PATH_ENV_VARS = path_helpers._PATH_ENV_VARS
_FORWARDED_ENV_VARS = path_helpers._FORWARDED_ENV_VARS
_host_nvidia_visible = path_helpers._host_nvidia_visible
_minimize_mounts = path_helpers._minimize_mounts
_absolute_host_path = path_helpers._absolute_host_path
_path_is_within = path_helpers._path_is_within
_workspace_path = path_helpers._workspace_path
_mount_root_for_path = path_helpers._mount_root_for_path
_mount_root_for_resolved_path = path_helpers._mount_root_for_resolved_path
_mount_is_already_covered = path_helpers._mount_is_already_covered
_iter_external_symlink_target_mounts = path_helpers._iter_external_symlink_target_mounts
_iter_absolute_pythonpath_entries = path_helpers._iter_absolute_pythonpath_entries
_container_pythonpath_entries = path_helpers._container_pythonpath_entries
_record_path_dependencies = path_helpers._record_path_dependencies
_normalize_output_path_for_container = path_helpers._normalize_output_path_for_container
_normalize_local_model_path_for_container = (
    path_helpers._normalize_local_model_path_for_container
)
_normalize_config_path_for_container = path_helpers._normalize_config_path_for_container
_path_env_value_for_container = path_helpers._path_env_value_for_container
_delegated_env_pairs = path_helpers._delegated_env_pairs
_merge_container_mounts = path_helpers._merge_container_mounts
_resolve_container_launch_context = path_helpers._resolve_container_launch_context
_compose_container_run_args = path_helpers._compose_container_run_args


def container_image_available_locally(
    image: str | None = None, *, engine: str | None = None
) -> bool:
    resolved_engine = engine or resolve_container_engine()
    if resolved_engine is None:
        return False
    resolved_image = image or resolve_runtime_image()
    exists, _ = _inspect_container_image(resolved_engine, resolved_image)
    return exists


def build_runtime_security_policy(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> RuntimeSecurityPolicy:
    return RuntimeSecurityPolicy(
        allow_network=bool(allow_network),
        allow_host_execution=bool(allow_host_execution),
        allow_third_party_plugins=bool(allow_third_party_plugins),
        allow_remote_code=bool(allow_remote_code),
        allow_unverified_provenance=bool(allow_unverified_provenance),
    )


def current_runtime_security_policy() -> RuntimeSecurityPolicy | None:
    return _RUNTIME_SECURITY_POLICY.get()


def _resolve_runtime_security_policy(
    *,
    policy: RuntimeSecurityPolicy | None = None,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> RuntimeSecurityPolicy:
    if policy is not None:
        return policy
    return build_runtime_security_policy(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
    )


def _apply_runtime_security_policy(
    resolved_policy: RuntimeSecurityPolicy,
) -> Token[RuntimeSecurityPolicy | None]:
    token = _RUNTIME_SECURITY_POLICY.set(resolved_policy)
    try:
        from invarlock.security import enforce_network_policy

        enforce_network_policy(bool(resolved_policy.allow_network))
    except RuntimeError:
        _RUNTIME_SECURITY_POLICY.reset(token)
        raise
    return token


def apply_runtime_allowances(
    *,
    policy: RuntimeSecurityPolicy | None = None,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> Token[RuntimeSecurityPolicy | None]:
    resolved_policy = _resolve_runtime_security_policy(
        policy=policy,
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    return _apply_runtime_security_policy(resolved_policy)


def reset_runtime_allowances(
    token: Token[RuntimeSecurityPolicy | None] | None = None,
) -> None:
    if token is None:
        _RUNTIME_SECURITY_POLICY.set(None)
    else:
        _RUNTIME_SECURITY_POLICY.reset(token)

    from invarlock.security import enforce_network_policy

    enforce_network_policy(network_allowed())


@contextmanager
def runtime_allowances_scope(
    *,
    policy: RuntimeSecurityPolicy | None = None,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> Iterator[RuntimeSecurityPolicy]:
    resolved_policy = _resolve_runtime_security_policy(
        policy=policy,
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    token = apply_runtime_allowances(policy=resolved_policy)
    try:
        yield resolved_policy
    finally:
        reset_runtime_allowances(token)


def build_container_command(plan: Any) -> list[str]:
    context = _resolve_container_launch_context(plan)
    command: list[str] = _compose_container_run_args(
        context,
        plan,
        argv=tuple(plan.argv),
    )
    return command


def delegate_container_command(plan: Any) -> int:
    command = build_container_command(plan)
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=_CONTAINER_EXECUTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Container execution timed out after "
            f"{_CONTAINER_EXECUTION_TIMEOUT_SECONDS} seconds."
        ) from exc
    return int(completed.returncode)


def build_container_python_command(
    script_path: str | os.PathLike[str],
    plan: Any,
) -> list[str]:
    context = _resolve_container_launch_context(plan)
    cwd = context.cwd
    script_host_path = _absolute_host_path(Path(script_path), cwd=cwd)
    script_mounts: set[Path] = set()
    if _record_path_dependencies(script_host_path, script_mounts, cwd=cwd):
        container_script = _workspace_path(script_host_path, cwd=cwd)
    else:
        container_script = str(script_host_path)
    command: list[str] = _compose_container_run_args(
        context,
        plan,
        entrypoint=("--entrypoint", "python"),
        extra_mounts=tuple(script_mounts),
        argv=(container_script, *plan.argv),
    )
    return command


def build_container_python_module_command(
    module_name: str,
    plan: Any,
) -> list[str]:
    context = _resolve_container_launch_context(plan)
    command: list[str] = _compose_container_run_args(
        context,
        plan,
        entrypoint=("--entrypoint", "python"),
        argv=("-m", module_name, *plan.argv),
    )
    return command


def delegate_python_script_to_container(
    script_path: str | os.PathLike[str],
    plan: Any,
) -> int:
    command = build_container_python_command(script_path, plan)
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=_CONTAINER_EXECUTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Container execution timed out after "
            f"{_CONTAINER_EXECUTION_TIMEOUT_SECONDS} seconds."
        ) from exc
    return int(completed.returncode)


def delegate_python_module_to_container(
    module_name: str,
    plan: Any,
) -> int:
    command = build_container_python_module_command(module_name, plan)
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=_CONTAINER_EXECUTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Container execution timed out after "
            f"{_CONTAINER_EXECUTION_TIMEOUT_SECONDS} seconds."
        ) from exc
    return int(completed.returncode)


def _config_digest(
    *,
    config_path: str | os.PathLike[str] | None = None,
    config_payload: Any | None = None,
) -> tuple[str | None, str]:
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            return _sha256_path(path), "file"
    if config_payload is not None:
        payload = serialize_canonical_json(config_payload).encode("utf-8")
        return _sha256_bytes(payload), "inline"
    return None, "missing"


def write_runtime_manifest(
    report_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    config_payload: Any | None = None,
    extra: dict[str, Any] | None = None,
    execution: RuntimeManifestExecution | None = None,
) -> Path:
    report = Path(report_path).resolve()
    digest, digest_source = _config_digest(
        config_path=config_path, config_payload=config_payload
    )
    runtime_execution = execution or RuntimeManifestExecution(
        execution_mode=current_execution_mode(),
        container_execution=running_inside_container(),
        image_ref=resolve_runtime_image(),
        image_digest=resolve_runtime_image_digest(),
        allow_network=network_allowed(),
        allow_remote_code=remote_code_allowed(),
        allow_third_party_plugins=third_party_plugins_allowed(),
    )
    manifest: dict[str, Any] = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            # Runtime manifests travel with their sibling report into evidence
            # packs and public archives.  Record the sibling name instead of a
            # machine-local absolute path so the binding remains portable and
            # does not disclose the producer filesystem.
            "path": report.name,
            "filename": report.name,
            "sha256": _sha256_path(report),
        },
        "config": {
            "path": Path(config_path).name if config_path is not None else None,
            "sha256": digest,
            "source": digest_source,
        },
        "execution_mode": runtime_execution.execution_mode,
        "runtime": {
            "image_ref": _runtime_provenance_image_ref(
                runtime_execution.image_ref,
                runtime_execution.image_digest,
            ),
            "image_digest": runtime_execution.image_digest,
            "container_execution": runtime_execution.container_execution,
            "allow_network": runtime_execution.allow_network,
            "allow_remote_code": runtime_execution.allow_remote_code,
            "allow_third_party_plugins": runtime_execution.allow_third_party_plugins,
        },
    }
    context = dict(extra) if isinstance(extra, dict) else {}
    source_bundle_digest = os.environ.get(SOURCE_BUNDLE_DIGEST_ENV, "").strip()
    source_bundle_read_only = _coerce_bool(os.environ.get(SOURCE_BUNDLE_READ_ONLY_ENV))
    if source_bundle_digest or source_bundle_read_only is not None:
        if (
            not _SHA256_DIGEST_RE.fullmatch(source_bundle_digest)
            or source_bundle_read_only is not True
        ):
            raise RuntimeError(
                "Source-bundle provenance requires a lowercase sha256 digest "
                "and a read-only delegated workspace."
            )
        context["source_bundle"] = {
            "read_only": True,
            "sha256": source_bundle_digest,
        }
    if context:
        manifest["context"] = _json_safe(context)
    manifest_path = report.parent / RUNTIME_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_runtime_manifest(
    report_path: str | os.PathLike[str],
) -> RuntimeManifestLoadResult:
    report = Path(report_path)
    manifest_path = report.parent / RUNTIME_MANIFEST_FILENAME
    if not manifest_path.exists():
        return RuntimeManifestLoadResult(
            path=manifest_path,
            payload=None,
            issue_code=RuntimeManifestLoadIssueCode.MISSING,
        )
    try:
        raw = read_regular_file_bytes(manifest_path, label=RUNTIME_MANIFEST_FILENAME)
    except StrictJsonError:
        return RuntimeManifestLoadResult(
            path=manifest_path,
            payload=None,
            issue_code=RuntimeManifestLoadIssueCode.READ_FAILED,
            issue_message=f"unable to read {manifest_path.name}",
        )
    try:
        payload = parse_json_bytes(raw, label=RUNTIME_MANIFEST_FILENAME)
    except StrictJsonError:
        return RuntimeManifestLoadResult(
            path=manifest_path,
            payload=None,
            issue_code=RuntimeManifestLoadIssueCode.INVALID_JSON,
            issue_message=f"{manifest_path.name} is not valid JSON",
        )
    if not isinstance(payload, dict):
        return RuntimeManifestLoadResult(
            path=manifest_path,
            payload=None,
            issue_code=RuntimeManifestLoadIssueCode.INVALID_PAYLOAD,
            issue_message=f"{manifest_path.name} must decode to a JSON object",
        )
    return RuntimeManifestLoadResult(path=manifest_path, payload=payload)
