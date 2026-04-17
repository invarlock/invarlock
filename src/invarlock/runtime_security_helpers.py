from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

# Canonical runtime-security helpers live here; container execution and manifest
# serialization are split into owner modules imported below.
from invarlock import runtime_security_container as _container_impl
from invarlock import runtime_security_manifest as _manifest_impl
from invarlock import runtime_security_paths as path_helpers
from invarlock.core import config_loader as _config_loader

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
RUNTIME_MANIFEST_FILENAME = "runtime.manifest.json"
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_VERIFIER_CONTRACT_VERSION = "runtime-manifest-v1"
RUNTIME_IMAGE_LOCAL_DEFAULT = "invarlock-runtime:local"
RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT = "invarlock-runtime:cuda-local"
RUNTIME_IMAGE_DEFAULT = "ghcr.io/invarlock/invarlock-runtime:latest"
_CONTAINER_INSPECT_TIMEOUT_SECONDS = 30
_CONTAINER_EXECUTION_TIMEOUT_SECONDS = 24 * 60 * 60

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
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def serialize_canonical_json(payload: Any) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _runtime_provenance_image_ref(image_ref: str, image_digest: str | None) -> str:
    if image_ref in {
        RUNTIME_IMAGE_LOCAL_DEFAULT,
        RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT,
    }:
        return image_ref
    if "@sha256:" in image_ref:
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


def _inspect_container_image(engine: str, image: str) -> tuple[bool, str | None]:
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
        return False, None
    if completed.returncode != 0:
        return False, None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    repo_digests: list[str] = []
    if lines:
        try:
            payload = json.loads(lines[0])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            repo_digests = [str(item) for item in payload if isinstance(item, str)]
    for digest_ref in repo_digests:
        if "@sha256:" in digest_ref:
            return True, digest_ref.split("@", 1)[1]
    if len(lines) >= 2 and lines[1].startswith("sha256:"):
        return True, lines[1]
    return True, None


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
    return _container_impl.build_container_command(plan)


def delegate_container_command(plan: Any) -> int:
    return _container_impl.delegate_container_command(plan)


def build_container_python_command(
    script_path: str | os.PathLike[str],
    plan: Any,
) -> list[str]:
    return _container_impl.build_container_python_command(script_path, plan)


def build_container_python_module_command(
    module_name: str,
    plan: Any,
) -> list[str]:
    return _container_impl.build_container_python_module_command(module_name, plan)


def delegate_python_script_to_container(
    script_path: str | os.PathLike[str],
    plan: Any,
) -> int:
    return _container_impl.delegate_python_script_to_container(script_path, plan)


def delegate_python_module_to_container(
    module_name: str,
    plan: Any,
) -> int:
    return _container_impl.delegate_python_module_to_container(module_name, plan)


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
    return _manifest_impl.write_runtime_manifest(
        report_path,
        config_path=config_path,
        config_payload=config_payload,
        extra=extra,
        execution=execution,
    )


def load_runtime_manifest(
    report_path: str | os.PathLike[str],
) -> RuntimeManifestLoadResult:
    return cast(
        RuntimeManifestLoadResult,
        _manifest_impl.load_runtime_manifest(report_path),
    )
