"""Runtime-security owner API.

Container/manifest primitives live in `invarlock.runtime_security_helpers`;
delegated command launch planning lives here so CLI and config-entrypoint callers
share the same runtime boundary.

Typed request-scoped policy surface retained at the owner boundary:
- class RuntimeSecurityPolicy
- def build_runtime_security_policy(
- policy: RuntimeSecurityPolicy | None = None
- ContextVar(
- def reset_runtime_allowances(
"""

from __future__ import annotations

import os as _os
import sys
from collections.abc import Iterable
from pathlib import Path as _Path
from typing import Protocol

from invarlock.runtime_security_helpers import (
    ALLOW_HOST_EXECUTION_ENV,
    ALLOW_NETWORK_ENV,
    ALLOW_REMOTE_CODE_ENV,
    ALLOW_THIRD_PARTY_PLUGINS_ENV,
    ALLOW_UNVERIFIED_PROVENANCE_ENV,
    CONTAINER_ENGINE_ENV,
    CONTAINER_EXECUTION_ENV,
    RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT,
    RUNTIME_IMAGE_DEFAULT,
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
    RUNTIME_IMAGE_LOCAL_DEFAULT,
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
    SOURCE_BUNDLE_DIGEST_ENV,
    SOURCE_BUNDLE_READ_ONLY_ENV,
    ContainerLaunchPlan,
    RuntimeManifestExecution,
    RuntimeManifestLoadIssueCode,
    RuntimeManifestLoadResult,
    RuntimeSecurityPolicy,
    _host_nvidia_visible,
    _minimize_mounts,
    _normalize_config_path_for_container,
    _normalize_local_model_path_for_container,
    _normalize_output_path_for_container,
    apply_runtime_allowances,
    build_container_command,
    build_container_python_command,
    build_container_python_module_command,
    build_runtime_security_policy,
    container_image_available_locally,
    current_execution_mode,
    current_runtime_security_policy,
    delegate_container_command,
    delegate_python_module_to_container,
    delegate_python_script_to_container,
    host_execution_allowed,
    load_runtime_manifest,
    network_allowed,
    remote_code_allowed,
    reset_runtime_allowances,
    resolve_container_engine,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    running_inside_container,
    runtime_allowances_scope,
    third_party_plugins_allowed,
    unverified_provenance_allowed,
    write_runtime_manifest,
)


class SupportsContainerLaunchRequest(Protocol):
    def to_internal_argv(self, command_name: str | Iterable[str]) -> list[str]: ...


_CONFIG_SCAN_ARG_FLAGS = {"--config", "-c", "--preset", "--edit-config"}
_CONFIG_PATH_ARG_FLAGS = _CONFIG_SCAN_ARG_FLAGS | {"--baseline-report"}
_OUTPUT_PATH_ARG_FLAGS = {"--out", "--report-out"}
_LOCAL_MODEL_ARG_FLAGS = {"--baseline", "--subject"}


def _command_tokens(argv: list[str]) -> list[str]:
    return [token for token in argv if not token.startswith("-")]


def _leading_command_path(argv: list[str]) -> tuple[str, ...]:
    path: list[str] = []
    for token in argv:
        if token.startswith("-"):
            break
        path.append(token)
    return tuple(path)


def _requested_device(argv: list[str]) -> str | None:
    if "--device" in argv:
        idx = argv.index("--device")
        if idx + 1 < len(argv):
            return str(argv[idx + 1]).strip().lower()
        return None

    command_path = _leading_command_path(argv)
    if not command_path:
        return None
    if command_path[0] in {"evaluate", "run", "calibrate"}:
        return "auto"
    if command_path[:2] == ("advanced", "calibrate"):
        return "auto"
    return None


def _iter_flag_occurrences(
    argv: list[str], *, flags: set[str]
) -> list[tuple[int, str, str, int | None]]:
    occurrences: list[tuple[int, str, str, int | None]] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in flags and idx + 1 < len(argv):
            occurrences.append((idx, token, argv[idx + 1], idx + 1))
            idx += 2
            continue
        matched = False
        for flag in flags:
            prefix = f"{flag}="
            if token.startswith(prefix):
                occurrences.append((idx, flag, token[len(prefix) :], None))
                matched = True
                break
        idx += 1 if not matched else 1
    return occurrences


def _replace_flag_value(
    argv: list[str],
    *,
    token_index: int,
    flag: str,
    value_index: int | None,
    new_value: str,
) -> None:
    if value_index is None:
        argv[token_index] = f"{flag}={new_value}"
    else:
        argv[value_index] = new_value


def _needs_gpu_passthrough(argv: list[str]) -> bool:
    requested = _requested_device(argv)
    if requested not in {"auto", "cuda"}:
        return False
    return _host_nvidia_visible()


def _validate_requested_device(argv: list[str]) -> None:
    requested = _requested_device(argv)
    if requested != "cuda":
        return
    if _host_nvidia_visible():
        return
    raise RuntimeError(
        "Requested --device cuda, but no NVIDIA runtime is visible on the host. "
        "Install NVIDIA container support so `nvidia-smi` or `/dev/nvidiactl` is "
        "available, or use --device auto/cpu."
    )


def normalize_delegated_argv(argv: list[str], *, cwd: _Path) -> ContainerLaunchPlan:
    rewritten = list(argv)
    mounts: set[_Path] = set()
    needs_cwd_host_mirror = False

    for token_index, flag, value, value_index in _iter_flag_occurrences(
        rewritten, flags=_CONFIG_PATH_ARG_FLAGS
    ):
        scan_dependencies = flag in _CONFIG_SCAN_ARG_FLAGS
        new_value, extra_mounts, needs_mirror = _normalize_config_path_for_container(
            value,
            cwd=cwd,
            scan_dependencies=scan_dependencies,
        )
        mounts.update(extra_mounts)
        needs_cwd_host_mirror = needs_cwd_host_mirror or needs_mirror
        _replace_flag_value(
            rewritten,
            token_index=token_index,
            flag=flag,
            value_index=value_index,
            new_value=new_value,
        )

    for token_index, flag, value, value_index in _iter_flag_occurrences(
        rewritten, flags=_OUTPUT_PATH_ARG_FLAGS
    ):
        new_value, extra_mounts = _normalize_output_path_for_container(
            value,
            cwd=cwd,
        )
        mounts.update(extra_mounts)
        _replace_flag_value(
            rewritten,
            token_index=token_index,
            flag=flag,
            value_index=value_index,
            new_value=new_value,
        )

    for token_index, flag, value, value_index in _iter_flag_occurrences(
        rewritten, flags=_LOCAL_MODEL_ARG_FLAGS
    ):
        new_value, extra_mounts, treated_as_path = (
            _normalize_local_model_path_for_container(
                value,
                cwd=cwd,
            )
        )
        if not treated_as_path:
            continue
        mounts.update(extra_mounts)
        _replace_flag_value(
            rewritten,
            token_index=token_index,
            flag=flag,
            value_index=value_index,
            new_value=new_value,
        )

    _validate_requested_device(rewritten)

    return ContainerLaunchPlan(
        argv=tuple(rewritten),
        argv_mounts=tuple(_minimize_mounts(mounts)),
        needs_cwd_host_mirror=needs_cwd_host_mirror,
        gpu_passthrough=_needs_gpu_passthrough(rewritten),
        workspace_read_only=(
            _os.environ.get(SOURCE_BUNDLE_READ_ONLY_ENV, "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    )


def build_current_process_container_launch_plan(
    argv: list[str] | None = None,
) -> ContainerLaunchPlan:
    delegated_argv = list(sys.argv[1:] if argv is None else argv)
    return normalize_delegated_argv(delegated_argv, cwd=_Path.cwd().resolve())


def build_request_container_launch_plan(
    command_name: str | Iterable[str],
    request: SupportsContainerLaunchRequest,
) -> ContainerLaunchPlan:
    return normalize_delegated_argv(
        request.to_internal_argv(command_name),
        cwd=_Path.cwd().resolve(),
    )


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
    "RUNTIME_IMAGE_DEFAULT",
    "RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT",
    "RUNTIME_IMAGE_ENV",
    "RUNTIME_IMAGE_DIGEST_ENV",
    "RUNTIME_IMAGE_LOCAL_DEFAULT",
    "RUNTIME_MANIFEST_FILENAME",
    "RUNTIME_MANIFEST_VERSION",
    "RUNTIME_VERIFIER_CONTRACT_VERSION",
    "SOURCE_BUNDLE_DIGEST_ENV",
    "SOURCE_BUNDLE_READ_ONLY_ENV",
    "apply_runtime_allowances",
    "build_container_command",
    "build_container_python_module_command",
    "build_container_python_command",
    "build_runtime_security_policy",
    "container_image_available_locally",
    "current_execution_mode",
    "current_runtime_security_policy",
    "delegate_container_command",
    "delegate_python_module_to_container",
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
