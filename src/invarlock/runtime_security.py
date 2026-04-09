"""Public runtime-security facade.

Implementation helpers live in `invarlock.runtime_security_helpers`. The facade
keeps the supported public API stable without mirroring helper globals back into
the implementation module at call time.

Typed request-scoped policy surface retained at the owner boundary:
- class RuntimeSecurityPolicy
- def build_runtime_security_policy(
- policy: RuntimeSecurityPolicy | None = None
- ContextVar(
- def reset_runtime_allowances(
"""

from __future__ import annotations

from invarlock import runtime_security_helpers as _helpers
from invarlock.runtime_security_helpers import (
    ALLOW_HOST_EXECUTION_ENV,
    ALLOW_NETWORK_ENV,
    ALLOW_REMOTE_CODE_ENV,
    ALLOW_THIRD_PARTY_PLUGINS_ENV,
    ALLOW_UNATTESTED_ARTIFACTS_ENV,
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
    ContainerLaunchPlan,
    RuntimeManifestExecution,
    RuntimeManifestLoadIssueCode,
    RuntimeManifestLoadResult,
    RuntimeSecurityPolicy,
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
    serialize_canonical_json,
    third_party_plugins_allowed,
    unattested_artifacts_allowed,
    write_runtime_manifest,
)

_CONTAINER_EXECUTION_TIMEOUT_SECONDS = _helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS
_CONTAINER_INSPECT_TIMEOUT_SECONDS = _helpers._CONTAINER_INSPECT_TIMEOUT_SECONDS
_PATH_ENV_VARS = _helpers._PATH_ENV_VARS
_absolute_host_path = _helpers._absolute_host_path
_attested_runtime_image_ref = _helpers._attested_runtime_image_ref
_coerce_bool = _helpers._coerce_bool
_config_digest = _helpers._config_digest
_container_pythonpath_entries = _helpers._container_pythonpath_entries
_delegated_env_pairs = _helpers._delegated_env_pairs
_host_nvidia_visible = _helpers._host_nvidia_visible
_inspect_container_image = _helpers._inspect_container_image
_iter_absolute_pythonpath_entries = _helpers._iter_absolute_pythonpath_entries
_iter_external_symlink_target_mounts = _helpers._iter_external_symlink_target_mounts
_minimize_mounts = _helpers._minimize_mounts
_mount_root_for_path = _helpers._mount_root_for_path
_mount_root_for_resolved_path = _helpers._mount_root_for_resolved_path
_normalize_config_path_for_container = _helpers._normalize_config_path_for_container
_normalize_local_model_path_for_container = (
    _helpers._normalize_local_model_path_for_container
)
_normalize_output_path_for_container = _helpers._normalize_output_path_for_container
_path_env_value_for_container = _helpers._path_env_value_for_container
_path_is_within = _helpers._path_is_within
_runtime_flag_value = _helpers._runtime_flag_value
_workspace_path = _helpers._workspace_path
inspect_config_dependencies = _helpers.inspect_config_dependencies
os = _helpers.os
Path = _helpers.Path
shutil = _helpers.shutil
subprocess = _helpers.subprocess

__all__ = [
    "ALLOW_HOST_EXECUTION_ENV",
    "ALLOW_NETWORK_ENV",
    "ALLOW_REMOTE_CODE_ENV",
    "ALLOW_THIRD_PARTY_PLUGINS_ENV",
    "ALLOW_UNATTESTED_ARTIFACTS_ENV",
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
    "serialize_canonical_json",
    "third_party_plugins_allowed",
    "unattested_artifacts_allowed",
    "write_runtime_manifest",
]
