from __future__ import annotations

import invarlock.runtime_security as runtime_security


def test_runtime_security_facade_exposes_only_public_surface() -> None:
    banned_names = {
        "_coerce_bool",
        "_config_digest",
        "_inspect_container_image",
        "_runtime_flag_value",
        "os",
        "Path",
        "shutil",
        "subprocess",
    }

    for name in banned_names:
        assert not hasattr(runtime_security, name), name

    expected_public = {
        "ALLOW_HOST_EXECUTION_ENV",
        "ALLOW_NETWORK_ENV",
        "ALLOW_REMOTE_CODE_ENV",
        "ALLOW_THIRD_PARTY_PLUGINS_ENV",
        "ALLOW_UNVERIFIED_PROVENANCE_ENV",
        "CONTAINER_ENGINE_ENV",
        "CONTAINER_EXECUTION_ENV",
        "ContainerLaunchPlan",
        "RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT",
        "RUNTIME_IMAGE_DEFAULT",
        "RUNTIME_IMAGE_DIGEST_ENV",
        "RUNTIME_IMAGE_ENV",
        "RUNTIME_IMAGE_LOCAL_DEFAULT",
        "RUNTIME_MANIFEST_FILENAME",
        "RUNTIME_MANIFEST_VERSION",
        "RUNTIME_VERIFIER_CONTRACT_VERSION",
        "SOURCE_BUNDLE_DIGEST_ENV",
        "SOURCE_BUNDLE_READ_ONLY_ENV",
        "RuntimeManifestExecution",
        "RuntimeManifestLoadIssueCode",
        "RuntimeManifestLoadResult",
        "RuntimeSecurityPolicy",
        "apply_runtime_allowances",
        "build_container_command",
        "build_container_python_command",
        "build_container_python_module_command",
        "build_runtime_security_policy",
        "container_image_available_locally",
        "current_execution_mode",
        "current_runtime_security_policy",
        "delegate_container_command",
        "delegate_python_module_to_container",
        "delegate_python_script_to_container",
        "load_runtime_manifest",
        "network_allowed",
        "host_execution_allowed",
        "remote_code_allowed",
        "reset_runtime_allowances",
        "resolve_container_engine",
        "resolve_runtime_image",
        "resolve_runtime_image_digest",
        "running_inside_container",
        "runtime_allowances_scope",
        "third_party_plugins_allowed",
        "unverified_provenance_allowed",
        "write_runtime_manifest",
    }

    assert set(runtime_security.__all__) == expected_public
