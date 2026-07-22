from __future__ import annotations

import pytest

import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as helpers

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64


def test_runtime_allowances_fail_closed_and_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "invalid")
    monkeypatch.setenv(runtime_security.ALLOW_REMOTE_CODE_ENV, "0")
    monkeypatch.setenv(runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV, "1")

    assert runtime_security.network_allowed() is False
    assert runtime_security.remote_code_allowed() is False
    assert runtime_security.third_party_plugins_allowed() is True


def test_strict_container_boundary_requires_intent_and_kernel_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helpers, "_regular_file_marker_present", lambda _path: True)
    monkeypatch.delenv(runtime_security.CONTAINER_EXECUTION_ENV, raising=False)
    assert runtime_security.strict_container_boundary_present() is False

    monkeypatch.setenv(runtime_security.CONTAINER_EXECUTION_ENV, "1")
    assert runtime_security.strict_container_boundary_present() is True


def test_runtime_image_resolution_composes_exact_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_ENV,
        "registry.example/invarlock/runtime:release",
    )
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, _DIGEST_A)

    assert runtime_security.resolve_runtime_image_digest() == _DIGEST_A
    assert runtime_security.resolve_runtime_image() == (
        f"registry.example/invarlock/runtime:release@{_DIGEST_A}"
    )


def test_runtime_image_resolution_rejects_malformed_or_conflicting_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, "sha256:BAD")
    with pytest.raises(RuntimeError, match="lowercase sha256"):
        runtime_security.resolve_runtime_image_digest()

    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, _DIGEST_A)
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_ENV,
        f"registry.example/invarlock/runtime@{_DIGEST_B}",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        runtime_security.resolve_runtime_image()


def test_runtime_security_facade_is_narrow() -> None:
    assert set(runtime_security.__all__) == {
        "ALLOW_NETWORK_ENV",
        "ALLOW_REMOTE_CODE_ENV",
        "ALLOW_THIRD_PARTY_PLUGINS_ENV",
        "CONTAINER_EXECUTION_ENV",
        "RUNTIME_IMAGE_DIGEST_ENV",
        "RUNTIME_IMAGE_ENV",
        "RUNTIME_MANIFEST_FILENAME",
        "RUNTIME_MANIFEST_VERSION",
        "RuntimeManifestExecution",
        "RuntimeProviderManifestFiles",
        "current_execution_mode",
        "network_allowed",
        "remote_code_allowed",
        "resolve_runtime_image",
        "resolve_runtime_image_digest",
        "running_inside_container",
        "strict_container_boundary_present",
        "third_party_plugins_allowed",
    }
