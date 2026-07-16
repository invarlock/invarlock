from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from invarlock.core.evaluation_request import (
    ArtifactRequest,
    ComparisonSideRequest,
    RuntimeRequest,
)
from invarlock.core.runtime_provider import RuntimeProvider
from invarlock.evaluation_runtime import (
    CallerRuntimeResources,
    ProviderResourceBinding,
    RuntimeResourceResolutionError,
    caller_runtime_resources_from_environment,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64


def _side(artifact: Path, *, provider: str) -> ComparisonSideRequest:
    return ComparisonSideRequest(
        artifact=ArtifactRequest(
            path=artifact,
            model_id="portable-model",
            locator="artifact:portable-model",
        ),
        runtime=RuntimeRequest(provider=provider, settings={}),
    )


def _provider(name: str) -> RuntimeProvider:
    return cast(RuntimeProvider, SimpleNamespace(name=name))


def test_caller_runtime_resources_bind_hf_artifact_without_request_support_fields(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "models" / "checkpoint"
    checkpoint.mkdir(parents=True)
    resolver = CallerRuntimeResources(
        container_image_digest=_IMAGE_DIGEST,
        default_device="cpu",
        side_devices={"subject": "cuda"},
    )

    baseline = resolver.resolve(
        request_root=tmp_path,
        role="baseline",
        side=_side(checkpoint, provider="hf_transformers"),
        provider=_provider("hf_transformers"),
    )
    subject = resolver.resolve(
        request_root=tmp_path,
        role="subject",
        side=_side(checkpoint, provider="hf_transformers"),
        provider=_provider("hf_transformers"),
    )

    assert baseline.primary_artifact == "models/checkpoint"
    assert dict(baseline.support_resources) == {}
    assert baseline.device_kind == "cpu"
    assert subject.device_kind == "cuda"


def test_optional_provider_requires_explicit_caller_owned_resources(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    resolver = CallerRuntimeResources(container_image_digest=_IMAGE_DIGEST)

    with pytest.raises(
        RuntimeResourceResolutionError, match="caller-owned resource configuration"
    ):
        resolver.resolve(
            request_root=tmp_path,
            role="baseline",
            side=_side(artifact, provider="llama_cpp"),
            provider=_provider("llama_cpp"),
        )


def test_optional_provider_rejects_artifact_outside_trusted_resource_root(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    trusted.joinpath("backend").write_bytes(b"backend")
    trusted.joinpath("source.tar").write_bytes(b"source")
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"gguf")
    resolver = CallerRuntimeResources(
        container_image_digest=_IMAGE_DIGEST,
        provider_bindings={
            "llama_cpp": ProviderResourceBinding(
                root=trusted,
                support_resources={
                    "backend_executable": "backend",
                    "backend_source": "source.tar",
                },
            )
        },
    )

    with pytest.raises(RuntimeResourceResolutionError, match="outside"):
        resolver.resolve(
            request_root=tmp_path,
            role="subject",
            side=_side(outside, provider="llama_cpp"),
            provider=_provider("llama_cpp"),
        )


def test_environment_resolver_fails_closed_on_partial_addin_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_GGUF_RESOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("INVARLOCK_GGUF_BACKEND_EXECUTABLE", "bin/backend")
    monkeypatch.delenv("INVARLOCK_GGUF_BACKEND_SOURCE", raising=False)

    with pytest.raises(RuntimeResourceResolutionError, match="BACKEND_SOURCE"):
        caller_runtime_resources_from_environment()


def test_environment_resolver_does_not_accept_support_without_provider_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.delenv("INVARLOCK_GGUF_RESOURCE_ROOT", raising=False)
    monkeypatch.setenv("INVARLOCK_GGUF_BACKEND_EXECUTABLE", "bin/backend")

    with pytest.raises(RuntimeResourceResolutionError, match="RESOURCE_ROOT"):
        caller_runtime_resources_from_environment()
