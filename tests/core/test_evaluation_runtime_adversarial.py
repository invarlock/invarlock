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

_DIGEST = "sha256:" + "e" * 64


def _side(
    path: Path | None, provider: str = "hf_transformers"
) -> ComparisonSideRequest:
    return ComparisonSideRequest(
        artifact=ArtifactRequest(path=path, model_id="model", locator="artifact:model"),
        runtime=RuntimeRequest(provider=provider, settings={}),
    )


def _provider(name: str) -> RuntimeProvider:
    return cast(RuntimeProvider, SimpleNamespace(name=name))


def test_runtime_resources_reject_relative_roots_and_unknown_side_keys() -> None:
    with pytest.raises(RuntimeResourceResolutionError, match="root must be absolute"):
        ProviderResourceBinding(root=Path("relative"))
    with pytest.raises(RuntimeResourceResolutionError, match="side device key"):
        CallerRuntimeResources(
            container_image_digest=_DIGEST,
            side_devices={"other": "cpu"},  # type: ignore[dict-item]
        )


def test_runtime_resolution_rejects_provider_substitution_and_missing_artifact(
    tmp_path: Path,
) -> None:
    resolver = CallerRuntimeResources(container_image_digest=_DIGEST)
    with pytest.raises(RuntimeResourceResolutionError, match="identity does not match"):
        resolver.resolve(
            request_root=tmp_path,
            role="baseline",
            side=_side(tmp_path / "model"),
            provider=_provider("llama_cpp"),
        )
    with pytest.raises(
        RuntimeResourceResolutionError, match="requires a local artifact"
    ):
        resolver.resolve(
            request_root=tmp_path,
            role="subject",
            side=_side(None),
            provider=_provider("hf_transformers"),
        )


def test_runtime_resolution_rejects_root_itself_and_invalid_resource_contract(
    tmp_path: Path,
) -> None:
    resolver = CallerRuntimeResources(container_image_digest=_DIGEST)
    with pytest.raises(RuntimeResourceResolutionError, match="beneath its root"):
        resolver.resolve(
            request_root=tmp_path,
            role="baseline",
            side=_side(tmp_path),
            provider=_provider("hf_transformers"),
        )

    artifact = tmp_path / "model"
    artifact.write_bytes(b"model")
    malformed = CallerRuntimeResources(container_image_digest="not-a-digest")
    with pytest.raises(RuntimeResourceResolutionError, match="resources are invalid"):
        malformed.resolve(
            request_root=tmp_path,
            role="subject",
            side=_side(artifact),
            provider=_provider("hf_transformers"),
        )


def test_environment_requires_image_digest_and_loads_complete_optional_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    with pytest.raises(RuntimeResourceResolutionError, match="must bind"):
        caller_runtime_resources_from_environment()

    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_DEVICE", "cuda")
    monkeypatch.setenv("INVARLOCK_BASELINE_RUNTIME_DEVICE", "cpu")
    monkeypatch.setenv("INVARLOCK_SUBJECT_RUNTIME_DEVICE", "cuda:1")
    monkeypatch.setenv(
        "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT", str(tmp_path / "vision")
    )
    monkeypatch.setenv("INVARLOCK_HF_VISION_TEXT_CONTENT_STORE", "images")
    monkeypatch.setenv("INVARLOCK_GGUF_RESOURCE_ROOT", str(tmp_path / "gguf"))
    monkeypatch.setenv("INVARLOCK_GGUF_BACKEND_EXECUTABLE", "bin/llama-cli")
    monkeypatch.setenv("INVARLOCK_GGUF_BACKEND_SOURCE", "src/llama.cpp.tar")
    monkeypatch.setenv("INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT", str(tmp_path / "trt"))
    monkeypatch.setenv("INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT", "tokenizer.json")
    monkeypatch.setenv("INVARLOCK_TENSORRT_LLM_RUNNER_EXECUTABLE", "runner")

    resources = caller_runtime_resources_from_environment()

    assert resources.default_device == "cuda"
    assert dict(resources.side_devices) == {"baseline": "cpu", "subject": "cuda:1"}
    assert set(resources.provider_bindings) == {
        "hf_vision_text",
        "llama_cpp",
        "tensorrt_llm",
    }
    assert dict(resources.provider_bindings["hf_vision_text"].support_resources) == {
        "content_store": "images"
    }
    assert dict(resources.provider_bindings["llama_cpp"].support_resources) == {
        "backend_executable": "bin/llama-cli",
        "backend_source": "src/llama.cpp.tar",
    }


def test_environment_fails_closed_when_tensorrt_support_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _DIGEST)
    monkeypatch.setenv("INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT", "tokenizer.json")
    monkeypatch.delenv("INVARLOCK_TENSORRT_LLM_RUNNER_EXECUTABLE", raising=False)

    with pytest.raises(RuntimeResourceResolutionError, match="RUNNER_EXECUTABLE"):
        caller_runtime_resources_from_environment()
