"""Immutable host-launch metadata for maintained optional runtime providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type AutomaticEntrypoint = Literal["python", "nvidia"]
type ScratchProfile = Literal["default", "tensorrt_engine"]


@dataclass(frozen=True, slots=True)
class OptionalRuntimeProviderProfile:
    """Caller-owned resources and host launch behavior for one optional provider."""

    provider_name: str
    resource_root_environment: str
    support_resource_environment: tuple[tuple[str, str], ...]
    automatic_entrypoint: AutomaticEntrypoint = "python"
    scratch_profile: ScratchProfile = "default"


_PROFILES = (
    OptionalRuntimeProviderProfile(
        provider_name="hf_vision_text",
        resource_root_environment="INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT",
        support_resource_environment=(
            ("content_store", "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE"),
        ),
    ),
    OptionalRuntimeProviderProfile(
        provider_name="llama_cpp",
        resource_root_environment="INVARLOCK_GGUF_RESOURCE_ROOT",
        support_resource_environment=(
            ("backend_executable", "INVARLOCK_GGUF_BACKEND_EXECUTABLE"),
            ("backend_source", "INVARLOCK_GGUF_BACKEND_SOURCE"),
        ),
    ),
    OptionalRuntimeProviderProfile(
        provider_name="tensorrt_llm",
        resource_root_environment="INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT",
        support_resource_environment=(
            ("tokenizer_contract", "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT"),
        ),
        automatic_entrypoint="nvidia",
        scratch_profile="tensorrt_engine",
    ),
)

OPTIONAL_RUNTIME_PROVIDER_PROFILES: Mapping[str, OptionalRuntimeProviderProfile] = (
    MappingProxyType({profile.provider_name: profile for profile in _PROFILES})
)


__all__ = [
    "OPTIONAL_RUNTIME_PROVIDER_PROFILES",
    "OptionalRuntimeProviderProfile",
]
