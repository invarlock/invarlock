"""Closed normalized-request field contracts for first-party runtimes.

These constants are intentionally torch-free and add-in-free. Runtime providers
use them to validate authored requests, while the independent verifier uses the
same contracts to reconcile those requests with authenticated receipts.
"""

from __future__ import annotations

from collections.abc import Mapping

LLAMA_CPP_MAX_CPU_THREADS = 256
LLAMA_CPP_MAX_PROMPT_BATCH_SIZE = 4096

RUNTIME_EXECUTION_REQUEST_SETTINGS = frozenset(
    {
        "batch_size",
        "context_length",
        "max_output_tokens",
        "seed",
        "timeout_seconds",
    }
)

HF_TRANSFORMERS_ARTIFACT_REQUEST_BINDINGS = (
    ("checkpoint_tree_sha256", "checkpoint_tree_sha256"),
    ("immutable_revision", "immutable_revision"),
    ("tokenizer_metadata_sha256", "tokenizer_metadata_sha256"),
)
HF_TRANSFORMERS_REQUEST_SETTINGS = frozenset(
    {
        *RUNTIME_EXECUTION_REQUEST_SETTINGS,
        *(
            request_field
            for request_field, _ in HF_TRANSFORMERS_ARTIFACT_REQUEST_BINDINGS
        ),
        "offline",
    }
)
HF_TRANSFORMERS_REQUIRED_REQUEST_SETTINGS = frozenset(
    {
        *RUNTIME_EXECUTION_REQUEST_SETTINGS,
        "offline",
        "tokenizer_metadata_sha256",
    }
)

HF_VISION_TEXT_REQUEST_SETTINGS = frozenset(
    {*HF_TRANSFORMERS_REQUEST_SETTINGS, "processor_metadata_sha256"}
)
HF_VISION_TEXT_REQUIRED_REQUEST_SETTINGS = frozenset(
    {
        *HF_TRANSFORMERS_REQUIRED_REQUEST_SETTINGS,
        "checkpoint_tree_sha256",
        "processor_metadata_sha256",
    }
)

LLAMA_CPP_REQUEST_SETTINGS_V1 = frozenset(
    {
        *RUNTIME_EXECUTION_REQUEST_SETTINGS,
        "artifact_byte_length",
        "artifact_sha256",
        "backend_binary_sha256",
        "backend_source_sha256",
        "backend_version",
        "gguf_metadata_sha256",
        "tensor_inventory_sha256",
        "tokenizer_metadata_sha256",
    }
)
LLAMA_CPP_REQUEST_SETTINGS = frozenset(
    {
        *LLAMA_CPP_REQUEST_SETTINGS_V1,
        "cpu_threads",
        "prompt_batch_size",
        "prompt_microbatch_size",
    }
)
LLAMA_CPP_VERIFIABLE_REQUEST_SETTING_SETS = frozenset(
    {LLAMA_CPP_REQUEST_SETTINGS_V1, LLAMA_CPP_REQUEST_SETTINGS}
)


def llama_cpp_execution_profile_errors(
    settings: Mapping[str, object],
) -> tuple[str, ...]:
    """Validate llama.cpp controls whose semantics differ from record batching."""

    errors: list[str] = []

    def positive_integer(name: str) -> int | None:
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(
                f"llama_cpp request setting {name!r} must be a positive integer"
            )
            return None
        return value

    context_length = positive_integer("context_length")
    cpu_threads = positive_integer("cpu_threads")
    prompt_batch_size = positive_integer("prompt_batch_size")
    prompt_microbatch_size = positive_integer("prompt_microbatch_size")
    if cpu_threads is not None and cpu_threads > LLAMA_CPP_MAX_CPU_THREADS:
        errors.append("llama_cpp request cpu_threads exceeds the supported limit")
    if (
        context_length is not None
        and prompt_batch_size is not None
        and prompt_batch_size > min(context_length, LLAMA_CPP_MAX_PROMPT_BATCH_SIZE)
    ):
        errors.append(
            "llama_cpp request prompt_batch_size must not exceed "
            "context_length or the supported limit"
        )
    if (
        prompt_batch_size is not None
        and prompt_microbatch_size is not None
        and prompt_microbatch_size > prompt_batch_size
    ):
        errors.append(
            "llama_cpp request prompt_microbatch_size must not exceed prompt_batch_size"
        )
    return tuple(errors)


TENSORRT_LLM_ARTIFACT_REQUEST_BINDINGS = (
    ("builder_config_sha256", "builder_config_sha256"),
    ("engine_bundle_tree_sha256", "engine_bundle_tree_sha256"),
    ("engine_metadata_sha256", "engine_metadata_sha256"),
    ("file_inventory_sha256", "file_inventory_sha256"),
    ("target_compute_capability", "target_compute_capability"),
    ("tokenizer_metadata_sha256", "tokenizer_metadata_sha256"),
)
TENSORRT_LLM_BACKEND_REQUEST_BINDINGS = (
    ("backend_build_sha256", "build_sha256"),
    ("backend_version", "version"),
    ("runner_binary_sha256", "binary_sha256"),
)
TENSORRT_LLM_REQUEST_SETTINGS = frozenset(
    {
        *RUNTIME_EXECUTION_REQUEST_SETTINGS,
        *(request_field for request_field, _ in TENSORRT_LLM_ARTIFACT_REQUEST_BINDINGS),
        *(request_field for request_field, _ in TENSORRT_LLM_BACKEND_REQUEST_BINDINGS),
    }
)

__all__ = [
    "HF_TRANSFORMERS_ARTIFACT_REQUEST_BINDINGS",
    "HF_TRANSFORMERS_REQUEST_SETTINGS",
    "HF_TRANSFORMERS_REQUIRED_REQUEST_SETTINGS",
    "HF_VISION_TEXT_REQUEST_SETTINGS",
    "HF_VISION_TEXT_REQUIRED_REQUEST_SETTINGS",
    "LLAMA_CPP_MAX_CPU_THREADS",
    "LLAMA_CPP_MAX_PROMPT_BATCH_SIZE",
    "LLAMA_CPP_REQUEST_SETTINGS",
    "LLAMA_CPP_REQUEST_SETTINGS_V1",
    "LLAMA_CPP_VERIFIABLE_REQUEST_SETTING_SETS",
    "llama_cpp_execution_profile_errors",
    "RUNTIME_EXECUTION_REQUEST_SETTINGS",
    "TENSORRT_LLM_ARTIFACT_REQUEST_BINDINGS",
    "TENSORRT_LLM_BACKEND_REQUEST_BINDINGS",
    "TENSORRT_LLM_REQUEST_SETTINGS",
]
