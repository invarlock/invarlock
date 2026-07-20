"""Closed normalized-request field contracts for first-party runtimes.

These constants are intentionally torch-free and add-in-free. Runtime providers
use them to validate authored requests, while the independent verifier uses the
same contracts to reconcile those requests with authenticated receipts.
"""

from __future__ import annotations

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

LLAMA_CPP_REQUEST_SETTINGS = frozenset(
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
    "LLAMA_CPP_REQUEST_SETTINGS",
    "RUNTIME_EXECUTION_REQUEST_SETTINGS",
    "TENSORRT_LLM_ARTIFACT_REQUEST_BINDINGS",
    "TENSORRT_LLM_BACKEND_REQUEST_BINDINGS",
    "TENSORRT_LLM_REQUEST_SETTINGS",
]
