"""Shared compact checkpoint profile for runnable integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.runtime_providers.hf_transformers import (
    hf_tokenizer_contract_sha256,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_TYPE = "qwen3_5"
MODEL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
PEFT_TARGET_MODULES = ("q_proj", "v_proj")


def _validate_config(config: Any) -> None:
    """Reject a changed repository architecture before loading model weights."""

    architectures = getattr(config, "architectures", None)
    if (
        getattr(config, "model_type", None) != MODEL_TYPE
        or not isinstance(architectures, list)
        or MODEL_ARCHITECTURE not in architectures
    ):
        raise RuntimeError(
            "the pinned compact checkpoint is not the expected Qwen3.5 text architecture"
        )


def load_model_and_tokenizer(*, torch: Any, transformers: Any) -> tuple[Any, Any]:
    """Load the immutable official checkpoint without executing remote code."""

    del torch
    config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    _validate_config(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("the pinned compact tokenizer has no padding token")
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype="auto",
        use_safetensors=True,
        trust_remote_code=False,
    )
    if getattr(model.config, "model_type", None) != "qwen3_5_text":
        raise RuntimeError(
            "the pinned compact checkpoint did not resolve to its causal text model"
        )
    model.config.use_cache = False
    return model, tokenizer


def save_checkpoint(model: Any, tokenizer: Any, checkpoint: Path) -> str:
    """Save a local worker-readable snapshot and return its tokenizer identity."""

    checkpoint.mkdir(parents=True)
    model.eval()
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    checkpoint.chmod(0o755)
    for path in checkpoint.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return hf_tokenizer_contract_sha256(tokenizer)


def provenance(*, checkpoint_tree_sha256: str) -> dict[str, str]:
    """Return the source fields authenticated by each transformation summary."""

    return {
        "source_model_id": MODEL_ID,
        "source_model_revision": MODEL_REVISION,
        "source_checkpoint_tree_sha256": checkpoint_tree_sha256,
    }
