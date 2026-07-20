"""Shared official Qwen3 checkpoint profile for runnable integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.runtime_providers.hf_transformers import (
    hf_tokenizer_contract_sha256,
)

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PEFT_TARGET_MODULES = ("q_proj", "v_proj")


def load_model_and_tokenizer(*, torch: Any, transformers: Any) -> tuple[Any, Any]:
    """Load the immutable official checkpoint without executing remote code."""

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("the pinned Qwen3 tokenizer has no padding token")
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype="auto",
        use_safetensors=True,
        trust_remote_code=False,
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
