from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    from transformers import AutoTokenizer, GPT2Tokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
except Exception:  # pragma: no cover - exercised only when transformers is absent
    AutoTokenizer = None  # type: ignore[assignment]
    GPT2Tokenizer = None  # type: ignore[assignment]

    class PreTrainedTokenizerBase:  # type: ignore[no-redef]
        """Lightweight stub used when transformers is not installed."""

        def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(
                "Tokenization requires the 'transformers' extra. "
                "Install it with: pip install 'invarlock[adapters]'."
            )


TokenizerFactory = Callable[[], tuple[PreTrainedTokenizerBase, str]]


def _hash_tokenizer(tokenizer: PreTrainedTokenizerBase) -> str:
    try:
        if hasattr(tokenizer, "get_vocab"):
            vocab_mapping = tokenizer.get_vocab()
        else:
            vocab_mapping = getattr(tokenizer, "vocab", {})
        if hasattr(vocab_mapping, "items"):
            vocab_items = list(vocab_mapping.items())
        else:
            vocab_items = []
    except Exception:
        vocab_items = []

    hasher = hashlib.blake2s(digest_size=16)
    try:
        for token, idx in sorted(vocab_items, key=lambda x: x[0]):
            token_str = token if isinstance(token, str) else str(token)
            hasher.update(token_str.encode("utf-8", "ignore"))
            try:
                hasher.update(int(idx).to_bytes(4, "little", signed=False))
            except Exception:
                hasher.update(str(idx).encode("utf-8", "ignore"))
    except Exception:
        return "unknown"

    hasher.update(tokenizer.__class__.__name__.encode("utf-8", "ignore"))
    name_path = getattr(tokenizer, "name_or_path", "")
    hasher.update(str(name_path).encode("utf-8", "ignore"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class ModelProfile:
    """Captured capabilities for a recognised model family."""

    family: str
    default_loss: str
    make_tokenizer: TokenizerFactory
    default_metric: str = "ppl_causal"
    # Must correspond to a registered provider in invarlock.eval.data.get_provider
    default_provider: str = "wikitext2"
    module_selectors: dict[str, list[str]] = field(default_factory=dict)
    invariants: tuple[str, ...] = ()
    cert_lints: tuple[dict[str, str], ...] = ()


def _bert_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ],
        "ffn": [
            "intermediate.dense",
            "output.dense",
        ],
    }


def _gpt2_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "attn.c_attn",
            "attn.c_proj",
        ],
        "ffn": [
            "mlp.c_fc",
            "mlp.c_proj",
        ],
    }


def _llama_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
        "ffn": [
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate_proj",
        ],
    }


def _unknown_selectors() -> dict[str, list[str]]:
    return {
        "attention": ["attention"],
        "ffn": [],
    }


def _make_bert_tokenizer(model_id: str):
    def factory() -> tuple[PreTrainedTokenizerBase, str]:
        if AutoTokenizer is None:
            raise RuntimeError(
                "BERT tokenizers require the 'transformers' extra. "
                "Install it with: pip install 'invarlock[adapters]'."
            )
        # Prefer offline/local cache first to respect network guard
        tokenizer: PreTrainedTokenizerBase | None = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception:
            # Try a common local BERT if specific model is not cached
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-uncased", local_files_only=True
                )
            except Exception:
                # If network is permitted, attempt remote fetch; otherwise propagate
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                except Exception:
                    tokenizer = None
        if tokenizer is None:
            raise RuntimeError(
                "Unable to load a BERT tokenizer locally. Set INVARLOCK_ALLOW_NETWORK=1 "
                "to allow fetching from the Hugging Face Hub, or pre-cache a BERT tokenizer."
            )
        if getattr(tokenizer, "mask_token", None) is None:
            raise ValueError(
                f"Tokenizer for '{model_id}' does not expose [MASK]; cannot run MLM evaluation."
            )
        if getattr(tokenizer, "pad_token", None) is None:
            for candidate in (
                getattr(tokenizer, "sep_token", None),
                getattr(tokenizer, "cls_token", None),
            ):
                if candidate is not None:
                    tokenizer.pad_token = candidate
                    break
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_gpt2_tokenizer(model_id: str):
    def factory() -> tuple[PreTrainedTokenizerBase, str]:
        if GPT2Tokenizer is None:
            raise RuntimeError(
                "GPT-2 tokenizers require the 'transformers' extra. "
                "Install it with: pip install 'invarlock[adapters]'."
            )
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_llama_tokenizer(model_id: str):
    def factory() -> tuple[PreTrainedTokenizerBase, str]:
        if AutoTokenizer is None and GPT2Tokenizer is None:
            raise RuntimeError(
                "LLaMA-style tokenizers require the 'transformers' extra. "
                "Install it with: pip install 'invarlock[adapters]'."
            )
        # Try offline-first to respect InvarLock network guard; fall back to a
        # local GPT-2 tokenizer if the model assets are not cached or network
        # access is denied.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception:
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Ensure padding/bos tokens are configured so downstream encoding
        # yields stable non-zero ids and a valid attention mask regardless of
        # environment defaults or tokenizer variants.
        # Prefer EOS as pad token when no explicit pad token is defined.
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        # Some LLaMA tokenizers default to not adding a BOS token on encode;
        # enable it to guarantee at least one non-pad, non-zero token id.
        if hasattr(tokenizer, "add_bos_token"):
            try:
                tokenizer.add_bos_token = True
            except Exception:
                pass
        if getattr(tokenizer, "pad_token", None) is None:
            raise ValueError(
                f"Tokenizer for '{model_id}' does not define a pad token and no EOS fallback is available."
            )
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_unknown_tokenizer(model_id: str):
    def factory() -> tuple[PreTrainedTokenizerBase, str]:
        if AutoTokenizer is None and GPT2Tokenizer is None:
            raise RuntimeError(
                "Text tokenization requires the 'transformers' extra. "
                "Install it with: pip install 'invarlock[adapters]'."
            )
        # Unknown families: try local-only first, then remote, then degrade to GPT-2
        try:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                model_id, local_files_only=True
            )
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception:
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def detect_model_profile(model_id: str, adapter: str | None = None) -> ModelProfile:
    """
    Infer the model family and provide profile metadata used for evaluation.
    """

    adapter_lower = (adapter or "").lower()
    model_lower = (model_id or "").lower()

    if any(
        keyword in adapter_lower for keyword in ("bert", "roberta", "deberta")
    ) or any(keyword in model_lower for keyword in ("bert", "roberta", "deberta")):
        return ModelProfile(
            family="bert",
            default_loss="mlm",
            make_tokenizer=_make_bert_tokenizer(model_id),
            default_metric="ppl_mlm",
            default_provider="hf_text",
            module_selectors=_bert_selectors(),
            invariants=("mlm_mask_alignment",),
            cert_lints=(
                {
                    "type": "equals",
                    "path": "metrics.loss_type",
                    "value": "mlm",
                    "message": "BERT cert must record MLM loss type.",
                },
                {
                    "type": "gte",
                    "path": "metrics.masked_tokens_total",
                    "value": "1",
                    "message": "BERT cert must report masked tokens.",
                },
            ),
        )

    if any(keyword in adapter_lower for keyword in ("llama", "mistral", "qwen")) or any(
        keyword in model_lower for keyword in ("llama", "mistral", "qwen")
    ):
        return ModelProfile(
            family="llama",
            default_loss="causal",
            make_tokenizer=_make_llama_tokenizer(model_id),
            default_metric="ppl_causal",
            default_provider="wikitext2",
            module_selectors=_llama_selectors(),
            invariants=("rope_rotary_embedding",),
            cert_lints=(
                {
                    "type": "equals",
                    "path": "metrics.loss_type",
                    "value": "causal",
                    "message": "LLaMA cert should report causal loss.",
                },
            ),
        )

    if any(
        keyword in adapter_lower for keyword in ("gpt", "neox", "opt", "phi")
    ) or any(keyword in model_lower for keyword in ("gpt", "neox", "opt", "phi")):
        return ModelProfile(
            family="gpt2",
            default_loss="causal",
            make_tokenizer=_make_gpt2_tokenizer(model_id),
            default_metric="ppl_causal",
            default_provider="wikitext2",
            module_selectors=_gpt2_selectors(),
            invariants=("causal_masking",),
            cert_lints=(
                {
                    "type": "equals",
                    "path": "metrics.loss_type",
                    "value": "causal",
                    "message": "GPT-style cert should record causal loss.",
                },
            ),
        )

    return ModelProfile(
        family="unknown",
        default_loss="causal",
        make_tokenizer=_make_unknown_tokenizer(model_id),
        default_metric="ppl_causal",
        default_provider="wikitext2",
        module_selectors=_unknown_selectors(),
        invariants=(),
        cert_lints=(),
    )


def resolve_tokenizer(profile: ModelProfile) -> tuple[PreTrainedTokenizerBase, str]:
    """
    Instantiate a tokenizer for the given profile and return it with its hash.
    """

    tokenizer, hash_value = profile.make_tokenizer()
    if not isinstance(hash_value, str) or not hash_value:
        hash_value = _hash_tokenizer(tokenizer)
    return tokenizer, hash_value
