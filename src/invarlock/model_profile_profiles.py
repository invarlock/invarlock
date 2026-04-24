from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def _helpers():
    import invarlock.model_profile as helpers

    return helpers


@dataclass(frozen=True)
class ModelProfile:
    """Captured capabilities for a recognised model family."""

    family: str
    default_loss: str
    make_tokenizer: Callable[[], tuple[Any, str]]
    default_metric: str = "ppl_causal"
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


def _rope_decoder_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.out_proj",
        ],
        "ffn": [
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate_proj",
        ],
    }


def _gpt_oss_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
        "ffn": [
            "mlp.router",
            "mlp.experts",
        ],
    }


def _phi_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.dense",
            "self_attn.o_proj",
            "self_attn.qkv_proj",
        ],
        "ffn": [
            "mlp.fc1",
            "mlp.fc2",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ],
    }


def _unknown_selectors() -> dict[str, list[str]]:
    return {
        "attention": ["attention"],
        "ffn": [],
    }


def _make_bert_tokenizer(model_id: str):
    def factory() -> tuple[Any, str]:
        helpers = _helpers()
        tokenizer = helpers._load_tokenizer_for_model(model_id, family_label="BERT")
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
        hash_value = helpers._hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_gpt2_tokenizer(model_id: str):
    def factory() -> tuple[Any, str]:
        helpers = _helpers()
        tokenizer = helpers._load_tokenizer_for_model(model_id, family_label="causal")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hash_value = helpers._hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_causal_auto_tokenizer(model_id: str):
    def factory() -> tuple[Any, str]:
        helpers = _helpers()
        tokenizer = helpers._load_tokenizer_for_model(model_id, family_label="causal")
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        if hasattr(tokenizer, "add_bos_token"):
            tokenizer.add_bos_token = True
        if getattr(tokenizer, "pad_token", None) is None:
            raise ValueError(
                f"Tokenizer for '{model_id}' does not define a pad token and no EOS fallback is available."
            )
        hash_value = helpers._hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_unknown_tokenizer(model_id: str):
    def factory() -> tuple[Any, str]:
        helpers = _helpers()
        tokenizer = helpers._load_tokenizer_for_model(model_id, family_label="text")
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        hash_value = helpers._hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def detect_model_profile(model_id: str, adapter: str | None = None) -> ModelProfile:
    """Infer the model family and provide profile metadata used for evaluation."""

    helpers = _helpers()
    adapter_lower = (adapter or "").lower()
    model_lower, arch_blob, is_encoder_decoder = helpers._profile_hints(model_id)
    masked_arch = "maskedlm" in arch_blob
    causal_arch = "causallm" in arch_blob or "forcausallm" in arch_blob
    seq2seq_arch = "conditionalgeneration" in arch_blob or "seq2seqlm" in arch_blob

    if (
        any(
            keyword in adapter_lower
            for keyword in ("hf_mlm", "bert", "roberta", "deberta")
        )
        or masked_arch
        or any(keyword in model_lower for keyword in ("bert", "roberta", "deberta"))
    ):
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
                    "path": "primary_metric.kind",
                    "value": "ppl_mlm",
                    "message": "BERT evaluation report must use MLM metric.",
                },
                {
                    "type": "gte",
                    "path": "telemetry.masked_tokens_total",
                    "value": "1",
                    "message": "BERT evaluation report must report masked tokens.",
                },
            ),
        )

    if any(keyword in adapter_lower for keyword in ("hf_seq2seq", "t5", "bart")) or (
        is_encoder_decoder
        or (
            seq2seq_arch
            and not any(
                keyword in model_lower
                for keyword in ("gemma3", "gemma4", "mistral3", "ministral")
            )
        )
        or any(keyword in model_lower for keyword in ("t5", "bart"))
    ):
        return ModelProfile(
            family="seq2seq",
            default_loss="seq2seq",
            make_tokenizer=_make_unknown_tokenizer(model_id),
            default_metric="ppl_seq2seq",
            default_provider="wikitext2",
            module_selectors=_unknown_selectors(),
            invariants=(),
            cert_lints=(),
        )

    causal_family_aliases = (
        ("mixtral", "mixtral"),
        ("gpt-oss", "gpt_oss"),
        ("gpt_oss", "gpt_oss"),
        ("ministral", "mistral"),
        ("mistral", "mistral"),
        ("qwen", "qwen"),
        ("yi", "yi"),
        ("llama", "llama"),
        ("gemma", "gemma"),
        ("olmo", "olmo"),
    )
    if any(
        keyword in adapter_lower for keyword, _family in causal_family_aliases
    ) or any(keyword in model_lower for keyword, _family in causal_family_aliases):
        family = "causal"
        for keyword, mapped_family in causal_family_aliases:
            if keyword in adapter_lower or keyword in model_lower:
                family = mapped_family
                break
        module_selectors = (
            _gpt_oss_selectors() if family == "gpt_oss" else _rope_decoder_selectors()
        )
        return ModelProfile(
            family=family,
            default_loss="causal",
            make_tokenizer=_make_causal_auto_tokenizer(model_id),
            default_metric="ppl_causal",
            default_provider="wikitext2",
            module_selectors=module_selectors,
            invariants=("rope_rotary_embedding",),
            cert_lints=(
                {
                    "type": "equals",
                    "path": "primary_metric.kind",
                    "value": "ppl_causal",
                    "message": "Causal evaluation report must use causal ppl metric.",
                },
            ),
        )

    if any(keyword in adapter_lower for keyword in ("phi",)) or any(
        keyword in model_lower for keyword in ("phi",)
    ):
        family = "phi"
        if any(keyword in adapter_lower for keyword in ("phi4", "phi-4", "phi_4")) or (
            any(keyword in model_lower for keyword in ("phi4", "phi-4", "phi_4"))
        ):
            family = "phi4"
        return ModelProfile(
            family=family,
            default_loss="causal",
            make_tokenizer=_make_causal_auto_tokenizer(model_id),
            default_metric="ppl_causal",
            default_provider="wikitext2",
            module_selectors=_phi_selectors(),
            invariants=("causal_masking",),
            cert_lints=(
                {
                    "type": "equals",
                    "path": "primary_metric.kind",
                    "value": "ppl_causal",
                    "message": "Causal evaluation report must use causal ppl metric.",
                },
            ),
        )

    if (
        any(keyword in adapter_lower for keyword in ("gpt", "neox", "opt"))
        or causal_arch
        or any(keyword in model_lower for keyword in ("gpt", "neox", "opt"))
    ):
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
                    "path": "primary_metric.kind",
                    "value": "ppl_causal",
                    "message": "GPT-style evaluation report must use causal ppl metric.",
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


__all__ = [
    "ModelProfile",
    "detect_model_profile",
]
