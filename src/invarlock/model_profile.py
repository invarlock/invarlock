from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from invarlock.runtime_security import network_allowed

from . import model_profile_tokenizers as _tokenizers

_TRANSFORMERS_UNSET = _tokenizers._TRANSFORMERS_UNSET
AutoTokenizer: Any = _tokenizers.AutoTokenizer
_TOKENIZERS_UNSET = _tokenizers._TOKENIZERS_UNSET
TokenizerImpl: Any = _tokenizers.TokenizerImpl
_TOKENIZER_LOOKUP_ERRORS = _tokenizers._TOKENIZER_LOOKUP_ERRORS
_TOKENIZER_LOAD_ERRORS = _tokenizers._TOKENIZER_LOAD_ERRORS
PreTrainedTokenizerBase = _tokenizers.PreTrainedTokenizerBase
TokenizerFactory = Callable[..., tuple[PreTrainedTokenizerBase, str]]
_LocalFastTokenizer = _tokenizers._LocalFastTokenizer

_TOKENIZER_CANDIDATE_OVERRIDES: dict[str, tuple[str, ...]] = {
    # These DeepSeek releases publish Qwen-compatible model configs without
    # tokenizer files in the HF snapshot used by the evidence runner. Loading
    # the model id directly can produce all-pad samples; prefer the matching
    # base-family tokenizer before falling back to the model id.
    "deepseek-ai/deepseek-r1-distill-qwen-14b": ("Qwen/Qwen2.5-14B",),
    "deepseek-ai/deepseek-r1-0528-qwen3-8b": ("Qwen/Qwen3-8B",),
}


def _sync_tokenizer_state() -> None:
    _tokenizers.AutoTokenizer = AutoTokenizer
    _tokenizers.TokenizerImpl = TokenizerImpl


def _refresh_tokenizer_state() -> None:
    globals()["AutoTokenizer"] = _tokenizers.AutoTokenizer
    globals()["TokenizerImpl"] = _tokenizers.TokenizerImpl


def _call_tokenizer_helper(name: str, *args: Any, **kwargs: Any) -> Any:
    _sync_tokenizer_state()
    try:
        return getattr(_tokenizers, name)(*args, **kwargs)
    finally:
        _refresh_tokenizer_state()


def _ensure_transformers_tokenizer_support() -> Any:
    return _call_tokenizer_helper("_ensure_transformers_tokenizer_support")


def _ensure_tokenizers_support() -> Any:
    return _call_tokenizer_helper("_ensure_tokenizers_support")


def _hash_tokenizer(tokenizer: PreTrainedTokenizerBase) -> str:
    return _tokenizers._hash_tokenizer(tokenizer)


def _read_local_hf_config(model_id: str) -> dict[str, Any] | None:
    """Read a local Hugging Face config.json when `model_id` is a directory."""

    return _tokenizers._read_local_hf_config(model_id)


def _read_local_json_file(path: Path) -> dict[str, Any] | None:
    return _tokenizers._read_local_json_file(path)


def _coerce_special_token_value(value: Any) -> str | None:
    return _tokenizers._coerce_special_token_value(value)


def _load_local_tokenizer_metadata(model_dir: Path) -> dict[str, str | None]:
    return _tokenizers._load_local_tokenizer_metadata(model_dir)


def _load_local_fast_tokenizer(candidate: str) -> PreTrainedTokenizerBase | None:
    return cast(
        "PreTrainedTokenizerBase | None",
        _call_tokenizer_helper("_load_local_fast_tokenizer", candidate),
    )


def _is_tokenizer_cache_miss(error: Exception) -> bool:
    return bool(_tokenizers._is_tokenizer_cache_miss(error))


def _is_slow_tokenizer_fallback_candidate(error: Exception) -> bool:
    return bool(_tokenizers._is_slow_tokenizer_fallback_candidate(error))


def _load_tokenizer_with_factory_retry(
    tokenizer_factory: Any,
    candidate: str,
    *,
    local_files_only: bool,
    load_kwargs: dict[str, Any] | None = None,
) -> PreTrainedTokenizerBase:
    kwargs: dict[str, Any] = _resolve_tokenizer_load_kwargs(load_kwargs)
    if local_files_only:
        kwargs["local_files_only"] = True
    try:
        tokenizer = tokenizer_factory.from_pretrained(candidate, **kwargs)
        return cast("PreTrainedTokenizerBase", tokenizer)
    except _TOKENIZER_LOAD_ERRORS as exc:
        if not _is_slow_tokenizer_fallback_candidate(exc):
            raise
        try:
            tokenizer = tokenizer_factory.from_pretrained(
                candidate,
                use_fast=False,
                **kwargs,
            )
            return cast("PreTrainedTokenizerBase", tokenizer)
        except _TOKENIZER_LOAD_ERRORS as slow_exc:
            if not _is_slow_tokenizer_fallback_candidate(slow_exc):
                raise
            explicit_factory = _resolve_explicit_slow_tokenizer_factory(candidate)
            if explicit_factory is None:
                raise
            tokenizer = explicit_factory.from_pretrained(candidate, **kwargs)
            return cast("PreTrainedTokenizerBase", tokenizer)


def _resolve_tokenizer_load_kwargs(
    load_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(load_kwargs, dict):
        return {}
    resolved: dict[str, Any] = {}
    if "trust_remote_code" in load_kwargs:
        from invarlock.adapters.hf_loading import resolve_trust_remote_code

        resolved["trust_remote_code"] = resolve_trust_remote_code(
            {"trust_remote_code": load_kwargs.get("trust_remote_code")}
        )
    return resolved


def _merge_tokenizer_load_kwargs(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _resolve_explicit_slow_tokenizer_factory(candidate: str) -> Any | None:
    hint_blob, arch_blob, _ = _profile_hints(candidate)
    hint_space = f"{hint_blob} {arch_blob}".strip()
    for key, symbol_name in (
        ("deberta-v2", "DebertaV2Tokenizer"),
        ("deberta_v2", "DebertaV2Tokenizer"),
        ("debertav2", "DebertaV2Tokenizer"),
        ("deberta", "DebertaTokenizer"),
        ("distilbert", "DistilBertTokenizer"),
        ("roberta", "RobertaTokenizer"),
        ("albert", "AlbertTokenizer"),
        ("electra", "ElectraTokenizer"),
        ("bert", "BertTokenizer"),
    ):
        if key not in hint_space:
            continue
        try:
            import transformers

            return getattr(transformers, symbol_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            return None
    return None


def _tokenizer_candidates(model_id: str) -> list[str]:
    """Return ordered tokenizer identifiers tied to the requested model."""

    model_id_str = str(model_id).strip()
    raw_candidates = [
        *(_TOKENIZER_CANDIDATE_OVERRIDES.get(model_id_str.lower(), ())),
        model_id_str,
    ]
    cfg = _read_local_hf_config(model_id)
    if isinstance(cfg, dict):
        for key in (
            "tokenizer_name",
            "_name_or_path",
            "name_or_path",
            "base_model_name_or_path",
        ):
            value = cfg.get(key)
            if isinstance(value, str):
                raw_candidates.append(value.strip())

    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _load_tokenizer_for_model(
    model_id: str,
    *,
    family_label: str,
    load_kwargs: dict[str, Any] | None = None,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer without falling back to unrelated model families."""

    candidates = _tokenizer_candidates(model_id)
    for candidate in candidates:
        tokenizer = _load_local_fast_tokenizer(candidate)
        if tokenizer is not None:
            return tokenizer

    tokenizer_factory = _ensure_transformers_tokenizer_support()
    if tokenizer_factory is None:
        raise RuntimeError(
            f"{family_label} tokenizers require the 'transformers' extra. "
            "Install the invarlock adapters extra to enable tokenizer loading."
        )

    for candidate in candidates:
        try:
            return _load_tokenizer_with_factory_retry(
                tokenizer_factory,
                candidate,
                local_files_only=True,
                load_kwargs=load_kwargs,
            )
        except _TOKENIZER_LOAD_ERRORS as exc:
            if not _is_tokenizer_cache_miss(exc):
                raise
            continue

    if not network_allowed():
        raise RuntimeError(
            f"Unable to load a cached {family_label} tokenizer for '{model_id}'. "
            "Network tokenizer downloads are disabled."
        )

    for candidate in candidates:
        try:
            candidate_path = Path(candidate)
        except (OSError, TypeError, ValueError):
            candidate_path = None
        if candidate_path is not None and candidate_path.exists():
            continue
        try:
            return _load_tokenizer_with_factory_retry(
                tokenizer_factory,
                candidate,
                local_files_only=False,
                load_kwargs=load_kwargs,
            )
        except _TOKENIZER_LOAD_ERRORS as exc:
            if not _is_tokenizer_cache_miss(exc):
                raise
            continue

    raise RuntimeError(
        f"Unable to load a {family_label} tokenizer for '{model_id}' from the local cache or trusted remote candidates."
    )


def _profile_hints(model_id: str) -> tuple[str, str, bool]:
    """Collect lightweight model-family hints from path/config metadata."""

    cfg = _read_local_hf_config(model_id)
    model_type = ""
    arch_blob = ""
    is_encoder_decoder = False
    parts = [str(model_id or "").lower()]
    if isinstance(cfg, dict):
        model_type = str(cfg.get("model_type", "") or "").lower()
        arch_blob = " ".join(
            str(arch).lower()
            for arch in cfg.get("architectures", [])
            if isinstance(arch, str)
        )
        is_encoder_decoder = bool(cfg.get("is_encoder_decoder", False))
        if model_type:
            parts.append(model_type)
        if arch_blob:
            parts.append(arch_blob)
    return " ".join(parts), arch_blob, is_encoder_decoder


@dataclass(frozen=True)
class ModelProfile:
    """Captured capabilities for a recognised model family."""

    family: str
    default_loss: str
    make_tokenizer: TokenizerFactory
    default_metric: str = "ppl_causal"
    default_provider: str = "wikitext2"
    module_selectors: dict[str, list[str]] = field(default_factory=dict)
    invariants: tuple[str, ...] = ()
    cert_lints: tuple[dict[str, str], ...] = ()
    tokenizer_load_kwargs: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


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


def _make_bert_tokenizer(
    model_id: str,
    *,
    tokenizer_load_kwargs: dict[str, Any] | None = None,
) -> TokenizerFactory:
    def factory(
        *,
        tokenizer_load_kwargs: dict[str, Any] | None = None,
    ) -> tuple[PreTrainedTokenizerBase, str]:
        tokenizer = _load_tokenizer_for_model(
            model_id,
            family_label="BERT",
            load_kwargs=_merge_tokenizer_load_kwargs(
                factory.tokenizer_load_kwargs,
                tokenizer_load_kwargs,
            ),
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

    factory.tokenizer_load_kwargs = dict(tokenizer_load_kwargs or {})  # type: ignore[attr-defined]
    return factory


def _make_gpt2_tokenizer(
    model_id: str,
    *,
    tokenizer_load_kwargs: dict[str, Any] | None = None,
) -> TokenizerFactory:
    def factory(
        *,
        tokenizer_load_kwargs: dict[str, Any] | None = None,
    ) -> tuple[PreTrainedTokenizerBase, str]:
        tokenizer = _load_tokenizer_for_model(
            model_id,
            family_label="causal",
            load_kwargs=_merge_tokenizer_load_kwargs(
                factory.tokenizer_load_kwargs,
                tokenizer_load_kwargs,
            ),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    factory.tokenizer_load_kwargs = dict(tokenizer_load_kwargs or {})  # type: ignore[attr-defined]
    return factory


def _make_causal_auto_tokenizer(
    model_id: str,
    *,
    tokenizer_load_kwargs: dict[str, Any] | None = None,
) -> TokenizerFactory:
    def factory(
        *,
        tokenizer_load_kwargs: dict[str, Any] | None = None,
    ) -> tuple[PreTrainedTokenizerBase, str]:
        tokenizer = _load_tokenizer_for_model(
            model_id,
            family_label="causal",
            load_kwargs=_merge_tokenizer_load_kwargs(
                factory.tokenizer_load_kwargs,
                tokenizer_load_kwargs,
            ),
        )
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
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    factory.tokenizer_load_kwargs = dict(tokenizer_load_kwargs or {})  # type: ignore[attr-defined]
    return factory


def _make_unknown_tokenizer(
    model_id: str,
    *,
    tokenizer_load_kwargs: dict[str, Any] | None = None,
) -> TokenizerFactory:
    def factory(
        *,
        tokenizer_load_kwargs: dict[str, Any] | None = None,
    ) -> tuple[PreTrainedTokenizerBase, str]:
        tokenizer = _load_tokenizer_for_model(
            model_id,
            family_label="text",
            load_kwargs=_merge_tokenizer_load_kwargs(
                factory.tokenizer_load_kwargs,
                tokenizer_load_kwargs,
            ),
        )
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    factory.tokenizer_load_kwargs = dict(tokenizer_load_kwargs or {})  # type: ignore[attr-defined]
    return factory


def detect_model_profile(
    model_id: str,
    adapter: str | None = None,
    tokenizer_load_kwargs: dict[str, Any] | None = None,
) -> ModelProfile:
    """Infer the model family and provide profile metadata used for evaluation."""

    adapter_lower = (adapter or "").lower()
    model_lower, arch_blob, is_encoder_decoder = _profile_hints(model_id)
    tokenizer_load_kwargs = dict(tokenizer_load_kwargs or {})
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
            make_tokenizer=_make_bert_tokenizer(
                model_id,
                tokenizer_load_kwargs=tokenizer_load_kwargs,
            ),
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
            tokenizer_load_kwargs=tokenizer_load_kwargs,
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
            make_tokenizer=_make_unknown_tokenizer(
                model_id,
                tokenizer_load_kwargs=tokenizer_load_kwargs,
            ),
            default_metric="ppl_seq2seq",
            default_provider="wikitext2",
            module_selectors=_unknown_selectors(),
            invariants=(),
            cert_lints=(),
            tokenizer_load_kwargs=tokenizer_load_kwargs,
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
            make_tokenizer=_make_causal_auto_tokenizer(
                model_id,
                tokenizer_load_kwargs=tokenizer_load_kwargs,
            ),
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
            tokenizer_load_kwargs=tokenizer_load_kwargs,
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
            make_tokenizer=_make_causal_auto_tokenizer(
                model_id,
                tokenizer_load_kwargs=tokenizer_load_kwargs,
            ),
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
            tokenizer_load_kwargs=tokenizer_load_kwargs,
        )

    if (
        any(keyword in adapter_lower for keyword in ("gpt", "neox", "opt"))
        or causal_arch
        or any(keyword in model_lower for keyword in ("gpt", "neox", "opt"))
    ):
        return ModelProfile(
            family="gpt2",
            default_loss="causal",
            make_tokenizer=_make_gpt2_tokenizer(
                model_id,
                tokenizer_load_kwargs=tokenizer_load_kwargs,
            ),
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
            tokenizer_load_kwargs=tokenizer_load_kwargs,
        )

    return ModelProfile(
        family="unknown",
        default_loss="causal",
        make_tokenizer=_make_unknown_tokenizer(
            model_id,
            tokenizer_load_kwargs=tokenizer_load_kwargs,
        ),
        default_metric="ppl_causal",
        default_provider="wikitext2",
        module_selectors=_unknown_selectors(),
        invariants=(),
        cert_lints=(),
        tokenizer_load_kwargs=tokenizer_load_kwargs,
    )


def resolve_tokenizer(profile: ModelProfile) -> tuple[PreTrainedTokenizerBase, str]:
    """
    Instantiate a tokenizer for the given profile and return it with its hash.
    """

    tokenizer_load_kwargs = getattr(profile, "tokenizer_load_kwargs", None)
    try:
        signature = inspect.signature(profile.make_tokenizer)
    except (TypeError, ValueError):
        signature = None
    supports_load_kwargs = False
    if signature is not None:
        supports_load_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == "tokenizer_load_kwargs"
            for parameter in signature.parameters.values()
        )
    if supports_load_kwargs:
        tokenizer, hash_value = profile.make_tokenizer(
            tokenizer_load_kwargs=tokenizer_load_kwargs
        )
    else:
        tokenizer, hash_value = profile.make_tokenizer()
    if not isinstance(hash_value, str) or not hash_value:
        hash_value = _hash_tokenizer(tokenizer)
    return tokenizer, hash_value
