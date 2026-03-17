from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

_TRANSFORMERS_UNSET = object()
AutoTokenizer: Any = _TRANSFORMERS_UNSET
_TOKENIZERS_UNSET = object()
TokenizerImpl: Any = _TOKENIZERS_UNSET


class PreTrainedTokenizerBase:
    """Lightweight stub used when transformers is not installed."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(
            "Tokenization requires the 'transformers' extra. "
            "Install it with: pip install 'invarlock[adapters]'."
        )


def _ensure_transformers_tokenizer_support() -> Any:
    global AutoTokenizer, PreTrainedTokenizerBase
    if AutoTokenizer is _TRANSFORMERS_UNSET:
        try:
            from transformers import AutoTokenizer as _AutoTokenizer
            from transformers.tokenization_utils_base import (
                PreTrainedTokenizerBase as _PreTrainedTokenizerBase,
            )
        except Exception:
            AutoTokenizer = None
        else:  # pragma: no cover - transformers optional
            AutoTokenizer = _AutoTokenizer
            PreTrainedTokenizerBase = _PreTrainedTokenizerBase
    return AutoTokenizer


def _ensure_tokenizers_support() -> Any:
    global TokenizerImpl
    if TokenizerImpl is _TOKENIZERS_UNSET:
        try:
            from tokenizers import Tokenizer as _TokenizerImpl
        except Exception:
            TokenizerImpl = None
        else:  # pragma: no cover - tokenizers optional
            TokenizerImpl = _TokenizerImpl
    return None if TokenizerImpl is _TOKENIZERS_UNSET else TokenizerImpl


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


def _read_local_hf_config(model_id: str) -> dict[str, Any] | None:
    """Read a local Hugging Face config.json when `model_id` is a directory."""

    try:
        cfg_path = Path(model_id) / "config.json"
    except Exception:
        return None
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_local_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_local_tokenizer_metadata(model_dir: Path) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "bos_token": None,
        "cls_token": None,
        "eos_token": None,
        "mask_token": None,
        "pad_token": None,
        "sep_token": None,
        "unk_token": None,
    }
    for path in (
        model_dir / "special_tokens_map.json",
        model_dir / "tokenizer_config.json",
    ):
        data = _read_local_json_file(path)
        if not isinstance(data, dict):
            continue
        for key in tuple(metadata):
            value = data.get(key)
            if isinstance(value, str) and value:
                metadata[key] = value
    return metadata


class _LocalFastTokenizer(PreTrainedTokenizerBase):
    """Lightweight wrapper around `tokenizers.Tokenizer` for local checkpoints."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        name_or_path: str,
        special_tokens: dict[str, str | None],
    ) -> None:
        self._tokenizer = tokenizer
        self._vocab = dict(tokenizer.get_vocab())
        self.name_or_path = name_or_path
        self.add_bos_token = False
        self._special_tokens = dict(special_tokens)

    def _token_to_id(self, token: str | None) -> int | None:
        if not token:
            return None
        try:
            token_id = self._tokenizer.token_to_id(token)
        except Exception:
            return None
        return None if token_id is None else int(token_id)

    @property
    def bos_token(self) -> str | None:
        return self._special_tokens.get("bos_token")

    @bos_token.setter
    def bos_token(self, value: str | None) -> None:
        self._special_tokens["bos_token"] = value

    @property
    def bos_token_id(self) -> int | None:
        return self._token_to_id(self.bos_token)

    @property
    def cls_token(self) -> str | None:
        return self._special_tokens.get("cls_token")

    @cls_token.setter
    def cls_token(self, value: str | None) -> None:
        self._special_tokens["cls_token"] = value

    @property
    def cls_token_id(self) -> int | None:
        return self._token_to_id(self.cls_token)

    @property
    def eos_token(self) -> str | None:
        return self._special_tokens.get("eos_token")

    @eos_token.setter
    def eos_token(self, value: str | None) -> None:
        self._special_tokens["eos_token"] = value

    @property
    def eos_token_id(self) -> int | None:
        return self._token_to_id(self.eos_token)

    @property
    def mask_token(self) -> str | None:
        return self._special_tokens.get("mask_token")

    @mask_token.setter
    def mask_token(self, value: str | None) -> None:
        self._special_tokens["mask_token"] = value

    @property
    def mask_token_id(self) -> int | None:
        return self._token_to_id(self.mask_token)

    @property
    def pad_token(self) -> str | None:
        return self._special_tokens.get("pad_token")

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._special_tokens["pad_token"] = value

    @property
    def pad_token_id(self) -> int | None:
        return self._token_to_id(self.pad_token)

    @property
    def sep_token(self) -> str | None:
        return self._special_tokens.get("sep_token")

    @sep_token.setter
    def sep_token(self, value: str | None) -> None:
        self._special_tokens["sep_token"] = value

    @property
    def sep_token_id(self) -> int | None:
        return self._token_to_id(self.sep_token)

    @property
    def unk_token(self) -> str | None:
        return self._special_tokens.get("unk_token")

    @unk_token.setter
    def unk_token(self, value: str | None) -> None:
        self._special_tokens["unk_token"] = value

    @property
    def unk_token_id(self) -> int | None:
        return self._token_to_id(self.unk_token)

    @property
    def all_special_ids(self) -> list[int]:
        ids: list[int] = []
        for key in (
            "bos_token",
            "cls_token",
            "eos_token",
            "mask_token",
            "pad_token",
            "sep_token",
            "unk_token",
        ):
            token_id = self._token_to_id(self._special_tokens.get(key))
            if token_id is not None and token_id not in ids:
                ids.append(token_id)
        return ids

    @property
    def vocab_size(self) -> int:
        return int(len(self._vocab))

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _prepare_ids(
        self,
        token_ids: list[int],
        *,
        truncation: bool,
        max_length: int | None,
        padding: str | bool | None,
    ) -> tuple[list[int], list[int]]:
        ids = [int(token) for token in token_ids]
        bos_token_id = self.bos_token_id
        if self.add_bos_token and bos_token_id is not None:
            if not ids or ids[0] != bos_token_id:
                ids = [bos_token_id, *ids]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        if padding == "max_length" and max_length is not None and len(ids) < max_length:
            pad_token_id = int(self.pad_token_id or 0)
            pad_count = max_length - len(ids)
            ids.extend([pad_token_id] * pad_count)
            attention_mask.extend([0] * pad_count)
        return ids, attention_mask

    def encode(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        padding: str | bool | None = None,
        **_: Any,
    ) -> list[int]:
        encoding = self._tokenizer.encode(text)
        ids, _attention_mask = self._prepare_ids(
            [int(token) for token in encoding.ids],
            truncation=truncation,
            max_length=max_length,
            padding=padding,
        )
        return ids

    def __call__(
        self,
        text_or_texts: str | list[str],
        *,
        truncation: bool = False,
        padding: str | bool | None = None,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
            single = True
        else:
            texts = [str(text) for text in text_or_texts]
            single = False
        encodings = self._tokenizer.encode_batch(texts)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for encoding in encodings:
            ids, attention_mask = self._prepare_ids(
                [int(token) for token in encoding.ids],
                truncation=truncation,
                max_length=max_length,
                padding=padding,
            )
            input_ids.append(ids)
            attention_masks.append(attention_mask)
        result: dict[str, Any] = {
            "input_ids": input_ids[0] if single else input_ids,
        }
        if return_attention_mask:
            result["attention_mask"] = attention_masks[0] if single else attention_masks
        return result


def _load_local_fast_tokenizer(candidate: str) -> PreTrainedTokenizerBase | None:
    try:
        candidate_path = Path(candidate)
    except Exception:
        return None
    tokenizer_path = candidate_path / "tokenizer.json"
    if (
        not candidate_path.exists()
        or not candidate_path.is_dir()
        or not tokenizer_path.exists()
    ):
        return None
    tokenizer_impl = _ensure_tokenizers_support()
    if tokenizer_impl is None:
        return None
    try:
        tokenizer = tokenizer_impl.from_file(str(tokenizer_path))
    except Exception:
        return None
    return _LocalFastTokenizer(
        tokenizer=tokenizer,
        name_or_path=str(candidate_path),
        special_tokens=_load_local_tokenizer_metadata(candidate_path),
    )


def _tokenizer_candidates(model_id: str) -> list[str]:
    """Return ordered tokenizer identifiers tied to the requested model."""

    raw_candidates = [str(model_id).strip()]
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
    model_id: str, *, family_label: str
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
            "Install it with: pip install 'invarlock[adapters]'."
        )

    for candidate in candidates:
        try:
            tokenizer = tokenizer_factory.from_pretrained(
                candidate, local_files_only=True
            )
            return cast("PreTrainedTokenizerBase", tokenizer)
        except Exception:
            continue

    for candidate in candidates:
        try:
            candidate_path = Path(candidate)
        except Exception:
            candidate_path = None
        if candidate_path is not None and candidate_path.exists():
            continue
        try:
            tokenizer = tokenizer_factory.from_pretrained(candidate)
            return cast("PreTrainedTokenizerBase", tokenizer)
        except Exception:
            continue

    raise RuntimeError(
        f"Unable to load a {family_label} tokenizer for '{model_id}'. "
        "Set INVARLOCK_ALLOW_NETWORK=1 to allow fetching from the Hugging Face Hub, "
        "or pre-cache the tokenizer locally."
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


def _rope_decoder_selectors() -> dict[str, list[str]]:
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
        tokenizer = _load_tokenizer_for_model(model_id, family_label="BERT")
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
        tokenizer = _load_tokenizer_for_model(model_id, family_label="causal")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hash_value = _hash_tokenizer(tokenizer)
        return tokenizer, hash_value

    return factory


def _make_causal_auto_tokenizer(model_id: str):
    def factory() -> tuple[PreTrainedTokenizerBase, str]:
        tokenizer = _load_tokenizer_for_model(model_id, family_label="causal")
        # Ensure padding/bos tokens are configured so downstream encoding
        # yields stable non-zero ids and a valid attention mask regardless of
        # environment defaults or tokenizer variants.
        # Prefer EOS as pad token when no explicit pad token is defined.
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
        # Some causal tokenizers default to not adding a BOS token on encode;
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
        tokenizer = _load_tokenizer_for_model(model_id, family_label="text")
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
    model_lower, arch_blob, is_encoder_decoder = _profile_hints(model_id)
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
                    "message": "BERT cert must use MLM metric.",
                },
                {
                    "type": "gte",
                    "path": "telemetry.masked_tokens_total",
                    "value": "1",
                    "message": "BERT cert must report masked tokens.",
                },
            ),
        )

    if any(keyword in adapter_lower for keyword in ("hf_seq2seq", "t5", "bart")) or (
        is_encoder_decoder
        or seq2seq_arch
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

    if any(
        keyword in adapter_lower for keyword in ("mistral", "mixtral", "qwen", "yi")
    ) or any(
        keyword in model_lower for keyword in ("mistral", "mixtral", "qwen", "yi")
    ):
        family = "causal"
        for keyword in ("mixtral", "mistral", "qwen", "yi"):
            if keyword in adapter_lower or keyword in model_lower:
                family = keyword
                break
        return ModelProfile(
            family=family,
            default_loss="causal",
            make_tokenizer=_make_causal_auto_tokenizer(model_id),
            default_metric="ppl_causal",
            default_provider="wikitext2",
            module_selectors=_rope_decoder_selectors(),
            invariants=("rope_rotary_embedding",),
            cert_lints=(
                {
                    "type": "equals",
                    "path": "primary_metric.kind",
                    "value": "ppl_causal",
                    "message": "Causal cert must use causal ppl metric.",
                },
            ),
        )

    if (
        any(keyword in adapter_lower for keyword in ("gpt", "neox", "opt", "phi"))
        or causal_arch
        or any(keyword in model_lower for keyword in ("gpt", "neox", "opt", "phi"))
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
                    "message": "GPT-style cert must use causal ppl metric.",
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
