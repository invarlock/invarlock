from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

_TRANSFORMERS_UNSET = object()
AutoTokenizer: Any = _TRANSFORMERS_UNSET
_TOKENIZERS_UNSET = object()
TokenizerImpl: Any = _TOKENIZERS_UNSET
_TOKENIZER_LOOKUP_ERRORS = (RuntimeError, TypeError, ValueError)
_TOKENIZER_LOAD_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)


class PreTrainedTokenizerBase:
    """Failing sentinel used when the optional transformers package is absent."""

    pad_token: Any = None
    eos_token: Any = None
    sep_token: Any = None
    cls_token: Any = None
    add_bos_token: bool = False
    name_or_path: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(
            "Tokenization requires the 'transformers' extra. "
            "Install the invarlock adapters extra to enable tokenizer loading."
        )

    def get_vocab(self) -> dict[str, int]:
        return {}


TokenizerFactory = Callable[[], tuple[PreTrainedTokenizerBase, str]]


def _ensure_transformers_tokenizer_support() -> Any:
    global AutoTokenizer
    if AutoTokenizer is _TRANSFORMERS_UNSET:
        try:
            from transformers import AutoTokenizer as _AutoTokenizer
        except ModuleNotFoundError:
            AutoTokenizer = None
        else:  # pragma: no cover - transformers optional
            AutoTokenizer = _AutoTokenizer
    return AutoTokenizer


def _ensure_tokenizers_support() -> Any:
    global TokenizerImpl
    if TokenizerImpl is _TOKENIZERS_UNSET:
        try:
            tokenizers_module = importlib.import_module("tokenizers")
        except ModuleNotFoundError:
            TokenizerImpl = None
        else:  # pragma: no cover - tokenizers optional
            TokenizerImpl = tokenizers_module.Tokenizer
    return None if TokenizerImpl is _TOKENIZERS_UNSET else TokenizerImpl


def _hash_tokenizer(tokenizer: PreTrainedTokenizerBase) -> str:
    if hasattr(tokenizer, "get_vocab"):
        vocab_mapping = tokenizer.get_vocab()
    else:
        vocab_mapping = getattr(tokenizer, "vocab", {})
    if hasattr(vocab_mapping, "items"):
        vocab_items = list(vocab_mapping.items())
    else:
        vocab_items = []

    hasher = hashlib.blake2s(digest_size=16)
    for token, idx in sorted(vocab_items, key=lambda item: str(item[0])):
        token_str = token if isinstance(token, str) else str(token)
        hasher.update(token_str.encode("utf-8", "ignore"))
        try:
            idx_int = int(idx)
        except (TypeError, ValueError, OverflowError):
            hasher.update(str(idx).encode("utf-8", "ignore"))
            continue
        if idx_int < 0:
            hasher.update(str(idx_int).encode("utf-8", "ignore"))
            continue
        byte_len = max(1, (idx_int.bit_length() + 7) // 8)
        hasher.update(idx_int.to_bytes(byte_len, "little", signed=False))

    hasher.update(tokenizer.__class__.__name__.encode("utf-8", "ignore"))
    name_path = getattr(tokenizer, "name_or_path", "")
    hasher.update(str(name_path).encode("utf-8", "ignore"))
    return hasher.hexdigest()


def _read_local_hf_config(model_id: str) -> dict[str, Any] | None:
    """Read a local Hugging Face config.json when `model_id` is a directory."""

    try:
        cfg_path = Path(model_id) / "config.json"
    except (OSError, TypeError, ValueError):
        return None
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_local_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_special_token_value(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    return None


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
            value = _coerce_special_token_value(data.get(key))
            if value is not None:
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
        except _TOKENIZER_LOOKUP_ERRORS:
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
    except (OSError, TypeError, ValueError):
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
    except (OSError, TypeError, ValueError):
        return None
    return _LocalFastTokenizer(
        tokenizer=tokenizer,
        name_or_path=str(candidate_path),
        special_tokens=_load_local_tokenizer_metadata(candidate_path),
    )


def _is_tokenizer_cache_miss(error: Exception) -> bool:
    if isinstance(error, FileNotFoundError):
        return True
    if error.__class__.__name__ in {"LocalEntryNotFoundError", "EntryNotFoundError"}:
        return True
    if not isinstance(error, (OSError, ValueError)):
        return False
    message = str(error).strip().lower()
    return any(
        snippet in message
        for snippet in (
            "no such file",
            "not found",
            "could not locate",
            "missing cached",
            "files missing",
            "local files only",
            "cannot find",
            "couldn't find them in the cached files",
            "requested files in the disk cache",
            "outgoing traffic has been disabled",
            "couldn't connect to",
            "can't load the model",
            "can't load tokenizer",
            "is not a local folder",
            "is not a valid model identifier",
            "does not appear to have a file named",
        )
    )


def _is_slow_tokenizer_fallback_candidate(error: Exception) -> bool:
    if not isinstance(error, _TOKENIZER_LOAD_ERRORS):
        return False
    message = str(error).strip().lower()
    return any(
        snippet in message
        for snippet in (
            "couldn't instantiate the backend tokenizer",
            "you need to have sentencepiece",
            "you need to have sentencepiece or tiktoken installed",
            "convert a slow tokenizer",
        )
    )
