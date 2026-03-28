"""
Canonical evaluation data entrypoint.

Owns the provider registry and the public eval-data import surface.
"""

from __future__ import annotations

from typing import Any

from invarlock.core.exceptions import ValidationError as _ValErr

from .data_providers import (
    DatasetProvider,
    HFSeq2SeqProvider,
    HFTextProvider,
    LocalJSONLPairsProvider,
    LocalJSONLProvider,
    SyntheticProvider,
    WikiText2Provider,
)
from .data_support import EventEmitter
from .data_windows import EvaluationWindow, compute_window_hash
from .providers.seq2seq import Seq2SeqProvider

_PROVIDERS: dict[str, type[object]] = {
    "wikitext2": WikiText2Provider,
    "synthetic": SyntheticProvider,
    "hf_text": HFTextProvider,
    "local_jsonl": LocalJSONLProvider,
    "seq2seq": Seq2SeqProvider,
    "hf_seq2seq": HFSeq2SeqProvider,
    "local_jsonl_pairs": LocalJSONLPairsProvider,
}


def get_provider(
    name: str, *, emit: EventEmitter | None = None, **kwargs: Any
) -> DatasetProvider:
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise _ValErr(
            code="E308",
            message="PROVIDER-NOT-FOUND: unknown dataset provider",
            details={"provider": name, "available": available},
        )

    provider_class = _PROVIDERS[name]
    init_kwargs = dict(kwargs)
    init_kwargs["emit"] = emit
    return provider_class(**init_kwargs)  # type: ignore[call-arg,return-value]


def list_providers() -> list[str]:
    return list(_PROVIDERS.keys())


__all__ = [
    "DatasetProvider",
    "EvaluationWindow",
    "HFTextProvider",
    "LocalJSONLProvider",
    "LocalJSONLPairsProvider",
    "SyntheticProvider",
    "WikiText2Provider",
    "compute_window_hash",
    "get_provider",
    "list_providers",
]
