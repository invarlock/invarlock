from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TokenizerFactory = Callable[..., tuple[Any, str]]


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
