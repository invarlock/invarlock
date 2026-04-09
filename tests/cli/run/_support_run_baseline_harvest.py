from __future__ import annotations

from types import SimpleNamespace


class _Cfg:
    def __init__(self) -> None:
        self.dataset = SimpleNamespace(
            preview_n=1,
            final_n=1,
            seq_len=8,
            stride=8,
            provider="wikitext2",
            split="validation",
        )
