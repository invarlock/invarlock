from __future__ import annotations

from types import SimpleNamespace


class _ProviderConfig:
    def __init__(self, kind: str, **items: object) -> None:
        self._data = {"kind": kind, **items}

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._data.get(key, default)


class _DummyTokenizer:
    name_or_path = "tok"
    vocab_size = 123
    bos_token = "<s>"
    eos_token = "</s>"
    pad_token = "<pad>"
    add_prefix_space = False


class _Seq2SeqProvider:
    name = "seq2seq"
    last_preview_labels = [[11, 12]]
    last_final_labels = [[21, 22]]
    stratification_stats = {"mode": "balanced"}
    scorer_profile = {"kind": "seq2seq"}


class _TokenizerWithoutName:
    vocab_size = 321
    bos_token = None
    eos_token = None
    pad_token = None
    add_prefix_space = None


def _cfg(*, provider: object, release: bool = False) -> SimpleNamespace:
    del release
    return SimpleNamespace(
        dataset=SimpleNamespace(
            provider=provider,
            seq_len=8,
            stride=4,
            seed=43,
            dataset_name="demo",
            split="validation",
        ),
        eval=SimpleNamespace(capacity_fast=False),
        guards=SimpleNamespace(variance=SimpleNamespace(max_calib=16)),
        auto=SimpleNamespace(tier="balanced"),
    )
