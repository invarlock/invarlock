from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.exceptions import DataError
from invarlock.eval import data_providers


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [len(text), 2, 0],
            "attention_mask": [1, 1, 0],
        }

    def encode(self, text, max_length, truncation, padding):
        del truncation, padding
        return ([len(text), 2] + [0] * max_length)[:max_length]


def test_wikitext_load_keeps_larger_cache_and_exposes_runtime_info(monkeypatch):
    datasets = [
        [
            {"text": "alpha sample long enough for validation"},
            {"text": "beta sample long enough for validation"},
        ],
        [{"text": "gamma sample long enough for validation"}],
    ]
    monkeypatch.setattr(data_providers, "_require_load_dataset", lambda _message: None)
    monkeypatch.setattr(
        data_providers,
        "load_dataset_with_cache_fallback",
        lambda *_args, **_kwargs: datasets.pop(0),
    )
    provider = data_providers.WikiText2Provider()

    assert len(provider.load(max_samples=2)) == 2
    # Requesting more bypasses the cache, but the shorter reload must not
    # replace the previously observed larger deterministic sample pool.
    assert provider.load(max_samples=3) == ["gamma sample long enough for validation"]
    assert len(provider._texts_cache["validation"]) == 2
    assert provider.scorer_profile is None
    assert provider.info()["dataset"] == "wikitext-2-raw-v1"


def test_wikitext_tokenize_samples_returns_exact_window(monkeypatch):
    monkeypatch.setattr(data_providers, "_require_load_dataset", lambda _message: None)
    provider = data_providers.WikiText2Provider()
    monkeypatch.setattr(
        provider,
        "_collect_tokenized_samples",
        lambda _texts, _indices, _tokenizer, _seq_len: [(4, [9, 2, 0], [1, 1, 0], 2)],
    )

    window = provider._tokenize_samples(
        ["sample"], [4], SimpleNamespace(), 3, "preview"
    )
    assert window.input_ids == [[9, 2, 0]]
    assert window.attention_masks == [[1, 1, 0]]
    assert window.indices == [4]


def test_synthetic_provider_expands_beyond_named_variations_and_caches():
    provider = data_providers.SyntheticProvider(
        base_samples=["A sufficiently long deterministic sample sentence."]
    )

    samples = provider.load(max_samples=8)
    assert len(samples) == 8
    assert samples[-1].endswith("[synthetic #7]")
    assert provider.load(max_samples=8) is samples
    assert provider.info() == {
        "name": "synthetic",
        "type": "dataset_provider",
        "dataset": "synthetic",
        "source": "generated",
        "deterministic": True,
        "base_samples": 1,
    }


def test_synthetic_tokenizer_empty_output_fails_closed(monkeypatch):
    provider = data_providers.SyntheticProvider(base_samples=["long enough sample"])
    monkeypatch.setattr(
        data_providers,
        "tokenize_texts_padded",
        lambda *_args, **_kwargs: ([], [], []),
    )

    with pytest.raises(DataError, match="failed to tokenize synthetic samples"):
        provider._simple_tokenize(["sample"], _Tokenizer(), 4, [0])
