from __future__ import annotations

import os

import invarlock.eval.data as data_mod


class _EncodeTokenizer:
    pad_token_id = 0

    def encode(self, text, truncation=True, max_length=8):
        ids = list(range(1, min(len(text), max_length) + 1))
        return ids


def _fake_dataset(num_items: int = 50):
    return [{"text": f"sample text {i} with enough length"} for i in range(num_items)]


def test_wikitext2_estimate_capacity_fast_mode(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True, raising=False)
    monkeypatch.setattr(data_mod, "load_dataset", lambda *a, **k: _fake_dataset(30))
    provider = data_mod.WikiText2Provider()
    env_backup = os.environ.get("INVARLOCK_CAPACITY_FAST")
    os.environ["INVARLOCK_CAPACITY_FAST"] = "1"
    try:
        cap = provider.estimate_capacity(
            tokenizer=_EncodeTokenizer(), seq_len=8, stride=4, fast_mode=False
        )
    finally:
        if env_backup is None:
            os.environ.pop("INVARLOCK_CAPACITY_FAST", None)
        else:
            os.environ["INVARLOCK_CAPACITY_FAST"] = env_backup
    assert cap["available_unique"] == cap["available_nonoverlap"]
    assert cap["stride"] == 4


def test_wikitext2_windows_with_stubbed_tokenizer(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True, raising=False)
    monkeypatch.setattr(data_mod, "load_dataset", lambda *a, **k: _fake_dataset(200))

    def fake_collect(self, texts, indices, tokenizer, seq_len):
        results = []
        for idx in indices:
            if idx >= len(texts):
                continue
            token = (idx % 5) + 1
            input_ids = [token] * seq_len
            attention = [1] * seq_len
            results.append((idx, input_ids, attention, seq_len))
        return results

    monkeypatch.setattr(
        data_mod.WikiText2Provider,
        "_collect_tokenized_samples",
        fake_collect,
        raising=False,
    )
    provider = data_mod.WikiText2Provider()
    preview, final = provider.windows(
        tokenizer=_EncodeTokenizer(), preview_n=3, final_n=3, seq_len=6, seed=7
    )
    assert len(preview.input_ids) == 3
    assert len(final.input_ids) == 3
