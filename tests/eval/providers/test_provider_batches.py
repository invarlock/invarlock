from __future__ import annotations

import pytest

from invarlock.cli.run_pairing import _compute_mask_positions_digest
from invarlock.eval.data import Seq2SeqProvider, TextLMProvider


def _flatten_batches(batches):
    items = []
    for b in batches:
        for i in range(len(b["ids"])):
            items.append(
                {k: (v[i] if isinstance(v, list) else v) for k, v in b.items()}
            )
    return items


def test_text_lm_causal_batches_and_pairing():
    provider = TextLMProvider(task="causal", n=7, seq_len=6)
    batches = list(provider.batches(seed=123, batch_size=3))
    items = _flatten_batches(batches)
    assert len(items) == 7
    # Basic fields present
    for it in items:
        assert (
            "input_ids" in it
            and "attention_mask" in it
            and "weights" in it
            and "ids" in it
        )
        assert isinstance(it["weights"], int) and it["weights"] > 0
    # Pairing schedule stable and sorted
    schedule = provider.pairing_schedule()
    assert schedule == sorted(schedule)
    assert len(schedule) == 7


def test_text_lm_mlm_masks_and_digest_stability():
    provider = TextLMProvider(task="mlm", n=5, seq_len=8, mask_prob=0.2)
    batches1 = list(provider.batches(seed=7, batch_size=2))
    batches2 = list(provider.batches(seed=7, batch_size=3))
    items1 = _flatten_batches(batches1)
    items2 = _flatten_batches(batches2)
    # Build window-like dict to compute mask digest
    win1 = {
        "preview": {
            "labels": [it.get("labels", []) for it in items1[:3]],
            "window_ids": [it["ids"] for it in items1[:3]],
        },
        "final": {
            "labels": [it.get("labels", []) for it in items1[3:]],
            "window_ids": [it["ids"] for it in items1[3:]],
        },
    }
    win2 = {
        "preview": {
            "labels": [it.get("labels", []) for it in items2[:3]],
            "window_ids": [it["ids"] for it in items2[:3]],
        },
        "final": {
            "labels": [it.get("labels", []) for it in items2[3:]],
            "window_ids": [it["ids"] for it in items2[3:]],
        },
    }
    d1 = _compute_mask_positions_digest(win1)
    d2 = _compute_mask_positions_digest(win2)
    assert isinstance(d1, str) and d1
    assert d1 == d2  # same seed → same mask positions


def test_text_lm_mlm_masks_present_even_with_zero_prob():
    provider = TextLMProvider(task="mlm", n=1, seq_len=6, mask_prob=0.0)
    batch = next(iter(provider.batches(seed=5, batch_size=1)))
    labels = batch["labels"][0]
    assert any(val != -100 for val in labels)


def test_text_lm_provider_mlm_masks_present_extra():
    provider = TextLMProvider(task="mlm", n=5, seq_len=6, mask_prob=0.5)
    for batch in provider.batches(seed=7, batch_size=10):
        for labels, weight in zip(batch["labels"], batch["weights"], strict=False):
            assert weight > 0
            assert any(int(token) != -100 for token in labels)
    schedule = provider.pairing_schedule()
    assert schedule == sorted(schedule)


def test_seq2seq_provider_weights_match_target_tokens():
    provider = Seq2SeqProvider(n=6, src_len=5, tgt_len=7)
    batches = list(provider.batches(seed=42, batch_size=4))
    items = _flatten_batches(batches)
    assert len(items) == 6
    for it in items:
        tgt_ids = it["tgt_ids"]
        tgt_mask = it["tgt_mask"]
        expected = sum(1 for t, m in zip(tgt_ids, tgt_mask, strict=False) if m)
        assert it["weights"] == expected


def test_seq2seq_provider_default_schedule_and_digest():
    provider = Seq2SeqProvider(
        n=3,
        src_len=5,
        tgt_len=7,
        pad_id=9,
        bos_id=11,
        eos_id=13,
    )
    assert provider.pairing_schedule() == ["ex0000", "ex0001", "ex0002"]
    assert provider.digest() == {
        "provider": "seq2seq",
        "version": 1,
        "pad_id": 9,
        "eos_id": 13,
        "bos_id": 11,
    }


def test_seq2seq_windows_resizes_and_falls_back_when_mask_lengths_mismatch():
    class _Tok:
        pad_token_id = 0

    provider = Seq2SeqProvider(n=1, src_len=4, tgt_len=4, pad_id=0)

    def fake_batches(*, seed, batch_size):  # noqa: ARG001
        return [
            {
                "ids": ["ex0000", "ex0001"],
                "src_ids": [[9, 8, 7, 6], [4, 3, 2, 1]],
                "src_mask": [[1, 1], [1, 1, 1, 1]],
                "tgt_ids": [[4, 5, 6, 2], [7, 8, 9, 2]],
                "tgt_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
                "weights": [4, 4],
            }
        ]

    provider.batches = fake_batches
    prev, fin = provider.windows(_Tok(), seq_len=3, preview_n=1, final_n=1)
    assert provider._n == 2
    assert len(prev.input_ids[0]) == 3
    assert prev.input_ids[0] == [9, 8, 7]
    assert prev.attention_masks[0] == [1, 1, 1]
    assert len(fin.input_ids) == 1


def test_seq2seq_windows_rejects_empty_batches(monkeypatch):
    class _Tok:
        pad_token_id = 0

    provider = Seq2SeqProvider(n=1)
    monkeypatch.setattr(provider, "batches", lambda **kwargs: [])
    with pytest.raises(ValueError, match="seq2seq provider produced no examples"):
        provider.windows(_Tok(), seq_len=4, preview_n=1, final_n=1)
