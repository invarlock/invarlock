from __future__ import annotations

from invarlock.cli.commands.run import _compute_mask_positions_digest
from invarlock.eval.providers.seq2seq import Seq2SeqProvider
from invarlock.eval.providers.text_lm import TextLMProvider


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
