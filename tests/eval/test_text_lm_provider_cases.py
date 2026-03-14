from __future__ import annotations

from invarlock.eval.providers.text_lm import TextLMProvider


def test_text_lm_provider_mlm_masks_present_extra():
    p = TextLMProvider(task="mlm", n=5, seq_len=6, mask_prob=0.5)
    for batch in p.batches(seed=7, batch_size=10):
        for labels, w in zip(batch["labels"], batch["weights"], strict=False):
            assert w > 0
            assert any(int(x) != -100 for x in labels)
    sched = p.pairing_schedule()
    assert sched == sorted(sched)
