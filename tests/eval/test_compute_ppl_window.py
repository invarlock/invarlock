import math
from types import SimpleNamespace

import pytest
import torch.nn as nn

from invarlock.eval.metrics import ValidationError as MValidationError
from invarlock.eval.metrics import compute_ppl


class TinyCausal(nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):  # noqa: D401
        x = self.embed(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits) if return_dict else (logits,)


def test_compute_ppl_window_success():
    model = TinyCausal()
    # Build a small evaluation window with 2 samples
    window = SimpleNamespace(
        input_ids=[[1, 2, 3, 4], [4, 3, 2, 1]],
        attention_masks=[[1, 1, 1, 1], [1, 1, 1, 1]],
    )
    ppl = compute_ppl(model, adapter=None, window=window, device="cpu")
    assert isinstance(ppl, float) and math.isfinite(ppl) and ppl >= 1.0


def test_compute_ppl_window_degenerate_raises():
    model = TinyCausal()
    window = SimpleNamespace(input_ids=[], attention_masks=[])
    with pytest.raises(MValidationError):
        compute_ppl(model, adapter=None, window=window, device="cpu")
