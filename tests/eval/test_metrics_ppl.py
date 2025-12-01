import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    ValidationError as MValidationError,
)
from invarlock.eval.metrics import (
    compute_perplexity,
    compute_perplexity_strict,
)


class DummyCausalModel(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)
        self.config = SimpleNamespace(model_type="gpt2")

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):  # noqa: D401
        x = self.embed(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits) if return_dict else (logits,)


class DummyBertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="bert")
        self.dummy = nn.Linear(1, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=True,
    ):
        # Return a constant scalar loss to hit masked LM branch
        return SimpleNamespace(loss=torch.tensor(1.0, dtype=torch.float32))


def _make_dataloader(batch_count=3, seq_len=6, with_mask=True):
    data = []
    for _ in range(batch_count):
        ids = torch.randint(0, 16, (2, seq_len))
        attn = torch.ones_like(ids)
        if with_mask:
            attn[:, -1] = 0  # pad last token to exercise masking
        data.append({"input_ids": ids, "attention_mask": attn})
    return data


def test_compute_perplexity_strict_causal_success():
    model = DummyCausalModel()
    dl = _make_dataloader(batch_count=2, seq_len=5)
    ppl = compute_perplexity_strict(model, dl)
    assert isinstance(ppl, float) and math.isfinite(ppl) and ppl >= 1.0


def test_compute_perplexity_strict_masked_lm_branch():
    model = DummyBertModel()
    dl = _make_dataloader(batch_count=2, seq_len=4, with_mask=False)
    ppl = compute_perplexity_strict(model, dl)
    assert isinstance(ppl, float) and math.isfinite(ppl) and ppl >= 1.0


def test_compute_perplexity_strict_raises_on_no_valid_tokens():
    model = DummyCausalModel()
    # seq_len=1 produces too-short sequences
    dl = _make_dataloader(batch_count=2, seq_len=1)
    with pytest.raises(MValidationError):
        compute_perplexity_strict(model, dl)


def test_compute_perplexity_success_with_masking():
    model = DummyCausalModel()
    dl = _make_dataloader(batch_count=3, seq_len=7)
    ppl = compute_perplexity(model, dl, max_samples=2)
    assert isinstance(ppl, float) and math.isfinite(ppl) and ppl >= 1.0


def test_compute_perplexity_raises_on_all_masked():
    model = DummyCausalModel()
    # Build data where valid becomes zero: set attention all zeros after shift
    dl = []
    for _ in range(2):
        ids = torch.randint(0, 16, (2, 4))
        attn = torch.zeros_like(ids)
        dl.append({"input_ids": ids, "attention_mask": attn})
    with pytest.raises(MValidationError):
        compute_perplexity(model, dl)
