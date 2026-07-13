from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    DependencyError,
    DependencyManager,
    compute_perplexity_strict,
)


def test_compute_perplexity_strict_tuple_batch_variants():
    class TinyLM(nn.Module):
        def __init__(self, vocab_size=7):
            super().__init__()
            self.vocab = vocab_size

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=False
        ):
            B, T = input_ids.shape
            logits = torch.randn(B, T, self.vocab)
            if return_dict:
                return SimpleNamespace(logits=logits)
            return (logits,)

    model = TinyLM().eval()
    input_ids = torch.randint(0, model.vocab, (1, 5))
    labels = input_ids.clone()
    attn = torch.ones_like(input_ids)
    tok_types = torch.zeros_like(input_ids)
    # Provide tuple batch (input_ids, labels, attention_mask, token_type_ids)
    ppl = compute_perplexity_strict(
        model, [(input_ids, labels, attn, tok_types)], device="cpu"
    )
    assert isinstance(ppl, float) and ppl >= 1.0


def test_dependency_manager_get_module_missing_raises():
    dm = DependencyManager()
    with pytest.raises(DependencyError):
        _ = dm.get_module("totally_missing_module_name")
