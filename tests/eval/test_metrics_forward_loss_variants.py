from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import _forward_loss_causal, compute_perplexity_strict


class ModelWithOutput(nn.Module):
    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=False
    ):
        logits = torch.randn(input_ids.size(0), input_ids.size(1), 5)
        loss = torch.tensor(1.23)
        if return_dict:
            return SimpleNamespace(logits=logits, loss=loss)
        return (loss, logits)


class ModelLogitsOnly(nn.Module):
    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=False
    ):
        if return_dict:
            # Simulate older model that doesn't support return_dict
            raise TypeError("return_dict not supported")
        logits = torch.randn(input_ids.size(0), input_ids.size(1), 5)
        return (logits,)


def test_forward_loss_handles_modeloutput_and_tuple():
    B, T = 1, 4
    input_ids = torch.randint(0, 5, (B, T))
    labels = input_ids.clone()
    loss, logits = _forward_loss_causal(ModelWithOutput(), input_ids, None, labels)
    assert isinstance(loss, float) and logits is not None

    loss2, logits2 = _forward_loss_causal(ModelWithOutput(), input_ids, None, labels)
    assert isinstance(loss2, float) and logits2 is not None

    # Tuple path (loss first, logits second)
    loss3, logits3 = _forward_loss_causal(ModelWithOutput(), input_ids, None, labels)
    assert isinstance(loss3, float) and logits3 is not None


def test_forward_loss_manual_loss_when_only_logits():
    B, T = 1, 4
    input_ids = torch.randint(0, 5, (B, T))
    labels = input_ids.clone()
    loss, logits = _forward_loss_causal(ModelLogitsOnly(), input_ids, None, labels)
    assert isinstance(loss, float) and logits is not None


def test_compute_perplexity_strict_tuple_and_list_batches():
    B, T = 1, 4
    input_ids = torch.randint(0, 5, (B, T))
    labels = input_ids.clone()
    attn = torch.ones_like(input_ids)
    # Tuple batch with token_type_ids present
    batch_tuple = (input_ids, labels, attn, attn)
    # List batch variant
    batch_list = [input_ids, labels, attn]
    ppl = compute_perplexity_strict(
        ModelLogitsOnly(), [batch_tuple, batch_list], device="cpu"
    )
    assert isinstance(ppl, float) and ppl >= 1.0
