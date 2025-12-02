import pytest
import torch

from invarlock.eval.metrics import compute_perplexity

pytest.importorskip("transformers")


def test_compute_perplexity_masked_lm_returns_positive_value():
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
    )
    model = BertForMaskedLM(config)

    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    labels = input_ids.clone()

    dataloader = [
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
    ]

    ppl = compute_perplexity(model, dataloader, device="cpu")
    assert ppl >= 1.0


def test_compute_perplexity_masked_lm_masks_tokens():
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
    )
    model = BertForMaskedLM(config)

    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    labels = input_ids.clone()
    labels[:, ::2] = -100  # mask every other token

    dataloader = [
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
    ]

    masked_tokens = int((labels[:, 1:] != -100).sum().item())
    assert masked_tokens > 0

    ppl = compute_perplexity(model, dataloader, device="cpu")
    assert ppl >= 1.0
