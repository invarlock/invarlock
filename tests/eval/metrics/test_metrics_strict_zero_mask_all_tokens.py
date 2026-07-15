import torch
import torch.nn as nn

from invarlock.eval.metrics import compute_perplexity_strict


def test_compute_perplexity_strict_all_tokens_masked_raises():
    class TinyBert(nn.Module):
        def __init__(self, vocab=5):
            super().__init__()
            self.config = type("cfg", (), {"model_type": "bert"})()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=True,
        ):
            # Return a valid scalar loss, but with all tokens masked, valid_tokens will be zero
            return type(
                "Out",
                (),
                {
                    "loss": torch.tensor(0.5),
                    "logits": torch.zeros(input_ids.size(0), input_ids.size(1), 5),
                },
            )()

    ids = torch.randint(0, 5, (1, 6))
    attn = torch.zeros(1, 6, dtype=torch.long)  # masks everything → zero valid tokens
    batch = {"input_ids": ids, "attention_mask": attn}
    import pytest

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        compute_perplexity_strict(TinyBert(), [batch], device="cpu")
