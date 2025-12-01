import torch
import torch.nn as nn

from invarlock.eval.metrics import compute_perplexity_strict


def test_compute_perplexity_strict_masked_lm_path():
    class BertLike(nn.Module):
        def __init__(self, vocab=10):
            super().__init__()

            class Cfg:
                model_type = "bert"

            self.config = Cfg()
            self.vocab = vocab

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=False,
        ):
            # Return an object with a loss (mean over valid tokens)
            # Simulate loss ~ 1.0
            class Out:
                def __init__(self):
                    self.loss = torch.tensor(1.0)

            return Out()

    model = BertLike().eval()
    ids = torch.randint(0, model.vocab, (1, 6))
    attn = torch.ones_like(ids)
    # A small masked region ensures valid_tokens > 0
    dl = [{"input_ids": ids, "attention_mask": attn}]
    ppl = compute_perplexity_strict(model, dl, device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0
