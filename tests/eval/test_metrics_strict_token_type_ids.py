import torch
import torch.nn as nn

from invarlock.eval.metrics import compute_perplexity_strict


def test_compute_perplexity_strict_list_batch_with_token_type_ids():
    class TinyLM(nn.Module):
        def __init__(self, vocab=6):
            super().__init__()
            self.lin = nn.Linear(4, vocab)

        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.lin(torch.zeros(B, T, 4))
            return type("Out", (), {"logits": logits})()

    ids = torch.randint(0, 6, (1, 5))
    labels = ids.clone()
    attn = torch.ones_like(ids)
    ttype = torch.zeros_like(ids)
    batch_list = [ids, labels, attn, ttype]
    ppl = compute_perplexity_strict(TinyLM().eval(), [batch_list], device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0
