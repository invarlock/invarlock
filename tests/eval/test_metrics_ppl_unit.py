from types import SimpleNamespace

import torch

from invarlock.eval.metrics import compute_perplexity, compute_ppl


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.vocab = vocab_size
        self.lin = torch.nn.Linear(8, vocab_size)

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=False
    ):
        # Produce simple logits of shape [B, T, V]
        B, T = input_ids.shape
        # Map tokens to one-hot-like embeddings just for shape
        embed = torch.nn.functional.one_hot(
            input_ids.clamp_min(0) % self.vocab, num_classes=self.vocab
        ).float()
        logits = embed + 0.01  # make non-zero
        if return_dict:
            return SimpleNamespace(logits=logits)
        return (logits,)


def test_compute_perplexity_simple():
    model = TinyLM().eval()
    batch = {
        "input_ids": torch.randint(0, model.vocab, (1, 4)),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    ppl = compute_perplexity(model, [batch], max_samples=1)
    assert isinstance(ppl, float) and ppl >= 1.0


def test_compute_ppl_window_simple():
    model = TinyLM().eval()
    window = SimpleNamespace(
        input_ids=[[1, 2, 3, 4]],
        attention_masks=[[1, 1, 1, 1]],
    )
    ppl = compute_ppl(model, None, window)
    assert isinstance(ppl, float) and ppl >= 1.0
