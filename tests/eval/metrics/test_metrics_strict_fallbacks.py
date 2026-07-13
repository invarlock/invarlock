import torch

from invarlock.eval import metrics as M


def test_compute_perplexity_list_output_fallback():
    class DummyLM:
        def __init__(self, vocab=8):
            self.out = torch.nn.Linear(4, vocab)

        def parameters(self):
            yield from self.out.parameters()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return [logits]

    batch = {
        "input_ids": torch.ones(1, 6, dtype=torch.long),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    ppl = M.compute_perplexity(DummyLM(), [batch])
    assert isinstance(ppl, float) and ppl >= 1.0
