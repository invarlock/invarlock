from types import SimpleNamespace

import torch

from invarlock.eval import metrics as M


def test_compute_ppl_window_tuple_output_fallback():
    # Window with some valid tokens
    window = SimpleNamespace(input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]])

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
            return (logits,)

    ppl = M.compute_ppl(DummyLM(), None, window)
    assert isinstance(ppl, float) and ppl >= 1.0
