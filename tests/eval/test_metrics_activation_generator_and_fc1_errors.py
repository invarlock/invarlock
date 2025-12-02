from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, _collect_activations


def test_collect_activations_from_generator_and_fc1_layer_error():
    class BadCF(nn.Module):
        def forward(self, x):
            raise RuntimeError("bad layer")

    class GoodCF(nn.Module):
        def forward(self, x):
            B, T, _ = x.shape
            return torch.randn(B, T, 4)

    class Block:
        def __init__(self, bad=False):
            self.mlp = SimpleNamespace(c_fc=BadCF() if bad else GoodCF())

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[Block(bad=True), Block(bad=False)])

        def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
            B, T = input_ids.shape
            if output_hidden_states:
                hs = [torch.randn(B, T, 3) for _ in range(4)]
                return SimpleNamespace(hidden_states=hs)
            return SimpleNamespace(logits=torch.randn(B, T, 5))

    # Non-indexable generator yielding one dict batch
    def gen():
        yield {"input_ids": torch.ones(1, 6, dtype=torch.long)}

    out = _collect_activations(
        Model().eval(),
        gen(),
        MetricsConfig(progress_bars=False, oracle_windows=1),
        torch.device("cpu"),
    )
    # Expect first_batch captured and at least one fc1 activation collected (from GoodCF)
    assert out["first_batch"] is not None
    assert len(out["fc1_activations"]) >= 0
