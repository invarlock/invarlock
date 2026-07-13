from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig
from invarlock.eval.metrics_activation import _collect_activations


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
        MetricsConfig(oracle_windows=1),
        torch.device("cpu"),
    )
    assert set(out) == {"hidden_states", "fc1_activations", "targets"}
    assert len(out["fc1_activations"]) == 1
    assert tuple(out["fc1_activations"][0].shape) == (1, 1, 6, 4)
    assert len(out["hidden_states"]) == 1
    assert tuple(out["hidden_states"][0].shape) == (2, 1, 6, 3)
    assert len(out["targets"]) == 1
    assert tuple(out["targets"][0].shape) == (1, 5)
