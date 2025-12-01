from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, _collect_activations


def test_collect_activations_indexable_and_hidden_states_le_two():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()

            class CF(nn.Module):
                def forward(self, x):
                    B, T, _ = x.shape
                    return torch.randn(B, T, 4)

            self.mlp = SimpleNamespace(c_fc=CF())

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[Block(), Block()])

        def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
            B, T = input_ids.shape
            if output_hidden_states:
                # Exactly two hidden states → branch len>2 is False
                hs = [torch.randn(B, T, 3) for _ in range(2)]
                return SimpleNamespace(hidden_states=hs)
            return SimpleNamespace(logits=torch.randn(B, T, 5))

    # Indexable dataloader (list) with length > oracle_windows to flip len() branch
    dl = [
        {"input_ids": torch.ones(1, 6, dtype=torch.long)},
        {"input_ids": torch.ones(1, 6, dtype=torch.long)},
        {"input_ids": torch.ones(1, 6, dtype=torch.long)},
    ]
    out = _collect_activations(
        Model().eval(),
        dl,
        MetricsConfig(progress_bars=False, oracle_windows=2),
        torch.device("cpu"),
    )
    assert out["first_batch"] is not None
    # No hidden states collected (len <= 2), but fc1 activations path should still run or be empty without error
    assert isinstance(out["hidden_states"], list)
