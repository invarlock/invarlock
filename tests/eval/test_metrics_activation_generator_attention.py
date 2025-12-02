from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, _collect_activations


def test_collect_activations_generator_with_attention_mask_first_batch_capture():
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
            # Provide blocks for _extract_fc1_activations path
            self.transformer = SimpleNamespace(h=[Block(), Block()])

        def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
            B, T = input_ids.shape
            if output_hidden_states:
                hs = [torch.randn(B, T, 3) for _ in range(4)]
                return SimpleNamespace(hidden_states=hs)
            return SimpleNamespace(logits=torch.randn(B, T, 5))

    # Non-indexable generator yielding dict with attention_mask included
    def gen():
        yield {
            "input_ids": torch.ones(1, 6, dtype=torch.long),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }

    out = _collect_activations(
        Model().eval(),
        gen(),
        MetricsConfig(progress_bars=False, oracle_windows=1),
        torch.device("cpu"),
    )
    assert isinstance(out.get("first_batch"), dict)
    # Ensure attention_mask was preserved in first_batch payload
    assert "attention_mask" in out["first_batch"]
