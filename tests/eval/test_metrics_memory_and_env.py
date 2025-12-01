from types import SimpleNamespace

import torch

from invarlock.eval import metrics as M


def test_measure_memory_cpu_path_returns_float():
    # Build a tiny window with a few samples to exercise the CPU branch
    window = SimpleNamespace(
        input_ids=[[1, 2, 3], [4, 5, 6, 7], [8]],
        attention_masks=[[1, 1, 1], [1, 1, 1, 1], [1]],
    )

    class DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, input_ids=None, attention_mask=None):  # noqa: D401
            # Return a trivial tensor; memory measurement should still proceed
            bsz, seqlen = input_ids.shape
            return torch.zeros(bsz, seqlen, 8)

    model = DummyLM()
    mem = M.measure_memory(model, window, device="cpu")
    assert isinstance(mem, float)
    assert mem >= 0.0


def test_metrics_env_info_helpers():
    info = M.get_metrics_info()
    assert {"available_metrics", "default_config"}.issubset(info.keys())
    ok = M.validate_metrics_environment()
    assert isinstance(ok, bool)
