import types

import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    ResourceManager,
    _collect_activations,
    _extract_fc1_activations,
)


def test_resource_manager_mps_branch(monkeypatch):
    # Force CUDA not available and MPS available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

    class MPSBackends:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch.backends, "mps", MPSBackends(), raising=False)
    rm = ResourceManager(MetricsConfig())
    assert getattr(rm.device, "type", str(rm.device)).startswith("mps")


def test_get_memory_info_cuda_branch(monkeypatch):
    cfg = MetricsConfig()
    rm = ResourceManager(cfg)
    rm.device = torch.device("cuda")

    class Props:
        total_memory = 1024**3

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: Props(), raising=False
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 128, raising=False)
    info = rm._get_memory_info()
    assert info.get("gpu_total_gb") and info.get("gpu_free_gb")


def test_collect_activations_generator_dataloader_and_extract_fc1_exception(
    monkeypatch,
):
    # Model that returns hidden states
    class Block(nn.Module):
        def __init__(self):
            super().__init__()

            # c_fc will raise to exercise per-layer try/except
            class Bad(nn.Module):
                def forward(self, x):
                    raise RuntimeError("boom")

            self.mlp = types.SimpleNamespace(c_fc=Bad())

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = types.SimpleNamespace(h=[Block()])

        def forward(self, input_ids, output_hidden_states=False):
            B, T = input_ids.shape
            hs = [torch.randn(B, T, 4) for _ in range(3)]  # >=2 to pass check
            return types.SimpleNamespace(hidden_states=hs)

    # Generator dataloader (no __len__) forces oracle_windows path
    def gen():
        for _ in range(2):
            yield {"input_ids": torch.ones(1, 8, dtype=torch.long)}

    cfg = MetricsConfig(oracle_windows=2, max_tokens=4, progress_bars=False)
    device = torch.device("cpu")
    data = _collect_activations(Model(), gen(), cfg, device)
    assert isinstance(data, dict) and data.get("first_batch") is not None

    # _extract_fc1_activations should catch layer exception and return None
    out = _extract_fc1_activations(
        Model(),
        types.SimpleNamespace(hidden_states=[torch.randn(1, 8, 4) for _ in range(4)]),
        cfg,
    )
    assert out is None
