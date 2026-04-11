from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def t(self) -> _FakeTensor:
        return self


class _FakeCuda:
    def __init__(self) -> None:
        self._available = False

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1 if self._available else 0

    def memory_stats(self, index: int) -> dict[str, int]:
        return {
            "allocated_bytes.all.current": 0,
            "reserved_bytes.all.current": 0,
        }

    def get_device_properties(self, index: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            total_memory=16 * 1024**3,
            name=f"Fake GPU {index}",
        )

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"


def _fake_randn(*shape: int) -> _FakeTensor:
    return _FakeTensor(tuple(shape))


def _fake_mm(left: _FakeTensor, right: _FakeTensor) -> _FakeTensor:
    return _FakeTensor(left.shape)


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


@pytest.fixture(autouse=True)
def _install_observability_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _new_module("torch")
    fake_torch.__version__ = "0.0-test"
    fake_torch.__path__ = []
    fake_torch.version = types.SimpleNamespace(cuda=None)
    fake_torch.cuda = _FakeCuda()
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.randn = _fake_randn
    fake_torch.mm = _fake_mm

    fake_torch_nn = _new_module("torch.nn")
    fake_torch_nn.Module = object

    fake_transformers = _new_module("transformers")
    fake_numpy = _new_module("numpy")

    fake_hf_causal = _new_module("invarlock.adapters.hf_causal")
    fake_hf_causal.HF_Causal_Adapter = type("HF_Causal_Adapter", (), {})

    fake_hf_mlm = _new_module("invarlock.adapters.hf_mlm")
    fake_hf_mlm.HF_MLM_Adapter = type("HF_MLM_Adapter", (), {})

    fake_hf_multimodal = _new_module("invarlock.adapters.hf_multimodal")
    fake_hf_multimodal.HF_Multimodal_Adapter = type("HF_Multimodal_Adapter", (), {})

    fake_hf_seq2seq = _new_module("invarlock.adapters.hf_seq2seq")
    fake_hf_seq2seq.HF_Seq2Seq_Adapter = type("HF_Seq2Seq_Adapter", (), {})

    fake_invariants = _new_module("invarlock.guards.invariants")
    fake_invariants.InvariantsGuard = type("InvariantsGuard", (), {})

    fake_rmt = _new_module("invarlock.guards.rmt")
    fake_rmt.RMTGuard = type("RMTGuard", (), {})

    fake_spectral = _new_module("invarlock.guards.spectral")
    fake_spectral.SpectralGuard = type("SpectralGuard", (), {})

    fake_variance = _new_module("invarlock.guards.variance")
    fake_variance.VarianceGuard = type(
        "VarianceGuard",
        (),
        {"__init__": lambda self, policy=None: None},
    )

    fake_policies = _new_module("invarlock.guards.policies")
    fake_policies.get_variance_policy = lambda name: {"name": name}

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch_nn)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "invarlock.adapters.hf_causal", fake_hf_causal)
    monkeypatch.setitem(sys.modules, "invarlock.adapters.hf_mlm", fake_hf_mlm)
    monkeypatch.setitem(
        sys.modules,
        "invarlock.adapters.hf_multimodal",
        fake_hf_multimodal,
    )
    monkeypatch.setitem(sys.modules, "invarlock.adapters.hf_seq2seq", fake_hf_seq2seq)
    monkeypatch.setitem(sys.modules, "invarlock.guards.invariants", fake_invariants)
    monkeypatch.setitem(sys.modules, "invarlock.guards.rmt", fake_rmt)
    monkeypatch.setitem(sys.modules, "invarlock.guards.spectral", fake_spectral)
    monkeypatch.setitem(sys.modules, "invarlock.guards.variance", fake_variance)
    monkeypatch.setitem(sys.modules, "invarlock.guards.policies", fake_policies)
