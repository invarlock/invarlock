from types import SimpleNamespace

from invarlock.cli import run_runtime


def test_optional_module_helpers_preserve_explicit_bindings(monkeypatch):
    fake_psutil = SimpleNamespace(virtual_memory=lambda: "vm")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    monkeypatch.setattr(run_runtime, "psutil", fake_psutil)
    monkeypatch.setattr(run_runtime, "torch", fake_torch)

    run_runtime.reset_optional_runtime_caches()

    assert run_runtime.get_psutil() is fake_psutil
    assert run_runtime.get_torch() is fake_torch


def test_free_model_memory_invokes_cuda(monkeypatch):
    calls = {"empty_cache": 0, "synchronize": 0}

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            calls["empty_cache"] += 1

        def synchronize(self) -> None:
            calls["synchronize"] += 1

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    monkeypatch.setattr(run_runtime, "torch", fake_torch)

    run_runtime.free_model_memory(object())

    assert calls["empty_cache"] == 1
    assert calls["synchronize"] == 1


def test_free_model_memory_tolerates_missing_torch(monkeypatch):
    monkeypatch.setattr(run_runtime, "torch", None)
    # Should not raise when torch is unavailable
    run_runtime.free_model_memory(object())


def test_free_model_memory_swallows_cuda_exceptions(monkeypatch):
    class FakeCuda:
        def is_available(self) -> bool:
            raise RuntimeError("boom")

    monkeypatch.setattr(run_runtime, "torch", SimpleNamespace(cuda=FakeCuda()))
    run_runtime.free_model_memory(object())
