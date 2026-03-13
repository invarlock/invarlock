from types import SimpleNamespace

from invarlock.cli.commands import run


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
    monkeypatch.setattr(run, "torch", fake_torch)

    run._free_model_memory(object())

    assert calls["empty_cache"] == 1
    assert calls["synchronize"] == 1


def test_free_model_memory_tolerates_missing_torch(monkeypatch):
    monkeypatch.setattr(run, "torch", None)
    # Should not raise when torch is unavailable
    run._free_model_memory(object())


def test_free_model_memory_swallows_cuda_exceptions(monkeypatch):
    class FakeCuda:
        def is_available(self) -> bool:
            raise RuntimeError("boom")

    monkeypatch.setattr(run, "torch", SimpleNamespace(cuda=FakeCuda()))
    run._free_model_memory(object())
