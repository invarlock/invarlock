from __future__ import annotations

from invarlock import utils


def test_get_memory_usage_handles_cuda_probe_failure(monkeypatch) -> None:
    class _BrokenCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def memory_allocated() -> int:
            raise RuntimeError("probe failed")

        @staticmethod
        def memory_reserved() -> int:
            raise RuntimeError("probe failed")

    class _FakeTorch:
        cuda = _BrokenCuda()

    monkeypatch.setattr(utils, "_get_torch", lambda: _FakeTorch())

    memory = utils.get_memory_usage()

    assert "rss_mb" in memory
    assert "cuda_allocated_mb" not in memory
