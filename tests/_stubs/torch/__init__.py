from __future__ import annotations

# Minimal torch stub for tests that do not actually use torch.

__version__ = "0.0.0-stub"


class device(str):
    pass


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:  # no-op
        return None

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def memory_allocated() -> int:
        return 0

    @staticmethod
    def memory_reserved() -> int:
        return 0


class _MPS:
    @staticmethod
    def is_available() -> bool:
        return False


class _Backends:
    cudnn = type(
        "_CUDNN", (), {"allow_tf32": False, "version": staticmethod(lambda: 0)}
    )()
    cuda = type("_CUDA", (), {"matmul": type("_MATMUL", (), {"allow_tf32": False})()})()
    mps = _MPS()


def randint(low: int, high: int, size, device: device | None = None):  # type: ignore[no-untyped-def]
    # Return a simple Python list placeholder; tests that rely on real tensors skip when needed.
    import random

    n, m = size
    return [[random.randint(low, high - 1) for _ in range(m)] for _ in range(n)]


def manual_seed(seed: int) -> None:
    return None


def tensor(values, device: device | None = None):  # type: ignore[no-untyped-def]
    return values


cuda = _Cuda()
backends = _Backends()


class Tensor:  # typing stub
    pass


# Expose subpackages as attributes for type hints and imports
try:  # pragma: no cover - simple import wiring for stub
    import importlib as _importlib

    nn = _importlib.import_module(".nn", __name__)
    utils = _importlib.import_module(".utils", __name__)
    linalg = _importlib.import_module(".linalg", __name__)
except Exception:  # defensive: leave attributes absent if import fails
    pass


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return False

    def __call__(self, fn=None):  # type: ignore[no-untyped-def]
        if fn is None:

            def _decorator(f):
                return f

            return _decorator
        return fn


no_grad = _NoGrad()


def isfinite(x):  # type: ignore[no-untyped-def]
    return True
