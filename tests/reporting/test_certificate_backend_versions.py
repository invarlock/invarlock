from __future__ import annotations

import builtins
import os
from types import SimpleNamespace

from invarlock.reporting import certificate as C


def _make_fake_torch(
    *,
    cuda_available: bool = False,
    cudnn_version: int | None = 8900,
    nccl_version: str | None = "2.18.1",
    allow_tf32: bool = True,
):
    class _Props:
        def __init__(self) -> None:
            self.name = "Fake GPU"
            self.major = 8
            self.minor = 0

    fake = SimpleNamespace()
    fake.__version__ = "2.3.0"
    fake.version = SimpleNamespace(cuda="12.1", cudnn="8.9.0", git_version="deadbeef")
    fake.cuda = SimpleNamespace()
    fake.cuda.is_available = lambda: cuda_available
    fake.cuda.get_device_properties = lambda idx: _Props()
    fake.cuda.nccl = SimpleNamespace(version=lambda: nccl_version)  # type: ignore[arg-type]
    fake.backends = SimpleNamespace(
        cudnn=SimpleNamespace(version=lambda: cudnn_version, allow_tf32=allow_tf32),
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=allow_tf32)),
    )
    return fake


def test_collect_backend_versions_with_fake_torch(monkeypatch):
    # Ensure environment hint path is exercised
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Inject a fake torch module to exercise the "torch available" branches
    fake_torch = _make_fake_torch(cuda_available=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    info = C._collect_backend_versions()
    # Python/platform keys are always present when platform module works
    assert isinstance(info.get("python"), str)
    # Torch-derived keys should be populated from the fake module
    assert info.get("torch") == "2.3.0"
    assert info.get("torch_cuda") == "12.1"
    assert info.get("torch_cudnn") == "8.9.0"
    assert info.get("torch_git") == "deadbeef"
    assert info.get("device_name") == "Fake GPU"
    assert info.get("sm_capability") == "8.0"
    assert isinstance(info.get("cudnn_runtime"), int)
    assert info.get("nccl") == "2.18.1"
    assert isinstance(info.get("tf32"), dict)
    # Environment variable surfaced
    assert info.get("cublas_workspace_config") == os.environ["CUBLAS_WORKSPACE_CONFIG"]


def test_collect_backend_versions_without_torch(monkeypatch):
    # Force import of torch to fail inside the function by intercepting __import__
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    info = C._collect_backend_versions()
    # Should still return Python/platform basics, but no torch keys
    assert isinstance(info.get("python"), str)
    assert "torch" not in info
