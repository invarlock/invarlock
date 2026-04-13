from __future__ import annotations

import builtins
import os
import sys
from types import SimpleNamespace

from invarlock.reporting import report_provenance as provenance_mod


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
    fake.cuda.nccl = SimpleNamespace(version=lambda: nccl_version)
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

    info = provenance_mod.collect_backend_versions()
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

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    info = provenance_mod.collect_backend_versions()
    # Should still return Python/platform basics, but no torch keys
    assert isinstance(info.get("python"), str)
    assert "torch" not in info


def test_collect_backend_versions_tolerates_platform_and_env_failures(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    class RaisingEnv:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    def boom() -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(provenance_mod.platform, "python_version", boom)
    monkeypatch.setattr(provenance_mod, "os", SimpleNamespace(environ=RaisingEnv()))

    info = provenance_mod.collect_backend_versions()
    assert "python" not in info
    assert "cublas_workspace_config" not in info


def test_collect_backend_versions_handles_partial_torch_metadata(monkeypatch):
    class Props:
        name = "Partial GPU"
        major = 8
        minor = None

    fake_torch = SimpleNamespace()
    fake_torch.__version__ = "2.4.0"
    fake_torch.version = None
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _idx: Props(),
    )
    fake_torch.backends = SimpleNamespace(
        cudnn=SimpleNamespace(version=lambda: None, allow_tf32=False),
        cuda=SimpleNamespace(matmul=SimpleNamespace()),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = provenance_mod.collect_backend_versions()
    assert info["torch"] == "2.4.0"
    assert info["device_name"] == "Partial GPU"
    assert "torch_cuda" not in info
    assert "sm_capability" not in info
    assert "cudnn_runtime" not in info
    assert "nccl" not in info
    assert info["tf32"] == {"cudnn_allow_tf32": False}


def test_collect_backend_versions_tolerates_runtime_errors(monkeypatch):
    class BrokenNccl:
        def version(self):
            raise RuntimeError("nccl boom")

    class BrokenCudnn:
        allow_tf32 = True

        def version(self):
            raise RuntimeError("cudnn boom")

    class BrokenMatmul:
        @property
        def allow_tf32(self):
            raise RuntimeError("tf32 boom")

    class BrokenCuda:
        nccl = BrokenNccl()

        def is_available(self):
            raise RuntimeError("cuda boom")

    fake_torch = SimpleNamespace()
    fake_torch.__version__ = "2.5.0"
    fake_torch.version = SimpleNamespace(
        cuda="12.4",
        cudnn="9.0.0",
        git_version="cafebabe",
    )
    fake_torch.cuda = BrokenCuda()
    fake_torch.backends = SimpleNamespace(
        cudnn=BrokenCudnn(),
        cuda=SimpleNamespace(matmul=BrokenMatmul()),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = provenance_mod.collect_backend_versions()
    assert info["torch"] == "2.5.0"
    assert info["torch_cuda"] == "12.4"
    assert info["torch_cudnn"] == "9.0.0"
    assert info["torch_git"] == "cafebabe"
    assert "device_name" not in info
    assert "cudnn_runtime" not in info
    assert "nccl" not in info
    assert "tf32" not in info


def test_collect_backend_versions_skips_missing_optional_backends(monkeypatch):
    fake_torch = SimpleNamespace()
    fake_torch.__version__ = "2.6.0"
    fake_torch.version = SimpleNamespace(
        cuda="12.5",
        cudnn="9.1.0",
        git_version="feedface",
    )
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = provenance_mod.collect_backend_versions()
    assert info["torch"] == "2.6.0"
    assert info["torch_cuda"] == "12.5"
    assert info["torch_cudnn"] == "9.1.0"
    assert info["torch_git"] == "feedface"
    assert "cudnn_runtime" not in info
    assert "nccl" not in info
    assert "tf32" not in info
