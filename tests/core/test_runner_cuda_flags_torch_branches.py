import types

from invarlock.core.runner import _collect_cuda_flags


def test_collect_cuda_flags_minimal_torch(monkeypatch):
    # Torch stub with no cudnn/cuda backends
    TorchStub = types.SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: False,
        backends=types.SimpleNamespace(),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", TorchStub)

    flags = _collect_cuda_flags()
    assert flags.get("deterministic_algorithms") is False
    # No backends keys expected
    assert "cudnn_deterministic" not in flags
    assert "cuda_matmul_allow_tf32" not in flags


def test_collect_cuda_flags_with_cudnn_and_matmul_tf32(monkeypatch):
    # Torch stub with cudnn and cuda.matmul allow_tf32
    cudnn = types.SimpleNamespace(deterministic=True, benchmark=False, allow_tf32=True)
    matmul = types.SimpleNamespace(allow_tf32=True)
    cuda = types.SimpleNamespace(matmul=matmul)
    backends = types.SimpleNamespace(cudnn=cudnn, cuda=cuda)
    TorchStub = types.SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: True,
        backends=backends,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", TorchStub)

    flags = _collect_cuda_flags()
    assert flags["deterministic_algorithms"] is True
    assert flags["cudnn_deterministic"] is True
    assert flags["cudnn_benchmark"] is False
    assert flags["cudnn_allow_tf32"] is True
    assert flags["cuda_matmul_allow_tf32"] is True
