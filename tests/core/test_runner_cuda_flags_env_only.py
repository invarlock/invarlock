import builtins
import os

from invarlock.core.runner import _collect_cuda_flags


def test_collect_cuda_flags_env_only(monkeypatch):
    # Force ImportError when importing torch inside helper to take env-only path
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    flags = _collect_cuda_flags()

    # Only the environment-provided flag should be present when torch is absent
    assert flags == {"CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
