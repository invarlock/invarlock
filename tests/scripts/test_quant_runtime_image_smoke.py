from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "_runtime_images"
    / "quant_runtime_image_smoke.py"
)


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("quant_runtime_image_smoke", SMOKE)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cuda_toolchain_check_delegates_to_named_gptqmodel_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load_smoke_module()
    calls: list[tuple[tuple[str, ...], bool]] = []
    monkeypatch.setattr(
        smoke,
        "_require_gptqmodel_runtime",
        lambda selected_adapters, *, require_jit_toolchain: calls.append(
            (selected_adapters, require_jit_toolchain)
        ),
    )

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_utils = types.ModuleType("torch.utils")
    torch_cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    torch_cpp_extension.CUDA_HOME = "/cuda"
    torch_utils.cpp_extension = torch_cpp_extension
    torch.utils = torch_utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", torch_cpp_extension)

    selected_adapters = ("hf_gptq",)
    smoke._check_cuda_runtime(
        selected_adapters=selected_adapters,
        require_cuda_toolchain=True,
        require_gpu=False,
    )

    assert calls == [(selected_adapters, True)]


def test_cuda_toolchain_check_preserves_actionable_python_header_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load_smoke_module()

    def _missing_headers(*, require_jit_toolchain: bool) -> None:
        assert require_jit_toolchain is True
        raise RuntimeError("GPTQModel CUDA JIT toolchain unavailable: Python.h missing")

    from invarlock import gptqmodel_runtime

    monkeypatch.setattr(
        gptqmodel_runtime, "require_gptqmodel_runtime", _missing_headers
    )

    with pytest.raises(SystemExit, match="Python\\.h missing"):
        smoke._require_gptqmodel_runtime(
            ("hf_gptq",),
            require_jit_toolchain=True,
        )
