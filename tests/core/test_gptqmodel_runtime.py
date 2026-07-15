from __future__ import annotations

import os
import stat
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock import gptqmodel_runtime as runtime


def _import_map(
    modules: dict[str, object],
):
    def import_module(name: str) -> object:
        value = modules[name]
        if isinstance(value, Exception):
            raise value
        return value

    return import_module


def _runtime_status(
    *,
    importable: bool = True,
    jit_toolchain: runtime.GPTQModelJITToolchainStatus | None = None,
) -> runtime.GPTQModelRuntimeStatus:
    return runtime.GPTQModelRuntimeStatus(
        importable=importable,
        gptqmodel_version="7.0.0" if importable else None,
        import_error_type=None if importable else "ImportError",
        compatibility_bridge_required=False,
        compatibility_bridge_applied=False,
        compatibility_bridge_missing_symbols=(),
        compatibility_bridge_error_type=None,
        jit_toolchain=jit_toolchain,
    )


def test_prepare_bridges_only_missing_transformers_hub_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")
    transformers_hub = SimpleNamespace()
    transformers.utils = SimpleNamespace(hub=transformers_hub)
    huggingface_hub = types.ModuleType("huggingface_hub")
    create_repo = object()
    huggingface_hub.create_repo = create_repo

    class _HfApi:
        def list_repo_tree(self) -> list[object]:
            return []

    huggingface_hub.HfApi = _HfApi
    gptqmodel = types.ModuleType("gptqmodel")
    gptqmodel.__version__ = "test-version"
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "huggingface_hub": huggingface_hub,
                "gptqmodel": gptqmodel,
            }
        ),
    )

    status = runtime.prepare_gptqmodel_runtime()

    assert status.importable is True
    assert status.compatibility_bridge_required is True
    assert status.compatibility_bridge_applied is True
    assert status.compatibility_bridge_missing_symbols == ()
    assert status.compatibility_bridge_error_type is None
    assert transformers_hub.create_repo is create_repo
    assert transformers_hub.list_repo_tree.__self__.__class__ is _HfApi

    repeated = runtime.prepare_gptqmodel_runtime()
    assert repeated.compatibility_bridge_required is True
    assert repeated.compatibility_bridge_applied is True
    assert repeated.compatibility_bridge_missing_symbols == ()
    assert repeated.compatibility_bridge_error_type is None


def test_prepare_rejects_bridge_state_inherited_by_a_different_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")
    transformers_hub = SimpleNamespace()
    transformers.utils = SimpleNamespace(hub=transformers_hub)
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.create_repo = object()

    class _HfApi:
        def list_repo_tree(self) -> list[object]:
            return []

    huggingface_hub.HfApi = _HfApi
    gptqmodel = types.ModuleType("gptqmodel")
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "huggingface_hub": huggingface_hub,
                "gptqmodel": gptqmodel,
            }
        ),
    )
    monkeypatch.setattr(runtime, "_BRIDGE_OBSERVATION", None)
    monkeypatch.setattr(runtime.os, "getpid", lambda: 101)
    parent = runtime.prepare_gptqmodel_runtime()
    assert parent.compatibility_bridge_required is True
    assert parent.compatibility_bridge_applied is True

    monkeypatch.setattr(runtime.os, "getpid", lambda: 202)
    inherited = runtime.prepare_gptqmodel_runtime()
    assert inherited.compatibility_bridge_required is True
    assert inherited.compatibility_bridge_applied is False
    assert inherited.compatibility_bridge_error_type == "InheritedProcessState"
    assert inherited.ready is False


def test_prepare_leaves_supported_transformers_hub_namespace_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_create_repo = object()
    existing_list_repo_tree = object()
    transformers = types.ModuleType("transformers")
    transformers_hub = SimpleNamespace(
        create_repo=existing_create_repo,
        list_repo_tree=existing_list_repo_tree,
    )
    transformers.utils = SimpleNamespace(hub=transformers_hub)
    gptqmodel = types.ModuleType("gptqmodel")
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "gptqmodel": gptqmodel,
            }
        ),
    )

    status = runtime.prepare_gptqmodel_runtime()

    assert status.importable is True
    assert status.compatibility_bridge_required is False
    assert status.compatibility_bridge_applied is False
    assert transformers_hub.create_repo is existing_create_repo
    assert transformers_hub.list_repo_tree is existing_list_repo_tree


def test_prepare_records_gptqmodel_import_failure_without_claiming_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")
    transformers.utils = SimpleNamespace(
        hub=SimpleNamespace(create_repo=object(), list_repo_tree=object())
    )
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "gptqmodel": ImportError("gptqmodel unavailable"),
            }
        ),
    )

    status = runtime.prepare_gptqmodel_runtime()

    assert status.importable is False
    assert status.ready is False
    assert status.import_error_type == "ImportError"
    with pytest.raises(ImportError, match="GPTQModel runtime import failed"):
        runtime.require_gptqmodel_runtime()


def test_failed_required_bridge_fails_closed_even_when_raw_backend_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")
    transformers.utils = SimpleNamespace(hub=SimpleNamespace())
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.create_repo = object()

    class _HfApi:
        def __init__(self) -> None:
            raise TypeError("unsupported hub API")

    huggingface_hub.HfApi = _HfApi
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "huggingface_hub": huggingface_hub,
                "gptqmodel": types.ModuleType("gptqmodel"),
            }
        ),
    )

    status = runtime.prepare_gptqmodel_runtime()

    assert status.importable is True
    assert status.compatibility_bridge_required is True
    assert status.compatibility_bridge_applied is False
    assert status.compatibility_bridge_error_type == "TypeError"
    assert status.ready is False
    assert not hasattr(transformers.utils.hub, "create_repo")
    assert not hasattr(transformers.utils.hub, "list_repo_tree")
    with pytest.raises(ImportError, match="compatibility bridge unavailable"):
        runtime.require_gptqmodel_runtime()


def test_bridge_error_fails_closed_even_without_a_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gptqmodel = types.ModuleType("gptqmodel")
    monkeypatch.setattr(
        runtime,
        "_transformers_hub_compatibility_bridge",
        lambda: runtime._CompatibilityBridgeStatus(  # type: ignore[attr-defined]
            required=False,
            applied=False,
            missing_symbols=(),
            error_type="ImportError",
        ),
    )
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map({"gptqmodel": gptqmodel}),
    )

    status = runtime.prepare_gptqmodel_runtime()
    assert status.ready is False
    with pytest.raises(ImportError, match="compatibility bridge unavailable"):
        runtime.require_gptqmodel_runtime()


def test_bridge_rolls_back_when_second_symbol_assignment_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")

    class _Hub:
        def __setattr__(self, name: str, value: object) -> None:
            if name == "list_repo_tree":
                raise TypeError("second assignment rejected")
            object.__setattr__(self, name, value)

    transformers_hub = _Hub()
    transformers.utils = SimpleNamespace(hub=transformers_hub)
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.create_repo = object()

    class _HfApi:
        def list_repo_tree(self) -> list[object]:
            return []

    huggingface_hub.HfApi = _HfApi
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "huggingface_hub": huggingface_hub,
                "gptqmodel": types.ModuleType("gptqmodel"),
            }
        ),
    )
    monkeypatch.setattr(runtime, "_BRIDGE_OBSERVATION", None)

    status = runtime.prepare_gptqmodel_runtime()
    assert status.compatibility_bridge_required is True
    assert status.compatibility_bridge_applied is False
    assert status.compatibility_bridge_error_type == "TypeError"
    assert not hasattr(transformers_hub, "create_repo")
    assert not hasattr(transformers_hub, "list_repo_tree")
    assert status.ready is False


def test_simple_runtime_preparation_does_not_require_cuda_jit_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")
    transformers.utils = SimpleNamespace(
        hub=SimpleNamespace(create_repo=object(), list_repo_tree=object())
    )
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map(
            {
                "transformers": transformers,
                "gptqmodel": types.ModuleType("gptqmodel"),
            }
        ),
    )
    monkeypatch.setattr(
        runtime,
        "inspect_gptqmodel_jit_toolchain",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected JIT preflight")),
    )

    status = runtime.prepare_gptqmodel_runtime()

    assert status.importable is True
    assert status.jit_toolchain is None
    assert status.ready is True


def test_jit_toolchain_status_fails_closed_when_prerequisites_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", "")
    monkeypatch.setattr(
        runtime, "_ensure_active_interpreter_bin_on_path", lambda: False
    )
    monkeypatch.setattr(runtime, "_python_headers_available", lambda: False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)

    status = runtime.inspect_gptqmodel_jit_toolchain()

    assert status.required is True
    assert status.ready is False
    assert status.interpreter_bin_added_to_path is False
    assert status.missing_requirements == (
        "ninja",
        "python_headers",
        "nvcc",
        "cxx",
    )


def test_jit_toolchain_status_records_complete_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_ensure_active_interpreter_bin_on_path", lambda: True)
    monkeypatch.setattr(runtime, "_python_headers_available", lambda: True)
    monkeypatch.setattr(runtime, "_nvcc_available", lambda: True)
    monkeypatch.setattr(runtime, "_cxx_available", lambda: True)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/tool/ninja")

    status = runtime.inspect_gptqmodel_jit_toolchain()

    assert status.ready is True
    assert status.interpreter_bin_added_to_path is True
    assert status.ninja_available is True
    assert status.python_headers_available is True
    assert status.nvcc_available is True
    assert status.cxx_available is True
    assert status.missing_requirements == ()


def test_jit_preflight_adds_active_interpreter_bin_only_for_venv_ninja(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter_bin = tmp_path / "venv" / "bin"
    interpreter_bin.mkdir(parents=True)
    ninja = interpreter_bin / "ninja"
    ninja.write_text("#!/bin/sh\n", encoding="utf-8")
    ninja.chmod(ninja.stat().st_mode | stat.S_IXUSR)
    unrelated_path = tmp_path / "unrelated"
    unrelated_path.mkdir()
    monkeypatch.setenv("PATH", str(unrelated_path))
    monkeypatch.setattr(runtime, "_active_interpreter_bin", lambda: interpreter_bin)

    added = runtime._ensure_active_interpreter_bin_on_path()

    assert added is True
    assert os.environ["PATH"].split(os.pathsep)[0] == str(interpreter_bin)


def test_active_interpreter_bin_preserves_virtualenv_symlink_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter_bin = tmp_path / "venv" / "bin"
    interpreter_bin.mkdir(parents=True)
    system_interpreter = tmp_path / "system-python"
    system_interpreter.write_text("", encoding="utf-8")
    interpreter = interpreter_bin / "python"
    interpreter.symlink_to(system_interpreter)
    monkeypatch.setattr(runtime.sys, "executable", str(interpreter))

    assert runtime._active_interpreter_bin() == interpreter_bin


def test_nvcc_discovery_uses_torch_cuda_home_when_path_is_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_home = tmp_path / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.CUDA_HOME = str(cuda_home)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _import_map({"torch.utils.cpp_extension": cpp_extension}),
    )

    assert runtime._nvcc_available() is True


def test_require_runtime_rejects_missing_requested_jit_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_toolchain = runtime.GPTQModelJITToolchainStatus(
        required=True,
        interpreter_bin_added_to_path=False,
        ninja_available=False,
        python_headers_available=True,
        nvcc_available=True,
        cxx_available=True,
        ready=False,
        missing_requirements=("ninja",),
    )
    monkeypatch.setattr(
        runtime,
        "prepare_gptqmodel_runtime",
        lambda **_kwargs: _runtime_status(jit_toolchain=incomplete_toolchain),
    )

    with pytest.raises(RuntimeError, match="ninja"):
        runtime.require_gptqmodel_runtime(require_jit_toolchain=True)


def test_require_runtime_reports_missing_python_headers_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_toolchain = runtime.GPTQModelJITToolchainStatus(
        required=True,
        interpreter_bin_added_to_path=False,
        ninja_available=True,
        python_headers_available=False,
        nvcc_available=True,
        cxx_available=True,
        ready=False,
        missing_requirements=("python_headers",),
    )
    monkeypatch.setattr(
        runtime,
        "prepare_gptqmodel_runtime",
        lambda **_kwargs: _runtime_status(jit_toolchain=incomplete_toolchain),
    )

    with pytest.raises(RuntimeError, match="Python\\.h missing"):
        runtime.require_gptqmodel_runtime(require_jit_toolchain=True)
