from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _load_runtime_tools():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evidence_packs" / "python" / "runtime_tools.py"
    spec = importlib.util.spec_from_file_location("evidence_pack_runtime_tools", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_tools_python_helpers_use_portable_utc() -> None:
    runtime_tools = _load_runtime_tools()
    assert runtime_tools.iso_to_epoch("2025-01-01T00:00:10Z") == 1735689610
    assert runtime_tools.iso_to_epoch("") == 0
    assert runtime_tools.now_iso_plus_seconds(0).endswith("Z")


def test_dataset_preflight_skips_non_wikitext2(monkeypatch, capsys) -> None:
    runtime_tools = _load_runtime_tools()
    monkeypatch.setenv("INVARLOCK_DATASET", "synthetic")

    assert runtime_tools.dataset_preflight() == 0

    assert "[DATASET_PREFLIGHT] provider=synthetic: skipped" in capsys.readouterr().out


def test_env_report_emits_validation_markers(monkeypatch, capsys) -> None:
    runtime_tools = _load_runtime_tools()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Fake GPU"

        @staticmethod
        def get_device_properties(index: int) -> types.SimpleNamespace:
            assert index == 0
            return types.SimpleNamespace(total_memory=48 * 1024**3)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        cuda=_Cuda(),
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(
                allow_tf32=False,
                benchmark=False,
                enabled=False,
                deterministic=False,
            ),
        ),
        bfloat16=object(),
        float8_e4m3fn=object(),
        compile=object(),
        set_default_dtype=lambda _dtype: None,
    )
    transformers = types.ModuleType("transformers")
    transformers.__path__ = []  # type: ignore[attr-defined]
    transformers_utils = types.ModuleType("transformers.utils")
    transformers_utils.is_flash_attn_2_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)

    assert runtime_tools.env_report() == 0

    out = capsys.readouterr().out
    assert "[PACK_GPU_NAME=Fake GPU]" in out
    assert "[PACK_GPU_MEM_GB=48]" in out
    assert "[PACK_GPU_COUNT=1]" in out
    assert "[FP8_NATIVE_SUPPORT=true]" in out


def test_load_causal_model_threads_remote_code_and_falls_back(monkeypatch) -> None:
    runtime_tools = _load_runtime_tools()
    calls: list[dict[str, Any]] = []
    loader_calls: list[tuple[str, str, bool]] = []

    class _Loader:
        def __init__(self, label: str, *, fail: bool = False) -> None:
            self.label = label
            self.fail = fail

        def from_pretrained(self, model_path: str, **kwargs: Any) -> object:
            loader_calls.append(
                (self.label, model_path, bool(kwargs["trust_remote_code"]))
            )
            if self.fail:
                raise RuntimeError("loader failed")
            return {"label": self.label}

    def fake_resolve_core_loader_strategy(
        *,
        task: str,
        model_id: str,
        kwargs: dict[str, object],
        allow_direct_submodule: bool,
    ) -> types.SimpleNamespace:
        calls.append(
            {
                "task": task,
                "model_id": model_id,
                "kwargs": dict(kwargs),
                "allow_direct_submodule": allow_direct_submodule,
            }
        )
        if allow_direct_submodule and kwargs:
            return types.SimpleNamespace(
                strategy="direct_submodule",
                loader_label="direct",
                loader=_Loader("direct", fail=True),
            )
        return types.SimpleNamespace(
            strategy="auto",
            loader_label="auto",
            loader=_Loader("auto"),
        )

    monkeypatch.setattr(
        runtime_tools,
        "_resolve_core_loader_strategy_fn",
        lambda: fake_resolve_core_loader_strategy,
    )

    model, label = runtime_tools.load_causal_model(
        "org/model",
        trust_remote_code=True,
        dtype="bf16",
    )

    assert model == {"label": "auto"}
    assert label == "auto"
    assert calls[0]["kwargs"] == {"trust_remote_code": True}
    assert calls[0]["allow_direct_submodule"] is True
    assert calls[1]["allow_direct_submodule"] is False
    assert loader_calls == [
        ("direct", "org/model", True),
        ("auto", "org/model", True),
    ]
