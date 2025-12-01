from __future__ import annotations

import importlib
import platform as _platform
from typing import Any


def get_adapter_rows() -> list[dict[str, Any]]:
    """Build adapter rows similar to doctor output for testing.

    Applies optional-extra detection for hf_onnx (optimum/onnxruntime) even if
    registered as a core adapter, so missing extras are surfaced.
    """
    from invarlock.core.registry import get_registry

    try:
        import torch as _t  # noqa: F401

        has_cuda = bool(getattr(_t, "cuda", None) and _t.cuda.is_available())
    except Exception:
        has_cuda = False

    registry = get_registry()
    is_linux = _platform.system().lower() == "linux"

    rows: list[dict[str, Any]] = []
    for name in registry.list_adapters():
        info = registry.get_plugin_info(name, "adapters")
        module = str(info.get("module") or "")
        support = (
            "auto"
            if module.startswith("invarlock.adapters")
            and name in {"hf_causal_auto", "hf_mlm_auto"}
            else ("core" if module.startswith("invarlock.adapters") else "optional")
        )
        backend, status, enable = None, "ready", ""

        if name in {"hf_gpt2", "hf_bert", "hf_llama", "hf_causal_auto", "hf_mlm_auto"}:
            backend = "transformers"
        elif name == "hf_gptq":
            backend = "auto-gptq"
            if not is_linux:
                status, enable = "unsupported", "Linux-only"
        elif name == "hf_awq":
            backend = "autoawq"
            if not is_linux:
                status, enable = "unsupported", "Linux-only"
        elif name == "hf_bnb":
            backend = "bitsandbytes"
            if not has_cuda:
                status, enable = "unsupported", "Requires CUDA"
        elif name == "hf_onnx":
            backend = "onnxruntime"
            present = (
                importlib.util.find_spec("optimum.onnxruntime") is not None
                or importlib.util.find_spec("onnxruntime") is not None
            )
            if not present:
                status = "needs_extra"
                enable = "pip install 'invarlock[onnx]'"

        rows.append(
            {
                "name": name,
                "origin": "core" if support in {"core", "auto"} else "plugin",
                "mode": "auto-matcher" if support == "auto" else "adapter",
                "backend": backend,
                "version": None,
                "status": status,
                "enable": enable,
            }
        )

    return rows
