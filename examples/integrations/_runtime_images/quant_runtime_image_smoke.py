#!/usr/bin/env python3
"""Smoke-check the CUDA quant runtime image dependency surface."""

from __future__ import annotations

import argparse
import importlib
import sys

CORE_RUNTIME_IMPORTS = (
    "datasets",
    "peft",
    "safetensors",
    "torch",
    "transformers",
)

DEFAULT_QUANT_ADAPTERS = (
    "hf_bnb",
    "hf_awq",
    "hf_gptq",
    "hf_torchao",
    "hf_hqq",
    "hf_quanto",
    "hf_ct",
)

QUANT_BACKEND_IMPORTS = {
    "hf_bnb": "bitsandbytes",
    "hf_awq": "gptqmodel",
    "hf_gptq": "gptqmodel",
    "hf_torchao": "torchao",
    "hf_hqq": "hqq",
    "hf_quanto": "optimum.quanto",
    "hf_ct": "compressed_tensors",
}

BROAD_QUANT_BACKEND_IMPORTS = (
    "bitsandbytes",
    "gptqmodel",
    "hqq",
    "optimum.quanto",
    "compressed_tensors",
    "torchao",
)

QUANT_ADAPTER_BACKENDS = {
    "hf_bnb": "bitsandbytes",
    "hf_awq": "gptqmodel",
    "hf_gptq": "gptqmodel",
    "hf_torchao": "torchao",
    "hf_hqq": "hqq",
    "hf_quanto": "optimum-quanto",
    "hf_ct": "compressed-tensors",
}

OPTIONAL_IMPORT_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
)


def _fail(message: str) -> None:
    raise SystemExit(message)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check a CUDA quant runtime image dependency surface."
    )
    parser.add_argument(
        "--adapters",
        default=",".join(DEFAULT_QUANT_ADAPTERS),
        help=(
            "Comma-separated quant adapters to prove. "
            "Default: all supported quant adapters."
        ),
    )
    parser.add_argument(
        "--require-cuda-toolchain",
        action="store_true",
        help="Require nvcc, CUDA_HOME, and Python.h for JIT-backed quant runtimes.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require a visible CUDA device through the container runtime.",
    )
    return parser.parse_args(argv)


def _selected_adapters(raw_value: str) -> tuple[str, ...]:
    selected = tuple(
        adapter.strip() for adapter in raw_value.split(",") if adapter.strip()
    )
    if not selected:
        _fail("at least one quant adapter is required")
    unknown = sorted(set(selected) - set(QUANT_ADAPTER_BACKENDS))
    if unknown:
        _fail("unknown quant adapter(s): " + ", ".join(unknown))
    return selected


_GPTQMODEL_ADAPTERS = frozenset({"hf_awq", "hf_gptq"})


def _require_gptqmodel_runtime(
    selected_adapters: tuple[str, ...],
    *,
    require_jit_toolchain: bool,
) -> None:
    if not _GPTQMODEL_ADAPTERS.intersection(selected_adapters):
        return
    try:
        from invarlock.gptqmodel_runtime import require_gptqmodel_runtime

        require_gptqmodel_runtime(require_jit_toolchain=require_jit_toolchain)
    except RuntimeError as exc:
        # The named runtime boundary uses path-free diagnostic names such as
        # ``Python.h missing``.  Preserve those actionable prerequisites rather
        # than reducing a CUDA image failure to its exception type.
        _fail(str(exc))
    except OPTIONAL_IMPORT_ERRORS as exc:
        _fail(f"GPTQModel runtime unavailable: {type(exc).__name__}")


def _import_required_modules(selected_adapters: tuple[str, ...]) -> None:
    _require_gptqmodel_runtime(
        selected_adapters,
        require_jit_toolchain=False,
    )
    backend_imports = {
        QUANT_BACKEND_IMPORTS[adapter_name] for adapter_name in selected_adapters
    }
    if selected_adapters == DEFAULT_QUANT_ADAPTERS:
        backend_imports.update(BROAD_QUANT_BACKEND_IMPORTS)

    missing: list[str] = []
    for module_name in sorted(set(CORE_RUNTIME_IMPORTS) | backend_imports):
        try:
            importlib.import_module(module_name)
        except OPTIONAL_IMPORT_ERRORS:
            missing.append(module_name)
    if missing:
        _fail("missing quant runtime modules: " + ", ".join(sorted(missing)))


def _check_peft_awq_dispatcher(selected_adapters: tuple[str, ...]) -> None:
    """Exercise PEFT's deferred GPTQModel AWQ import on a non-AWQ probe."""
    if "hf_awq" not in selected_adapters:
        return
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.awq import dispatch_awq

    try:
        replacement = dispatch_awq(
            torch.nn.Linear(1, 1),
            "default",
            config=LoraConfig(r=1, target_modules=[]),
        )
    except OPTIONAL_IMPORT_ERRORS as exc:
        _fail(f"PEFT AWQ runtime unavailable: {type(exc).__name__}")
    if replacement is not None:
        _fail("PEFT AWQ dispatcher replaced a non-AWQ probe")


def _check_cuda_runtime(
    *,
    selected_adapters: tuple[str, ...],
    require_cuda_toolchain: bool,
    require_gpu: bool,
) -> None:
    import torch

    if torch.version.cuda is None:
        _fail("torch CUDA build missing")
    if require_gpu and not torch.cuda.is_available():
        _fail("CUDA device unavailable through the container runtime")
    if require_gpu:
        torch.cuda.get_device_name(0)
    if not require_cuda_toolchain:
        return

    from torch.utils.cpp_extension import CUDA_HOME

    if not CUDA_HOME:
        _fail("CUDA_HOME missing")
    _require_gptqmodel_runtime(
        selected_adapters,
        require_jit_toolchain=True,
    )


def _check_adapter_backend_contract(selected_adapters: tuple[str, ...]) -> None:
    from invarlock.core.backend_inventory import (
        extract_adapter_provenance,
        quantized_adapter_backend,
    )

    for adapter_name in selected_adapters:
        expected_backend = QUANT_ADAPTER_BACKENDS[adapter_name]
        backend = quantized_adapter_backend(adapter_name)
        if backend != expected_backend:
            _fail(
                f"{adapter_name} backend mismatch: expected "
                f"{expected_backend}, got {backend}"
            )

        provenance = extract_adapter_provenance(adapter_name)
        if not provenance.supported:
            _fail(
                f"{adapter_name} backend unavailable: "
                f"{provenance.library} {provenance.version or 'missing'}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    selected_adapters = _selected_adapters(args.adapters)
    _import_required_modules(selected_adapters)
    _check_peft_awq_dispatcher(selected_adapters)
    _check_cuda_runtime(
        selected_adapters=selected_adapters,
        require_cuda_toolchain=args.require_cuda_toolchain,
        require_gpu=args.require_gpu,
    )
    _check_adapter_backend_contract(selected_adapters)
    print("quant runtime image imports ok: " + ", ".join(sorted(selected_adapters)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
