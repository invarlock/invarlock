from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DoctorTorchRuntimeFacts:
    version: str | None
    device_info: dict[str, Any]
    cuda_toolkit_found: bool | None
    torch_cuda_build: bool | None
    cuda_available: bool | None
    gpu_memory_gb: float | None
    gpu_memory_low: bool


@dataclass(frozen=True)
class DoctorOptionalDependency:
    name: str
    description: str
    present: bool
    extra_hint: str
    runtime_available: bool | None = None


OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("datasets", "Dataset loading (WikiText-2, etc.)"),
    ("transformers", "Hugging Face model support"),
    ("auto_gptq", "GPTQ quantization (Linux/CUDA only)"),
    ("autoawq", "AWQ quantization (Linux/CUDA only)"),
    ("bitsandbytes", "8/4-bit loading (GPU)"),
)

_OPTIONAL_DEP_HINTS = {
    "datasets": "eval",
    "transformers": "adapters",
    "auto_gptq": "gptq",
    "autoawq": "awq",
    "bitsandbytes": "gpu",
}


def find_spec_safe(
    module_name: str,
    *,
    find_spec_fn: Callable[[str], object | None] | None = None,
) -> object | None:
    """Best-effort spec lookup that tolerates broken import hooks."""

    finder = find_spec_fn or importlib.util.find_spec
    try:
        return finder(module_name)
    except Exception:
        return None


def collect_torch_runtime_facts(
    *,
    import_torch_fn: Callable[[], Any],
    get_device_info_fn: Callable[[], dict[str, Any]],
    which_fn: Callable[[str], str | None],
) -> DoctorTorchRuntimeFacts:
    """Collect runtime/device facts for doctor without printing."""

    torch = import_torch_fn()
    torch_version = getattr(torch, "__version__", None)
    device_info = get_device_info_fn()

    cuda_toolkit_found: bool | None = None
    torch_cuda_build: bool | None = None
    cuda_available: bool | None = None
    gpu_memory_gb: float | None = None
    gpu_memory_low = False

    try:
        cuda_toolkit_found = bool(which_fn("nvcc") or which_fn("nvidia-smi"))
        torch_cuda_build = bool(getattr(torch.version, "cuda", None))
        cuda_available = bool(
            getattr(torch, "cuda", None) and torch.cuda.is_available()
        )
    except Exception:
        cuda_toolkit_found = None
        torch_cuda_build = None
        cuda_available = None

    try:
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_memory_low = gpu_memory_gb < 4.0
    except Exception:
        gpu_memory_gb = None
        gpu_memory_low = False

    return DoctorTorchRuntimeFacts(
        version=torch_version,
        device_info=device_info,
        cuda_toolkit_found=cuda_toolkit_found,
        torch_cuda_build=torch_cuda_build,
        cuda_available=cuda_available,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_low=gpu_memory_low,
    )


def collect_optional_dependency_facts(
    *,
    has_cuda: bool,
    bitsandbytes_runtime_available_fn: Callable[[], bool],
    find_spec_fn: Callable[[str], object | None] | None = None,
) -> list[DoctorOptionalDependency]:
    """Collect optional dependency presence/runtime facts for doctor."""

    results: list[DoctorOptionalDependency] = []
    for dep, description in OPTIONAL_DEPENDENCIES:
        present = find_spec_safe(dep, find_spec_fn=find_spec_fn) is not None
        runtime_available: bool | None = None
        if dep == "bitsandbytes":
            runtime_available = present and bitsandbytes_runtime_available_fn()
        results.append(
            DoctorOptionalDependency(
                name=dep,
                description=description,
                present=present,
                extra_hint=_OPTIONAL_DEP_HINTS.get(dep, dep),
                runtime_available=runtime_available,
            )
        )
    return results


__all__ = [
    "DoctorOptionalDependency",
    "DoctorTorchRuntimeFacts",
    "OPTIONAL_DEPENDENCIES",
    "collect_optional_dependency_facts",
    "collect_torch_runtime_facts",
    "find_spec_safe",
]
