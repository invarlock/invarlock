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
    cuda_probe_failed: bool = False
    gpu_memory_probe_failed: bool = False


@dataclass(frozen=True)
class DoctorOptionalDependency:
    name: str
    description: str
    present: bool
    extra_hint: str
    runtime_available: bool | None = None
    spec_probe_failed: bool = False
    runtime_probe_failed: bool = False


@dataclass(frozen=True)
class DoctorSpecProbeResult:
    spec: object | None
    failed: bool


OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("datasets", "Dataset loading (WikiText-2, etc.)"),
    ("transformers", "Hugging Face model support"),
    ("gptqmodel", "GPTQ/AWQ quantization backend loading"),
    ("bitsandbytes", "8/4-bit loading (GPU)"),
)

_OPTIONAL_DEP_HINTS = {
    "datasets": "eval",
    "transformers": "adapters",
    "gptqmodel": "gptq,awq",
    "bitsandbytes": "gpu",
}


def find_spec_safe(
    module_name: str,
    *,
    find_spec_fn: Callable[[str], object | None] | None = None,
) -> object | None:
    """Best-effort spec lookup that tolerates broken import hooks."""
    return probe_module_spec(module_name, find_spec_fn=find_spec_fn).spec


def probe_module_spec(
    module_name: str,
    *,
    find_spec_fn: Callable[[str], object | None] | None = None,
) -> DoctorSpecProbeResult:
    """Probe a module spec while preserving whether the probe itself failed."""
    finder = find_spec_fn or importlib.util.find_spec
    try:
        return DoctorSpecProbeResult(spec=finder(module_name), failed=False)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return DoctorSpecProbeResult(spec=None, failed=True)


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
    cuda_probe_failed = False
    gpu_memory_probe_failed = False

    try:
        cuda_toolkit_found = bool(which_fn("nvcc") or which_fn("nvidia-smi"))
        torch_cuda_build = bool(getattr(torch.version, "cuda", None))
        cuda_available = bool(
            getattr(torch, "cuda", None) and torch.cuda.is_available()
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        cuda_toolkit_found = None
        torch_cuda_build = None
        cuda_available = None
        cuda_probe_failed = True

    try:
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_memory_low = gpu_memory_gb < 4.0
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        gpu_memory_gb = None
        gpu_memory_low = False
        gpu_memory_probe_failed = True

    return DoctorTorchRuntimeFacts(
        version=torch_version,
        device_info=device_info,
        cuda_toolkit_found=cuda_toolkit_found,
        torch_cuda_build=torch_cuda_build,
        cuda_available=cuda_available,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_low=gpu_memory_low,
        cuda_probe_failed=cuda_probe_failed,
        gpu_memory_probe_failed=gpu_memory_probe_failed,
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
        probe = probe_module_spec(dep, find_spec_fn=find_spec_fn)
        present = probe.spec is not None
        runtime_available: bool | None = None
        runtime_probe_failed = False
        if dep == "bitsandbytes":
            if present:
                try:
                    runtime_available = bitsandbytes_runtime_available_fn()
                except (
                    AttributeError,
                    ImportError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    runtime_available = None
                    runtime_probe_failed = True
        results.append(
            DoctorOptionalDependency(
                name=dep,
                description=description,
                present=present,
                extra_hint=_OPTIONAL_DEP_HINTS.get(dep, dep),
                runtime_available=runtime_available,
                spec_probe_failed=probe.failed,
                runtime_probe_failed=runtime_probe_failed,
            )
        )
    return results


__all__ = [
    "DoctorOptionalDependency",
    "DoctorSpecProbeResult",
    "DoctorTorchRuntimeFacts",
    "OPTIONAL_DEPENDENCIES",
    "collect_optional_dependency_facts",
    "collect_torch_runtime_facts",
    "find_spec_safe",
    "probe_module_spec",
]
