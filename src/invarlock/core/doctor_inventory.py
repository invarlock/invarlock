from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any

OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("datasets", "Dataset loading (WikiText-2, etc.)"),
    ("transformers", "Hugging Face model support"),
    ("torchvision", "Hugging Face image-text / multimodal processor support"),
    ("gptqmodel", "GPTQ/AWQ quantization backend loading"),
    ("bitsandbytes", "8/4-bit loading (GPU)"),
)
_OPTIONAL_DEP_HINTS = {
    "datasets": "eval",
    "transformers": "adapters",
    "torchvision": "multimodal",
    "gptqmodel": "gptq,awq",
    "bitsandbytes": "gpu",
}


@dataclass(frozen=True)
class DoctorInventoryRow:
    name: str
    origin: str
    mode: str
    backend: str | None
    version: str | None
    status: str
    required_extra: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class DoctorDatasetRow:
    provider: str
    network_mode: str
    available: bool
    params: str


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


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except (TypeError, ValueError, OSError, RuntimeError):
        return None


def _version_key(value: str) -> tuple[int, int, int]:
    parts = [int(match.group(0)) for match in re.finditer(r"\d+", value)]
    padded = (parts + [0, 0, 0])[:3]
    return (padded[0], padded[1], padded[2])


def _version_at_least(version: str | None, minimum: str) -> bool:
    if version is None:
        return False
    return _version_key(version) >= _version_key(minimum)


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


def find_spec_safe(
    module_name: str,
    *,
    find_spec_fn: Callable[[str], object | None] | None = None,
) -> object | None:
    """Best-effort spec lookup that tolerates broken import hooks."""
    return probe_module_spec(module_name, find_spec_fn=find_spec_fn).spec


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

    _ = has_cuda
    results: list[DoctorOptionalDependency] = []
    for dep, description in OPTIONAL_DEPENDENCIES:
        probe = probe_module_spec(dep, find_spec_fn=find_spec_fn)
        present = probe.spec is not None
        runtime_available: bool | None = None
        runtime_probe_failed = False
        if dep == "bitsandbytes" and present:
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


def build_adapter_inventory_rows(
    registry: Any,
    *,
    has_cuda: bool,
    is_linux: bool,
    find_spec_safe: Callable[[str], object | None],
    bitsandbytes_runtime_ready: bool,
) -> list[DoctorInventoryRow]:
    _ = is_linux
    rows: list[DoctorInventoryRow] = []
    transformers_version = _package_version("transformers")

    for name in registry.list_adapters():
        info = registry.get_plugin_info(name, "adapters")
        module = str(info.get("module") or "")
        support = (
            "auto"
            if module.startswith("invarlock.adapters") and name in {"hf_auto"}
            else ("core" if module.startswith("invarlock.adapters") else "optional")
        )
        origin = "core" if support in {"core", "auto"} else "plugin"
        mode = "auto-matcher" if support == "auto" else "adapter"

        backend: str | None = None
        version: str | None = None
        status = "ready"
        required_extra: str | None = None
        detail: str | None = None

        if name in {
            "hf_causal",
            "hf_mlm",
            "hf_multimodal",
            "hf_seq2seq",
            "hf_auto",
        }:
            backend = "transformers"
            version = transformers_version
        elif name in {"hf_gptq", "hf_awq"}:
            backend = "gptqmodel"
        elif name == "hf_bnb":
            backend = "bitsandbytes"
        elif name == "hf_torchao":
            backend = "torchao"
        elif name == "hf_hqq":
            backend = "hqq"
        elif name == "hf_quanto":
            backend = "optimum.quanto"
        elif name == "hf_ct":
            backend = "compressed_tensors"

        if support == "optional":
            present = (
                find_spec_safe((backend or "").replace("-", "_")) is not None
                if backend
                else False
            )
            if not present:
                status = "needs_extra"
                hint = {
                    "hf_gptq": "invarlock[gptq]",
                    "hf_awq": "invarlock[awq]",
                    "hf_bnb": "invarlock[gpu]",
                    "hf_torchao": "invarlock[torchao]",
                    "hf_hqq": "invarlock[hqq]",
                    "hf_quanto": "invarlock[quanto]",
                    "hf_ct": "invarlock[compressed-tensors]",
                }.get(name)
                if hint:
                    required_extra = hint

        if name == "hf_multimodal":
            torchvision_version = _package_version("torchvision")
            if (
                not _version_at_least(transformers_version, "5.12.0")
                or find_spec_safe("torchvision") is None
                or not _version_at_least(torchvision_version, "0.26.0")
            ):
                status = "needs_extra"
                required_extra = "invarlock[multimodal]"
                detail = "Requires transformers>=5.12.0 and torchvision>=0.26.0"

        if (
            backend == "bitsandbytes"
            and find_spec_safe("bitsandbytes") is not None
            and not bitsandbytes_runtime_ready
        ):
            status = "unsupported"
            detail = (
                "bitsandbytes unavailable on this host"
                if has_cuda
                else "Requires CUDA or a compatible bitsandbytes runtime"
            )

        rows.append(
            DoctorInventoryRow(
                name=name,
                origin=origin,
                mode=mode,
                backend=backend,
                version=version,
                status=status,
                required_extra=required_extra,
                detail=detail,
            )
        )
    return rows


def build_generic_inventory_rows(
    registry: Any,
    *,
    kind: str,
    check_plugin_extras: Callable[[str, str], str],
) -> list[DoctorInventoryRow]:
    names = registry.list_guards() if kind == "guards" else registry.list_edits()
    rows: list[DoctorInventoryRow] = []
    for name in names:
        info = registry.get_plugin_info(name, kind)
        module = str(info.get("module") or "")
        origin = "core" if module.startswith(f"invarlock.{kind}") else "plugin"
        mode = "guard" if kind == "guards" else "edit"
        status = "ready"
        required_extra: str | None = None
        try:
            extras = check_plugin_extras(name, kind)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            extras = ""
        if isinstance(extras, str) and extras.startswith("⚠️") and "missing" in extras:
            status = "needs_extra"
            hint = extras.split("missing", 1)[-1].strip()
            if hint:
                required_extra = hint
        rows.append(
            DoctorInventoryRow(
                name=name,
                origin=origin,
                mode=mode,
                backend=None,
                version=None,
                status=status,
                required_extra=required_extra,
            )
        )
    return rows


def summarize_inventory_rows(rows: list[DoctorInventoryRow]) -> dict[str, int]:
    return {
        "total": len(rows),
        "ready": sum(1 for row in rows if row.status == "ready"),
        "needs_extra": sum(1 for row in rows if row.status == "needs_extra"),
        "unsupported": sum(1 for row in rows if row.status == "unsupported"),
        "auto": sum(1 for row in rows if row.mode == "auto-matcher"),
    }


def build_dataset_inventory_rows(
    providers: list[str],
    *,
    provider_network: Mapping[str, str],
    provider_params: Mapping[str, str],
) -> list[DoctorDatasetRow]:
    def _network_mode(name: str) -> str:
        value = (provider_network.get(name, "") or "").lower()
        if value == "cache":
            return "cache"
        if value == "yes":
            return "yes"
        if value == "no":
            return "no"
        return "unknown"

    return [
        DoctorDatasetRow(
            provider=name,
            network_mode=_network_mode(name),
            available=True,
            params=provider_params.get(name, "-"),
        )
        for name in providers
    ]


__all__ = [
    "OPTIONAL_DEPENDENCIES",
    "DoctorDatasetRow",
    "DoctorInventoryRow",
    "DoctorOptionalDependency",
    "DoctorSpecProbeResult",
    "DoctorTorchRuntimeFacts",
    "build_adapter_inventory_rows",
    "build_dataset_inventory_rows",
    "build_generic_inventory_rows",
    "collect_optional_dependency_facts",
    "collect_torch_runtime_facts",
    "find_spec_safe",
    "probe_module_spec",
    "summarize_inventory_rows",
]
