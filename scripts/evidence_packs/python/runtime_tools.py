from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Any

UTC = getattr(datetime, "UTC", datetime.timezone.utc)  # noqa: UP017
_TORCH_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError, RuntimeError)
_TORCH_CONFIG_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_HF_LOAD_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def iso_to_epoch(iso: str) -> int:
    iso = iso.strip()
    if not iso or iso == "null":
        return 0
    try:
        dt = datetime.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return 0
    return int(dt.timestamp())


def now_iso_plus_seconds(seconds: int) -> str:
    dt = datetime.datetime.now(UTC) + datetime.timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env(name: str) -> str:
    return str(os.environ.get(name, "")).strip()


def require_remote_code_opt_in(context: str) -> bool:
    if env_truthy("INVARLOCK_ALLOW_REMOTE_CODE"):
        return True
    raise RuntimeError(
        f"{context} requires INVARLOCK_ALLOW_REMOTE_CODE=1 before using "
        "trust_remote_code=True."
    )


def _resolve_core_loader_strategy_fn():
    try:
        from invarlock.adapters.hf_loading import resolve_core_loader_strategy
    except ImportError:  # pragma: no cover - direct module load under pytest
        src_root = Path(__file__).resolve().parents[2] / "src"
        sys.path.insert(0, str(src_root))
        from invarlock.adapters.hf_loading import resolve_core_loader_strategy
    return resolve_core_loader_strategy


def _checked_remote_code_flag(*, trust_remote_code: bool) -> bool:
    """Require callers to make the remote-code opt-in explicit."""
    return bool(trust_remote_code)


def load_causal_model(
    model_path: Path | str,
    *,
    trust_remote_code: bool,
    **load_kwargs: Any,
) -> tuple[Any, str]:
    resolve_core_loader_strategy = _resolve_core_loader_strategy_fn()
    checked_remote_code = _checked_remote_code_flag(trust_remote_code=trust_remote_code)
    loader_kwargs = dict(load_kwargs)
    loader_kwargs["trust_remote_code"] = checked_remote_code

    primary = resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_path),
        kwargs={"trust_remote_code": checked_remote_code},
        allow_direct_submodule=True,
    )
    strategies = [primary]

    auto_strategy = (
        primary
        if primary.strategy == "auto"
        else resolve_core_loader_strategy(
            task="causal",
            model_id=str(model_path),
            kwargs={"trust_remote_code": checked_remote_code},
            allow_direct_submodule=False,
        )
    )
    direct_fallback = resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_path),
        kwargs={},
        allow_direct_submodule=True,
    )

    if primary.strategy == "auto":
        if direct_fallback.strategy == "direct_submodule":
            strategies.append(direct_fallback)
    else:
        strategies.append(auto_strategy)

    last_error: Exception | None = None
    last_label = "unknown"
    for strategy in strategies:
        last_label = strategy.loader_label
        try:
            model = strategy.loader.from_pretrained(model_path, **loader_kwargs)
            return model, strategy.loader_label
        except _HF_LOAD_ERRORS as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            f"Failed to load causal model via {last_label}: {last_error}"
        ) from last_error
    raise RuntimeError("Failed to resolve causal model loader strategy")


def _load_torch(*, warning: bool) -> object | None:
    try:
        import torch
    except _TORCH_IMPORT_ERRORS as exc:
        level = "WARNING" if warning else "ERROR"
        print(f"{level}: failed to import torch: {exc}", file=sys.stderr)
        return None
    return torch


def torch_env_check() -> int:
    torch = _load_torch(warning=False)
    if torch is None:
        return 1

    print("torch", torch.__version__)
    if not torch.cuda.is_available():
        print("CUDA not available in torch", file=sys.stderr)
        return 1

    print("cuda", torch.version.cuda)
    print("gpus", torch.cuda.device_count())
    print("gpu0", torch.cuda.get_device_name(0))
    print("cc0", torch.cuda.get_device_capability(0))
    return 0


def torch_sm100_warning() -> int:
    torch = _load_torch(warning=True)
    if torch is None or not torch.cuda.is_available():
        return 0

    arch_list = torch.cuda.get_arch_list()
    has_sm100 = any(("sm_100" in arch) or ("compute_100" in arch) for arch in arch_list)
    if has_sm100:
        return 0

    print("WARNING: PyTorch does not report sm_100 (B200) support.")
    print("Install a build with CUDA 12.8+ / sm_100 support, for example:")
    print(
        "  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    return 0


def dataset_preflight() -> int:
    provider = _env("INVARLOCK_DATASET").lower() or "wikitext2"
    if provider != "wikitext2":
        print(f"[DATASET_PREFLIGHT] provider={provider}: skipped")
        return 0

    try:
        from datasets import load_dataset
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "[DATASET_PREFLIGHT] ERROR: datasets library is required for provider=wikitext2."
        )
        print(f"[DATASET_PREFLIGHT] import_error={type(exc).__name__}: {exc}")
        return 1

    offline = _env("HF_DATASETS_OFFLINE")
    hf_home = _env("HF_HOME")
    datasets_cache = _env("HF_DATASETS_CACHE")

    try:
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print("[DATASET_PREFLIGHT] ERROR: failed to load wikitext2 validation split.")
        if offline:
            print(f"[DATASET_PREFLIGHT] HF_DATASETS_OFFLINE={offline}")
        if hf_home:
            print(f"[DATASET_PREFLIGHT] HF_HOME={hf_home}")
        if datasets_cache:
            print(f"[DATASET_PREFLIGHT] HF_DATASETS_CACHE={datasets_cache}")
        print(f"[DATASET_PREFLIGHT] exception={type(exc).__name__}: {exc}")
        return 1

    try:
        size = len(ds)
    except TypeError:
        size = -1

    print(f"[DATASET_PREFLIGHT] OK: provider=wikitext2 split=validation size={size}")
    return 0


def env_report() -> int:
    try:
        import torch
    except _TORCH_IMPORT_ERRORS as exc:  # pragma: no cover - depends on torch install
        print(f"ERROR: torch is required for GPU environment reporting: {exc}")
        return 1

    print("=== Evidence Pack Environment Configuration ===\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    num_gpus = torch.cuda.device_count()
    print(f"GPUs Detected: {num_gpus}")

    mode = str(os.environ.get("PACK_DETERMINISM", "throughput")).strip().lower()
    if mode not in {"throughput", "strict"}:
        mode = "throughput"

    fp8_support = hasattr(torch, "float8_e4m3fn")

    gpu_names: list[str] = []
    gpu_mem_gb: list[float] = []
    total_vram = 0.0

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        gpu_names.append(name)
        gpu_mem_gb.append(mem)
        total_vram += mem
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    min_vram = min(gpu_mem_gb) if gpu_mem_gb else 0.0
    primary_name = gpu_names[0] if gpu_names else ""
    print(f"\nTotal VRAM: {total_vram:.1f} GB")
    print(f"Min GPU VRAM: {min_vram:.1f} GB")
    print(f"FP8 Support: {fp8_support}")

    if mode == "strict":
        print("\nDeterminism mode: strict (PACK_DETERMINISM=strict)")
        try:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=False)
        except _TORCH_CONFIG_ERRORS:
            print("WARNING: deterministic algorithms could not be fully enabled")
        try:
            cudnn_mod = getattr(torch.backends, "cudnn", None)
            if cudnn_mod is not None:
                cudnn_mod.benchmark = False
                cudnn_mod.enabled = True
                if hasattr(cudnn_mod, "deterministic"):
                    cudnn_mod.deterministic = True
                if hasattr(cudnn_mod, "allow_tf32"):
                    cudnn_mod.allow_tf32 = False
        except _TORCH_CONFIG_ERRORS:
            pass
        try:
            matmul = getattr(getattr(torch.backends, "cuda", object()), "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = False
        except _TORCH_CONFIG_ERRORS:
            pass
        print("\nTF32 enabled: False")
        print("cuDNN benchmark: False")
    else:
        print("\nDeterminism mode: throughput (PACK_DETERMINISM=throughput)")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except _TORCH_CONFIG_ERRORS:
            pass
        try:
            cudnn_mod = getattr(torch.backends, "cudnn", None)
            if cudnn_mod is not None:
                cudnn_mod.allow_tf32 = True
                cudnn_mod.benchmark = True
                cudnn_mod.enabled = True
        except _TORCH_CONFIG_ERRORS:
            pass
        print("\nTF32 enabled: True")
        print("cuDNN benchmark: True")

    if torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)
        print("Default dtype: bfloat16")
    else:
        print("Default dtype: float16 (BF16 not supported)")

    try:
        from transformers.utils import is_flash_attn_2_available

        flash_avail = is_flash_attn_2_available()
        print(f"\nFlash Attention 2: {flash_avail}")
    except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError):
        print("\nFlash Attention 2: Unknown (transformers too old)")

    compile_avail = hasattr(torch, "compile")
    print(f"torch.compile: {compile_avail}")

    print(f"\n[PACK_GPU_NAME={primary_name}]")
    print(f"[PACK_GPU_MEM_GB={int(round(min_vram))}]")
    print(f"[PACK_GPU_COUNT={num_gpus}]")
    print("[FP8_NATIVE_SUPPORT=true]" if fp8_support else "[FP8_NATIVE_SUPPORT=false]")

    print("\n=== Environment Ready for Evidence Pack Runs ===")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-pack runtime helpers.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    iso_parser = subparsers.add_parser(
        "iso-to-epoch", help="Convert ISO8601 UTC to epoch."
    )
    iso_parser.add_argument("iso", help="e.g. 2025-01-01T00:00:10Z")

    now_parser = subparsers.add_parser(
        "now-iso-plus-seconds", help="Return now() + delta seconds as ISO8601 UTC."
    )
    now_parser.add_argument("seconds", type=int)

    subparsers.add_parser("torch-env", help="Require torch CUDA and print GPU details.")
    subparsers.add_parser(
        "torch-sm100-warning", help="Warn when torch lacks sm_100 support."
    )
    subparsers.add_parser("dataset-preflight", help="Preflight evidence-pack datasets.")
    subparsers.add_parser("env-report", help="Report GPU environment markers.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "iso-to-epoch":
        print(iso_to_epoch(str(args.iso)))
        return 0
    if args.cmd == "now-iso-plus-seconds":
        print(now_iso_plus_seconds(int(args.seconds)))
        return 0
    if args.cmd == "torch-env":
        return torch_env_check()
    if args.cmd == "torch-sm100-warning":
        return torch_sm100_warning()
    if args.cmd == "dataset-preflight":
        return dataset_preflight()
    if args.cmd == "env-report":
        return env_report()
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
