from __future__ import annotations

import os


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on torch availability
        print(f"ERROR: torch is required for GPU environment reporting: {exc}")
        return 1

    print("=== Proof Pack Environment Configuration ===\n")

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
        except Exception:
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
        except Exception:
            pass
        try:
            matmul = getattr(getattr(torch.backends, "cuda", object()), "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = False
        except Exception:
            pass
        print("\nTF32 enabled: False")
        print("cuDNN benchmark: False")
    else:
        print("\nDeterminism mode: throughput (PACK_DETERMINISM=throughput)")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            cudnn_mod = getattr(torch.backends, "cudnn", None)
            if cudnn_mod is not None:
                cudnn_mod.allow_tf32 = True
                cudnn_mod.benchmark = True
                cudnn_mod.enabled = True
        except Exception:
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
    except Exception:
        print("\nFlash Attention 2: Unknown (transformers too old)")

    compile_avail = hasattr(torch, "compile")
    print(f"torch.compile: {compile_avail}")

    print(f"\n[PACK_GPU_NAME={primary_name}]")
    print(f"[PACK_GPU_MEM_GB={int(round(min_vram))}]")
    print(f"[PACK_GPU_COUNT={num_gpus}]")
    print("[FP8_NATIVE_SUPPORT=true]" if fp8_support else "[FP8_NATIVE_SUPPORT=false]")

    print("\n=== Environment Ready for Proof Pack Runs ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
