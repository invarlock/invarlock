from __future__ import annotations

import gc
import json
import shutil
import sys
from pathlib import Path

import torch

try:
    from edit_targeting import matches_edit_scope
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_targeting import matches_edit_scope
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
from transformers import AutoTokenizer


def _should_quantize(name: str, scope: str) -> bool:
    return matches_edit_scope(name, scope)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "Usage: create_fp8_model.py <baseline_path> <output_path> <format> <scope>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    output_path = Path(argv[2])
    format_type = argv[3]
    scope = argv[4]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print(f"Loading baseline from {baseline_path}...")
    trust_remote_code = require_remote_code_opt_in("create_fp8_model.py")
    model_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    model, _ = load_causal_model(baseline_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path, trust_remote_code=trust_remote_code
    )

    if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    else:
        fp8_dtype = getattr(torch, "float8_e5m2", None)

    if fp8_dtype is None:
        print(
            "WARNING: torch float8 dtype not available; falling back to float16 quantization"
        )

    @torch.no_grad()
    def quantize_fp8(tensor: torch.Tensor) -> torch.Tensor:
        if fp8_dtype is None:
            return tensor.to(torch.float16).to(tensor.dtype)
        return tensor.to(fp8_dtype).to(tensor.dtype)

    print(f"Applying FP8 quantization (format={format_type}, scope={scope})...")
    quantized_count = 0
    num_tensors = 0
    rel_error_total = 0.0
    edited_params = 0
    for name, param in model.named_parameters():
        if not _should_quantize(name, scope) or param.dim() < 2:
            continue
        original = param.data.clone()
        param.data = quantize_fp8(param.data)
        num_tensors += 1
        quantized_count += 1
        edited_params += param.numel()
        denom = original.abs().mean() + 1e-10
        rel_error_total += float((param.data - original).abs().mean() / denom)
        if quantized_count <= 3:
            print(f"  FP8: {name}")

    avg_error = rel_error_total / max(num_tensors, 1)
    print(f"Quantized {quantized_count} tensors, avg relative error: {avg_error:.4f}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    staging_path = output_path.parent / f".{output_path.name}.tmp"
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging_path, safe_serialization=True)
    tokenizer.save_pretrained(staging_path)

    metadata = {
        "edit_type": "fp8_quant",
        "format": format_type,
        "scope": scope,
        "quantized_tensors": quantized_count,
        "avg_relative_error": avg_error,
    }
    (staging_path / "edit_metadata.json").write_text(json.dumps(metadata, indent=2))

    if output_path.exists():
        shutil.rmtree(output_path)
    staging_path.rename(output_path)

    print(f"Saved FP8-quantized model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
