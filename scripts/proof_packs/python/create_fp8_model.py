from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _should_quantize(name: str, scope: str) -> bool:
    name_lower = name.lower()
    if scope == "all":
        return "weight" in name_lower and any(
            x in name_lower
            for x in ("linear", "proj", "fc", "mlp", "attn", "wqkv", "query_key_value")
        )
    if scope == "ffn":
        return "weight" in name_lower and any(
            x in name_lower
            for x in (
                "mlp",
                "fc",
                "gate",
                "up_proj",
                "down_proj",
                "dense_h_to_4h",
                "dense_4h_to_h",
            )
        )
    if scope == "attn":
        return "weight" in name_lower and any(
            x in name_lower
            for x in (
                "attn",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "wqkv",
                "out_proj",
                "query_key_value",
            )
        )
    return False


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
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)

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

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    metadata = {
        "edit_type": "fp8_quant",
        "format": format_type,
        "scope": scope,
        "quantized_tensors": quantized_count,
        "avg_relative_error": avg_error,
    }
    (output_path / "edit_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved FP8-quantized model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
