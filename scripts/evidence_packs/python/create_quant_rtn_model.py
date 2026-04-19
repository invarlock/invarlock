from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch

try:
    from edit_targeting import matches_edit_scope
    from runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_targeting import matches_edit_scope
    from runtime_tools import require_remote_code_opt_in
from transformers import AutoModelForCausalLM, AutoTokenizer


def _configure_determinism() -> None:
    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _should_quantize(name: str, scope: str) -> bool:
    return matches_edit_scope(name, scope)


@torch.no_grad()
def _round_to_nearest_gpu(
    tensor: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    qmin = -(2 ** (bits - 1))
    qmax = max((2 ** (bits - 1)) - 1, 1)
    orig_shape = tensor.shape
    flat = tensor.reshape(orig_shape[0], -1)
    in_features = flat.shape[1]
    eff_group_size = group_size if group_size > 0 else in_features
    if eff_group_size >= in_features:
        eff_group_size = in_features
    num_groups = (in_features + eff_group_size - 1) // eff_group_size
    pad = (num_groups * eff_group_size) - in_features
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(orig_shape[0], num_groups, eff_group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-10)
    quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
    quantized = quantized.reshape(orig_shape[0], num_groups * eff_group_size)
    if pad > 0:
        quantized = quantized[:, :in_features]
    return quantized.reshape(orig_shape).to(tensor.dtype)


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(
            "Usage: create_quant_rtn_model.py <baseline_path> <output_path> <bits> "
            "<group_size> <scope>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    output_path = Path(argv[2])
    bits = int(argv[3])
    group_size = int(argv[4])
    scope = argv[5]

    _configure_determinism()

    print(f"Loading baseline from {baseline_path}...")
    trust_remote_code = require_remote_code_opt_in("create_quant_rtn_model.py")
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path, trust_remote_code=trust_remote_code
    )
    flash_available = os.environ.get("FLASH_ATTENTION_AVAILABLE", "false") == "true"

    model_kwargs: dict[str, object] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if flash_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)

    print(f"Quantizing to {bits}-bit on GPU (scope={scope})...")
    quantized_count = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if _should_quantize(name, scope) and param.dim() >= 2:
            param.data = _round_to_nearest_gpu(param.data, bits, group_size)
            quantized_count += 1
            edited_params += param.numel()
            if quantized_count <= 3:
                print(f"  Quantized: {name} ({tuple(param.shape)})")

    coverage_pct = (
        100.0 * edited_params / total_model_params if total_model_params else 0.0
    )
    print(
        f"Quantized {quantized_count} parameters "
        f"({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)"
    )

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "quant_rtn",
        "bits": bits,
        "group_size": group_size,
        "scope": scope,
        "quantized_params": quantized_count,
    }
    (output_path / "edit_metadata.json").write_text(json.dumps(metadata, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved edited model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
