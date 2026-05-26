from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

try:
    from edit_implementations import apply_rtn_dequantized_simulation
    from edit_metadata import build_validation_edit_metadata
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_implementations import apply_rtn_dequantized_simulation
    from edit_metadata import build_validation_edit_metadata
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
from transformers import AutoTokenizer


def _configure_determinism() -> None:
    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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

    model, _ = load_causal_model(baseline_path, **model_kwargs)

    print(
        "Applying RTN quantize/dequantize simulation "
        f"to {bits}-bit on GPU (scope={scope}, group_size={group_size})..."
    )
    stats = apply_rtn_dequantized_simulation(
        model,
        bits=bits,
        group_size=group_size,
        scope=scope,
    )
    coverage_pct = 100.0 * stats.coverage_ratio
    print(
        f"Quantized {stats.edited_tensors} tensors "
        f"({stats.edited_params:,} / {stats.total_params:,} = {coverage_pct:.1f}% coverage)"
    )

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    metadata = build_validation_edit_metadata(
        edit_type="quant_rtn",
        scope=scope,
        parameters={"bits": bits, "group_size": group_size},
        coverage=stats.coverage_payload(),
        extra={
            "quantization_mode": "rtn_dequantized_external_subject_simulation",
            "quantized_params": stats.edited_tensors,
        },
    )
    save_edited_subject_artifact(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        metadata=metadata,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved edited model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
