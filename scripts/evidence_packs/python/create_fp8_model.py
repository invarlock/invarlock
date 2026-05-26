from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

try:
    from edit_implementations import apply_fp8_dequantized_simulation, fp8_dtype
    from edit_metadata import build_validation_edit_metadata
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_implementations import apply_fp8_dequantized_simulation, fp8_dtype
    from edit_metadata import build_validation_edit_metadata
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
from transformers import AutoTokenizer


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

    if fp8_dtype(format_type) is None:
        print(
            "WARNING: torch float8 dtype not available; falling back to float16 quantization"
        )

    print(f"Applying FP8 quantization (format={format_type}, scope={scope})...")
    stats = apply_fp8_dequantized_simulation(
        model,
        format_type=format_type,
        scope=scope,
    )
    avg_error = float(stats.details.get("avg_relative_error") or 0.0)
    print(
        f"Quantized {stats.edited_tensors} tensors, avg relative error: {avg_error:.4f}"
    )

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    metadata = build_validation_edit_metadata(
        edit_type="fp8_quant",
        scope=scope,
        parameters={"format": format_type},
        coverage=stats.coverage_payload(),
        extra={
            "quantization_mode": "fp8_dequantized_external_subject_simulation",
            "format": format_type,
            "quantized_tensors": stats.edited_tensors,
            "avg_relative_error": avg_error,
            "torch_fp8_dtype_available": bool(
                stats.details.get("torch_fp8_dtype_available")
            ),
        },
    )
    save_edited_subject_artifact(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        metadata=metadata,
    )

    print(f"Saved FP8-quantized model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
