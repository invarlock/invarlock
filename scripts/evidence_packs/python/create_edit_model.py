from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from .editing.implementations import (
        apply_dense_lowrank_approximation,
        apply_dense_magnitude_prune,
        apply_fp8_dequantized_simulation,
        apply_rtn_dequantized_simulation,
        build_validation_edit_metadata,
        fp8_dtype,
    )
    from .editing.validate_artifact import save_edited_subject_artifact
    from .runtime_tools import load_causal_model, require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.implementations import (
        apply_dense_lowrank_approximation,
        apply_dense_magnitude_prune,
        apply_fp8_dequantized_simulation,
        apply_rtn_dequantized_simulation,
        build_validation_edit_metadata,
        fp8_dtype,
    )
    from editing.validate_artifact import save_edited_subject_artifact
    from runtime_tools import load_causal_model, require_remote_code_opt_in
from transformers import AutoTokenizer


def _configure_determinism() -> None:
    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)


def _clear_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _load_model_and_tokenizer(
    baseline_path: Path,
    *,
    require_cuda: bool = False,
    flash_attention: bool = False,
) -> tuple[Any, Any]:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print(f"Loading baseline from {baseline_path}...")
    trust_remote_code = require_remote_code_opt_in("create_edit_model.py")
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path, trust_remote_code=trust_remote_code
    )
    model_kwargs: dict[str, object] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model, _ = load_causal_model(baseline_path, **model_kwargs)
    return model, tokenizer


def _save_model(
    *,
    model: Any,
    tokenizer: Any,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    model = model.cpu()
    _clear_memory()
    save_edited_subject_artifact(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        metadata=metadata,
    )
    del model
    _clear_memory()


def _create_quant_rtn(args: argparse.Namespace) -> int:
    _configure_determinism()
    flash_available = os.environ.get("FLASH_ATTENTION_AVAILABLE", "false") == "true"
    model, tokenizer = _load_model_and_tokenizer(
        Path(args.baseline_path), flash_attention=flash_available
    )

    bits = int(args.bits)
    group_size = int(args.group_size)
    scope = str(args.scope)
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
    _save_model(
        model=model,
        tokenizer=tokenizer,
        output_path=Path(args.output_path),
        metadata=metadata,
    )
    print(f"Saved edited model to {args.output_path}")
    return 0


def _create_magnitude_prune(args: argparse.Namespace) -> int:
    _configure_determinism()
    model, tokenizer = _load_model_and_tokenizer(Path(args.baseline_path))
    sparsity = float(args.sparsity)
    scope = str(args.scope)

    print(f"Pruning with sparsity={sparsity} (scope={scope})...")
    stats = apply_dense_magnitude_prune(model, sparsity=sparsity, scope=scope)
    actual_sparsity = float(stats.details.get("actual_sparsity") or 0.0)
    coverage_pct = 100.0 * stats.coverage_ratio
    print(
        f"Pruned {stats.edited_tensors} tensors "
        f"({stats.edited_params:,} / {stats.total_params:,} = {coverage_pct:.1f}% coverage)"
    )
    print(f"Actual sparsity within edited params: {actual_sparsity:.2%}")

    metadata = build_validation_edit_metadata(
        edit_type="magnitude_prune",
        scope=scope,
        parameters={"target_sparsity": sparsity},
        coverage=stats.coverage_payload(),
        extra={
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "pruned_params": stats.edited_tensors,
        },
    )
    _save_model(
        model=model,
        tokenizer=tokenizer,
        output_path=Path(args.output_path),
        metadata=metadata,
    )
    print(f"Saved pruned model to {args.output_path}")
    return 0


def _create_lowrank_svd(args: argparse.Namespace) -> int:
    _configure_determinism()
    model, tokenizer = _load_model_and_tokenizer(Path(args.baseline_path))
    rank = int(args.rank)
    scope = str(args.scope)

    print(f"Applying low-rank SVD with rank={rank} (scope={scope})...")
    stats = apply_dense_lowrank_approximation(model, rank=rank, scope=scope)
    avg_energy = float(stats.details.get("avg_energy_retained") or 1.0)
    coverage_pct = 100.0 * stats.coverage_ratio
    print(
        f"Modified {stats.edited_tensors} matrices "
        f"({stats.edited_params:,} / {stats.total_params:,} = {coverage_pct:.1f}% coverage)"
    )
    print(f"Average energy retained: {avg_energy:.2%}")

    metadata = build_validation_edit_metadata(
        edit_type="lowrank_svd",
        scope=scope,
        parameters={"rank": rank},
        coverage=stats.coverage_payload(),
        extra={
            "rank": rank,
            "modified_matrices": stats.edited_tensors,
            "avg_energy_retained": avg_energy,
            "base_scope": stats.details.get("base_scope"),
            "layer_limit": stats.details.get("layer_limit"),
            "layer": stats.details.get("layer"),
        },
    )
    _save_model(
        model=model,
        tokenizer=tokenizer,
        output_path=Path(args.output_path),
        metadata=metadata,
    )
    print(f"Saved low-rank model to {args.output_path}")
    return 0


def _create_fp8_quant(args: argparse.Namespace) -> int:
    _configure_determinism()
    format_type = str(args.format)
    scope = str(args.scope)
    model, tokenizer = _load_model_and_tokenizer(
        Path(args.baseline_path), require_cuda=True
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
    _save_model(
        model=model,
        tokenizer=tokenizer,
        output_path=Path(args.output_path),
        metadata=metadata,
    )
    print(f"Saved FP8-quantized model to {args.output_path}")
    return 0


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("baseline_path")
    parser.add_argument("output_path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a single edited evidence-pack subject checkpoint."
    )
    subparsers = parser.add_subparsers(dest="edit_type", required=True)

    quant = subparsers.add_parser("quant-rtn")
    _add_common_paths(quant)
    quant.add_argument("bits")
    quant.add_argument("group_size")
    quant.add_argument("scope")
    quant.set_defaults(func=_create_quant_rtn)

    prune = subparsers.add_parser("magnitude-prune")
    _add_common_paths(prune)
    prune.add_argument("sparsity")
    prune.add_argument("scope")
    prune.set_defaults(func=_create_magnitude_prune)

    lowrank = subparsers.add_parser("lowrank-svd")
    _add_common_paths(lowrank)
    lowrank.add_argument("rank")
    lowrank.add_argument("scope")
    lowrank.set_defaults(func=_create_lowrank_svd)

    fp8 = subparsers.add_parser("fp8-quant")
    _add_common_paths(fp8)
    fp8.add_argument("format")
    fp8.add_argument("scope")
    fp8.set_defaults(func=_create_fp8_quant)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    finally:
        _clear_memory()


if __name__ == "__main__":
    raise SystemExit(main())
