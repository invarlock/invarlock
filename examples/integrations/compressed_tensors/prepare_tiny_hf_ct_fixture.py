#!/usr/bin/env python3
"""Prepare tiny HF checkpoints and fixture for the hf_ct example."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TERMS = (
    "invarlock",
    "adapter",
    "compressed",
    "tensors",
    "quantized",
    "baseline",
    "subject",
    "regression",
    "metric",
    "window",
    "evidence",
    "runtime",
    "loader",
    "dataset",
    "guard",
    "report",
    "verify",
    "token",
    "checkpoint",
    "comparison",
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _checkpoint_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in _checkpoint_files(root)
    }


def _row_text(row_index: int, *, terms_per_row: int) -> str:
    return " ".join(
        f"{TERMS[(row_index + offset) % len(TERMS)]}-{row_index}-{offset}"
        for offset in range(terms_per_row)
    )


def write_text_fixture(
    output_dir: Path,
    *,
    model_id: str,
    rows: int,
    terms_per_row: int,
    seq_len: int,
    preview_n: int,
    final_n: int,
) -> dict[str, object]:
    if rows < preview_n + final_n:
        raise ValueError("rows must be at least preview_n + final_n")
    if terms_per_row < 1:
        raise ValueError("terms_per_row must be positive")
    if seq_len < 8:
        raise ValueError("seq_len must be at least 8")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tiny_causal_text.jsonl"
    preset_path = output_dir / "preset.yaml"
    summary_path = output_dir / "fixture_summary.json"

    with data_path.open("w", encoding="utf-8") as handle:
        for row_index in range(rows):
            payload = {"text": _row_text(row_index, terms_per_row=terms_per_row)}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    preset_text = f"""model:
  id: "{model_id}"
  adapter: "hf_causal"
  device: "auto"

dataset:
  provider:
    kind: "local_jsonl"
    file: "{data_path}"
    text_field: "text"
    max_samples: {rows}
  split: "validation"
  seq_len: {seq_len}
  stride: {seq_len}
  preview_n: {preview_n}
  final_n: {final_n}
  seed: 43

eval:
  metric:
    kind: "ppl_causal"
  loss:
    type: "causal"

edit:
  name: "noop"
  plan: {{}}

auto:
  enabled: true
  tier: "balanced"
  probes: 0

guards:
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]

output:
  dir: "runs"
  save_model: false
  save_report: true
"""
    preset_path.write_text(preset_text, encoding="utf-8")

    summary: dict[str, object] = {
        "format_version": "compressed-tensors-fixture-v1",
        "data_path": str(data_path),
        "preset_path": str(preset_path),
        "rows": rows,
        "terms_per_row": terms_per_row,
        "seq_len": seq_len,
        "preview_n": preview_n,
        "final_n": final_n,
        "data_sha256": _sha256_file(data_path),
        "preset_sha256": _sha256_file(preset_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _require_model_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from compressed_tensors.compressors import ModelCompressor
        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationConfig,
            QuantizationScheme,
            QuantizationStrategy,
            apply_quantization_config,
        )
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Missing example dependency. Install the compressed-tensors stack in "
            "your example environment, for example: python -m pip install "
            "'invarlock[compressed-tensors]'"
        ) from exc
    return (
        torch,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        QuantizationArgs,
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStrategy,
        apply_quantization_config,
        ModelCompressor,
        load_file,
    )


def _complete_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def _set_weight_qparams(model: Any) -> int:
    quantized_modules = 0
    for module in model.modules():
        scheme = getattr(module, "quantization_scheme", None)
        if not scheme or getattr(module, "weight", None) is None:
            continue
        quantized_modules += 1
        scale = getattr(module, "weight_scale", None)
        zero_point = getattr(module, "weight_zero_point", None)
        if scale is not None:
            max_abs = module.weight.detach().abs().max().clamp(min=1e-8)
            scale.data.fill_(float(max_abs / 127.0))
        if zero_point is not None:
            zero_point.data.zero_()
    return quantized_modules


def _packed_weight_summary(model_dir: Path, *, load_file: Any) -> dict[str, object]:
    packed_keys: list[str] = []
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    for safetensors_file in safetensors_files:
        for key in load_file(safetensors_file).keys():
            if key.endswith("_packed") or key.endswith("_shape"):
                packed_keys.append(key)
    return {
        "packed_tensor_count": len(
            [key for key in packed_keys if key.endswith("_packed")]
        ),
        "packed_metadata_count": len(
            [key for key in packed_keys if key.endswith("_shape")]
        ),
        "sample_keys": sorted(packed_keys)[:12],
    }


def materialize_tiny_compressed_tensors_model(
    baseline_dir: Path,
    subject_dir: Path,
    *,
    tokenizer_source: str,
    allow_network: bool,
    force: bool,
    seed: int,
    hidden_size: int,
    intermediate_size: int,
    max_position_embeddings: int,
) -> dict[str, object]:
    if baseline_dir.exists() or subject_dir.exists():
        if force:
            shutil.rmtree(baseline_dir, ignore_errors=True)
            shutil.rmtree(subject_dir, ignore_errors=True)
        elif _complete_checkpoint(baseline_dir) and _complete_checkpoint(subject_dir):
            return {
                "format_version": "compressed-tensors-model-v1",
                "baseline_path": str(baseline_dir),
                "subject_path": str(subject_dir),
                "reused": True,
                "baseline_files": _checkpoint_hashes(baseline_dir),
                "subject_files": _checkpoint_hashes(subject_dir),
            }
        else:
            raise SystemExit(
                "Model directories already exist but are incomplete. Pass --force "
                "to replace them."
            )

    baseline_dir.mkdir(parents=True, exist_ok=True)
    subject_dir.mkdir(parents=True, exist_ok=True)

    (
        torch,
        auto_tokenizer,
        llama_config,
        llama_model,
        quantization_args,
        quantization_config,
        quantization_scheme,
        quantization_strategy,
        apply_quantization_config,
        model_compressor,
        load_file,
    ) = _require_model_dependencies()

    random.seed(int(seed))
    torch.manual_seed(int(seed))

    tokenizer = auto_tokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=not bool(allow_network),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = llama_config(
        vocab_size=len(tokenizer),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=int(max_position_embeddings),
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    model = llama_model(config).eval()
    model.save_pretrained(baseline_dir, safe_serialization=True)
    tokenizer.save_pretrained(baseline_dir)

    qconfig = quantization_config(
        config_groups={
            "group_0": quantization_scheme(
                targets=["Linear"],
                weights=quantization_args(
                    num_bits=8,
                    strategy=quantization_strategy.TENSOR,
                    symmetric=True,
                ),
            )
        }
    )
    apply_quantization_config(model, qconfig)
    quantized_module_count = _set_weight_qparams(model)
    compressor = model_compressor.from_pretrained_model(
        model,
        quantization_format="pack-quantized",
    )
    compressor.compress_model(model)
    model.save_pretrained(subject_dir, safe_serialization=True)
    tokenizer.save_pretrained(subject_dir)
    compressor.update_config(str(subject_dir))

    packed_summary = _packed_weight_summary(subject_dir, load_file=load_file)
    summary: dict[str, object] = {
        "format_version": "compressed-tensors-model-v1",
        "baseline_path": str(baseline_dir),
        "subject_path": str(subject_dir),
        "reused": False,
        "tokenizer_source": tokenizer_source,
        "seed": int(seed),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "max_position_embeddings": int(max_position_embeddings),
        "quantized_module_count": quantized_module_count,
        "compressed_tensors": {
            "format": "pack-quantized",
            "weights": {
                "num_bits": 8,
                "strategy": "tensor",
                "symmetric": True,
            },
            **packed_summary,
        },
        "baseline_files": _checkpoint_hashes(baseline_dir),
        "subject_files": _checkpoint_hashes(subject_dir),
    }
    (subject_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_adapter_runtime_summary(
    output_dir: Path,
    *,
    subject_adapter: str,
    model_summary: dict[str, object],
) -> dict[str, object]:
    summary = {
        "format_version": "compressed-tensors-adapter-runtime-v1",
        "subject_adapter": subject_adapter,
        "checkpoint_quantization": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "weights": {
                "num_bits": 8,
                "strategy": "tensor",
                "symmetric": True,
            },
        },
        "model": model_summary,
    }
    (output_dir / "adapter_runtime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_checkpoint_refs(
    output_dir: Path,
    *,
    baseline_dir: Path,
    subject_dir: Path,
) -> dict[str, object]:
    refs = {
        "format_version": "compressed-tensors-checkpoint-refs-v1",
        "baseline": str(baseline_dir),
        "subject": str(subject_dir),
        "baseline_adapter": "hf_causal",
        "subject_adapter": "hf_ct",
        "baseline_files": _checkpoint_hashes(baseline_dir),
        "subject_files": _checkpoint_hashes(subject_dir),
    }
    (output_dir / "checkpoint_refs.json").write_text(
        json.dumps(refs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return refs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the tiny dense baseline, compressed-tensors subject checkpoint, "
            "local JSONL dataset, and preset used by the hf_ct integration example."
        )
    )
    parser.add_argument("--baseline-model-dir", required=True, type=Path)
    parser.add_argument("--subject-model-dir", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-source", default="sshleifer/tiny-gpt2")
    parser.add_argument("--rows", type=int, default=860)
    parser.add_argument("--terms-per-row", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--preview-n", type=int, default=400)
    parser.add_argument("--final-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--max-position-embeddings", type=int, default=256)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_dir = args.baseline_model_dir.resolve()
    subject_dir = args.subject_model_dir.resolve()
    fixture_dir = args.fixture_dir.resolve()
    model_summary = materialize_tiny_compressed_tensors_model(
        baseline_dir,
        subject_dir,
        tokenizer_source=str(args.tokenizer_source),
        allow_network=bool(args.allow_network),
        force=bool(args.force),
        seed=int(args.seed),
        hidden_size=int(args.hidden_size),
        intermediate_size=int(args.intermediate_size),
        max_position_embeddings=int(args.max_position_embeddings),
    )
    write_checkpoint_refs(
        subject_dir,
        baseline_dir=baseline_dir,
        subject_dir=subject_dir,
    )
    write_adapter_runtime_summary(
        subject_dir,
        subject_adapter="hf_ct",
        model_summary=model_summary,
    )

    fixture_summary = write_text_fixture(
        fixture_dir,
        model_id=str(baseline_dir),
        rows=args.rows,
        terms_per_row=args.terms_per_row,
        seq_len=args.seq_len,
        preview_n=args.preview_n,
        final_n=args.final_n,
    )
    fixture_summary["model"] = model_summary
    (fixture_dir / "fixture_summary.json").write_text(
        json.dumps(fixture_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(fixture_summary["preset_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
