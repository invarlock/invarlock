#!/usr/bin/env python3
"""Prepare a deterministic local dataset and preset for the hf_bnb example."""

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
    "policy",
    "profile",
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


def write_fixture(
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
        "format_version": "hf-bnb-fixture-v1",
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


def _require_model_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Missing example dependency. Install the Hugging Face stack in your "
            "example environment, for example: python -m pip install 'invarlock[hf]'"
        ) from exc
    return torch, AutoTokenizer, LlamaConfig, LlamaForCausalLM


def materialize_tiny_llama_model(
    output_dir: Path,
    *,
    tokenizer_source: str,
    allow_network: bool,
    force: bool,
    seed: int,
    hidden_size: int,
    intermediate_size: int,
    max_position_embeddings: int,
) -> dict[str, object]:
    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        elif (output_dir / "config.json").is_file() and (
            output_dir / "model.safetensors"
        ).is_file():
            return {
                "format_version": "hf-bnb-model-v1",
                "model_path": str(output_dir),
                "reused": True,
                "files": _checkpoint_hashes(output_dir),
            }
        else:
            raise SystemExit(
                f"Output directory already exists but is incomplete: {output_dir}. "
                "Pass --force to replace it."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, auto_tokenizer, llama_config, llama_model = _require_model_dependencies()
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
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary: dict[str, object] = {
        "format_version": "hf-bnb-model-v1",
        "model_path": str(output_dir),
        "reused": False,
        "tokenizer_source": tokenizer_source,
        "seed": int(seed),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "max_position_embeddings": int(max_position_embeddings),
        "files": _checkpoint_hashes(output_dir),
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the local JSONL dataset and preset used by the tiny hf_bnb "
            "integration example."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-id", default="local-tiny-llama")
    parser.add_argument("--tokenizer-source", default="sshleifer/tiny-gpt2")
    parser.add_argument("--rows", type=int, default=860)
    parser.add_argument("--terms-per-row", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--preview-n", type=int, default=400)
    parser.add_argument("--final-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--max-position-embeddings", type=int, default=256)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_summary: dict[str, object] | None = None
    model_id = str(args.model_id)
    if args.model_dir is not None:
        model_summary = materialize_tiny_llama_model(
            args.model_dir.resolve(),
            tokenizer_source=str(args.tokenizer_source),
            allow_network=bool(args.allow_network),
            force=bool(args.force),
            seed=int(args.seed),
            hidden_size=int(args.hidden_size),
            intermediate_size=int(args.intermediate_size),
            max_position_embeddings=int(args.max_position_embeddings),
        )
        model_id = str(args.model_dir.resolve())

    summary = write_fixture(
        args.output_dir.resolve(),
        model_id=model_id,
        rows=args.rows,
        terms_per_row=args.terms_per_row,
        seq_len=args.seq_len,
        preview_n=args.preview_n,
        final_n=args.final_n,
    )
    if model_summary is not None:
        summary["model"] = model_summary
        (args.output_dir.resolve() / "fixture_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(summary["preset_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
