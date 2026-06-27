#!/usr/bin/env python3
"""Materialize a tiny GPTQModel AWQ subject checkpoint and local eval fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TERMS = (
    "invarlock",
    "awq",
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _row_text(row_index: int, *, terms_per_row: int) -> str:
    return " ".join(
        f"{TERMS[(row_index + offset) % len(TERMS)]}-{row_index}-{offset}"
        for offset in range(terms_per_row)
    )


def write_text_fixture(
    output_dir: Path,
    *,
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
  id: "local-tiny-llama"
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
  seed: 47

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
        "format_version": "awq-fixture-v1",
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
    _write_json(summary_path, summary)
    return summary


def _prepare_output_dirs(paths: Sequence[Path], *, force: bool) -> None:
    for path in paths:
        if path.exists():
            if not force:
                raise SystemExit(
                    f"Output directory already exists: {path}. "
                    "Pass --force to replace it."
                )
            shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch

        from invarlock.plugins import _patch_gptqmodel_transformers_hub_compat

        _patch_gptqmodel_transformers_hub_compat()
        from gptqmodel import AWQConfig, GPTQModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaConfig,
            LlamaForCausalLM,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Missing example dependency. Install GPTQModel in your example "
            "environment, for example: python -m pip install 'invarlock[awq]'"
        ) from exc
    return (
        torch,
        GPTQModel,
        AWQConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )


def _calibration_samples(
    tokenizer: Any,
    *,
    rows: int,
    terms_per_row: int,
    seq_len: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row_index in range(rows):
        encoded = tokenizer(
            _row_text(row_index, terms_per_row=terms_per_row),
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=False,
        )
        samples.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return samples


def _checkpoint_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in _checkpoint_files(root)
    }


def _quantization_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    quant_cfg = getattr(config, "quantization_config", None)
    if isinstance(quant_cfg, dict):
        return dict(quant_cfg)
    if quant_cfg is None:
        return {}
    to_dict = getattr(quant_cfg, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, dict) else {}
    return {
        key: value
        for key, value in vars(quant_cfg).items()
        if isinstance(value, str | int | float | bool | list | dict | type(None))
    }


def _pin_awq_backend(config_path: Path, *, backend: str) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    quant_cfg = payload.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        raise SystemExit("AWQ subject config did not include quantization_config.")

    quant_cfg["backend"] = backend
    _write_json(config_path, payload)
    return dict(quant_cfg)


def _quantized_module_types(model: Any) -> list[str]:
    module_types: set[str] = set()
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return []
    for _, module in named_modules():
        module_type = type(module)
        fqcn = f"{module_type.__module__}.{module_type.__name__}"
        normalized = fqcn.lower()
        if "awq" in normalized or "gptqmodel" in normalized or "qlinear" in normalized:
            module_types.add(fqcn)
    return sorted(module_types)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic tiny HF baseline, quantize it with "
            "GPTQModel AWQ, and write a local evaluation fixture."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        type=Path,
        help="Directory where the generated baseline checkpoint will be written.",
    )
    parser.add_argument(
        "--subject-dir",
        required=True,
        type=Path,
        help="Directory where the AWQ subject checkpoint will be written.",
    )
    parser.add_argument(
        "--fixture-dir",
        required=True,
        type=Path,
        help="Directory where the generated local JSONL fixture will be written.",
    )
    parser.add_argument(
        "--tokenizer-source",
        default="sshleifer/tiny-gpt2",
        help="Tokenizer ID or local path used by the tiny local model.",
    )
    parser.add_argument(
        "--quantize-device",
        default="cuda:0",
        help="CUDA device used for AWQ materialization. Default: cuda:0.",
    )
    parser.add_argument(
        "--awq-backend",
        default="torch_awq",
        help=(
            "GPTQModel/Transformers AWQ backend to use for this tiny example. "
            "Default: torch_awq."
        ),
    )
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--max-position-embeddings", type=int, default=256)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--calibration-rows", type=int, default=8)
    parser.add_argument("--calibration-terms-per-row", type=int, default=80)
    parser.add_argument("--rows", type=int, default=860)
    parser.add_argument("--terms-per-row", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--preview-n", type=int, default=400)
    parser.add_argument("--final-n", type=int, default=400)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow tokenizer downloads instead of using local cache only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing baseline, subject, and fixture directories.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.baseline_dir.resolve() == args.subject_dir.resolve():
        raise SystemExit("--baseline-dir and --subject-dir must be different paths.")

    (
        torch,
        gptq_model,
        awq_config,
        auto_model,
        auto_tokenizer,
        llama_config,
        llama_model,
    ) = _require_dependencies()

    if not torch.cuda.is_available():
        raise SystemExit(
            "AWQ materialization requires CUDA. Run this example on a CUDA host."
        )

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    _prepare_output_dirs(
        [args.baseline_dir, args.subject_dir, args.fixture_dir],
        force=bool(args.force),
    )

    fixture = write_text_fixture(
        args.fixture_dir,
        rows=int(args.rows),
        terms_per_row=int(args.terms_per_row),
        seq_len=int(args.seq_len),
        preview_n=int(args.preview_n),
        final_n=int(args.final_n),
    )

    tokenizer = auto_tokenizer.from_pretrained(
        args.tokenizer_source,
        local_files_only=not bool(args.allow_network),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = llama_config(
        vocab_size=len(tokenizer),
        hidden_size=int(args.hidden_size),
        intermediate_size=int(args.intermediate_size),
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=int(args.max_position_embeddings),
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    baseline_model = llama_model(config).eval()
    baseline_model.save_pretrained(args.baseline_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.baseline_dir)

    calibration = _calibration_samples(
        tokenizer,
        rows=int(args.calibration_rows),
        terms_per_row=int(args.calibration_terms_per_row),
        seq_len=min(int(args.seq_len), int(args.max_position_embeddings)),
    )
    qcfg = awq_config(
        bits=int(args.bits),
        group_size=int(args.group_size),
        device=str(args.quantize_device),
    )
    model = gptq_model.load(str(args.baseline_dir), qcfg)
    model.quantize(calibration, backend=str(args.awq_backend))
    model.save(str(args.subject_dir))
    tokenizer.save_pretrained(args.subject_dir)
    saved_quant_cfg = _pin_awq_backend(
        args.subject_dir / "config.json",
        backend=str(args.awq_backend),
    )

    reloaded = auto_model.from_pretrained(
        str(args.subject_dir),
        trust_remote_code=False,
        device_map="auto",
    )
    quant_cfg = _quantization_config(reloaded)
    if not quant_cfg:
        raise SystemExit("AWQ subject did not expose a quantization config.")

    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    baseline_files = _checkpoint_hashes(args.baseline_dir)
    subject_files = _checkpoint_hashes(args.subject_dir)
    summary = {
        "format_version": "integration-example-edit-summary-v1",
        "created_at": timestamp,
        "baseline_checkpoint_path": str(args.baseline_dir),
        "subject_checkpoint_path": str(args.subject_dir),
        "external_edit_type": "gptqmodel_awq_4bit",
        "toolchain": "gptqmodel",
        "toolchain_versions": {
            "gptqmodel": _version("gptqmodel"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
        "model": {
            "architecture": "tiny-llama-causal-lm",
            "tokenizer_source": str(args.tokenizer_source),
            "hidden_size": int(args.hidden_size),
            "intermediate_size": int(args.intermediate_size),
            "num_hidden_layers": 1,
        },
        "awq": {
            "bits": int(args.bits),
            "group_size": int(args.group_size),
            "backend": str(args.awq_backend),
            "quantize_device": str(args.quantize_device),
            "saved_quantization_config": saved_quant_cfg,
            "quantization_config": quant_cfg,
            "quantized_module_types": _quantized_module_types(reloaded),
        },
        "fixture": fixture,
        "files": {
            "baseline": baseline_files,
            "subject": subject_files,
        },
    }
    checkpoint_refs = {
        "format_version": "checkpoint-refs-v1",
        "lane_id": "tiny-llama-gptqmodel-awq-4bit",
        "created_at": timestamp,
        "baseline": {
            "kind": "byoe_checkpoint_ref",
            "path": str(args.baseline_dir),
            "purpose": "Deterministic tiny HF baseline for the AWQ example.",
        },
        "subject": {
            "kind": "byoe_checkpoint_ref",
            "path": str(args.subject_dir),
            "edit_workflow": "External GPTQModel AWQ quantization",
            "external_edit_type": "gptqmodel_awq_4bit",
            "purpose": "GPTQModel AWQ subject checkpoint loaded by hf_awq.",
        },
        "artifacts": {
            "external_edit_summary": "external_edit_summary.json",
            "fixture_summary": "fixture_summary.json",
            "files": {
                "baseline": baseline_files,
                "subject": subject_files,
            },
        },
    }
    _write_json(args.subject_dir / "external_edit_summary.json", summary)
    _write_json(args.subject_dir / "checkpoint_refs.json", checkpoint_refs)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
