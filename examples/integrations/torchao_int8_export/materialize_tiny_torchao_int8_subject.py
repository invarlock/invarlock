#!/usr/bin/env python3
"""Materialize a tiny torchao int8-export subject checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TERMS = (
    "invarlock",
    "torchao",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic tiny HF baseline, apply torchao int8 "
            "weight-only quantization, and export a dequantized HF-loadable "
            "subject checkpoint."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Directory where the generated baseline checkpoint will be written.",
    )
    parser.add_argument(
        "--subject-dir",
        required=True,
        help="Directory where the exported subject checkpoint will be written.",
    )
    parser.add_argument(
        "--fixture-dir",
        required=True,
        help="Directory where the generated local JSONL fixture will be written.",
    )
    parser.add_argument(
        "--tokenizer-source",
        default="sshleifer/tiny-gpt2",
        help="Tokenizer ID or local path used by the tiny local model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
        help="Seed for deterministic baseline materialization.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Tiny Llama hidden size.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=64,
        help="Tiny Llama MLP intermediate size.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=64,
        help="Tiny Llama context length.",
    )
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
        help="Replace existing baseline and subject directories.",
    )
    return parser.parse_args()


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaConfig,
            LlamaForCausalLM,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Missing example dependency. Install torchao in your example "
            "environment, for example: python -m pip install torchao"
        ) from exc
    return (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        Int8WeightOnlyConfig,
        quantize_,
    )


def _prepare_output_dirs(paths: list[Path], *, force: bool) -> None:
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


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _row_text(row_index: int, *, terms_per_row: int) -> str:
    return " ".join(
        f"{TERMS[(row_index + offset) % len(TERMS)]}-{row_index}-{offset}"
        for offset in range(terms_per_row)
    )


def write_text_fixture(
    output_dir: Path,
    *,
    model_id: str = "local-tiny-llama",
    rows: int,
    terms_per_row: int,
    seq_len: int,
    preview_n: int,
    final_n: int,
) -> dict[str, Any]:
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

    summary: dict[str, Any] = {
        "format_version": "torchao-fixture-v1",
        "data_path": str(data_path),
        "preset_path": str(preset_path),
        "rows": rows,
        "terms_per_row": terms_per_row,
        "seq_len": seq_len,
        "preview_n": preview_n,
        "final_n": final_n,
        "data_sha256": _sha256(data_path),
        "preset_sha256": _sha256(preset_path),
    }
    _write_json(summary_path, summary)
    return summary


def _quantized_tensor_type(value: Any) -> str | None:
    value_type = type(value)
    fqcn = f"{value_type.__module__}.{value_type.__name__}"
    return fqcn if "torchao" in fqcn.lower() else None


def _dense_state_dict(
    quantized_model: Any,
    baseline_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    exported: dict[str, Any] = {}
    changed = 0
    max_abs_delta = 0.0
    quantized_tensors = 0
    quantized_types: set[str] = set()
    by_parameter: dict[str, float] = {}

    for name, value in quantized_model.state_dict().items():
        quantized_type = _quantized_tensor_type(value)
        if quantized_type is not None:
            quantized_tensors += 1
            quantized_types.add(quantized_type)
        dense = value.dequantize() if hasattr(value, "dequantize") else value
        dense = dense.detach().cpu().to(dtype=baseline_state[name].dtype)
        exported[name] = dense
        delta = (dense - baseline_state[name].detach().cpu()).abs()
        parameter_delta = float(delta.max().item())
        by_parameter[name] = parameter_delta
        max_abs_delta = max(max_abs_delta, parameter_delta)
        if parameter_delta > 0.0:
            changed += 1

    return exported, {
        "checked_parameters": len(exported),
        "changed_parameters": changed,
        "max_abs_delta": max_abs_delta,
        "quantized_tensors": quantized_tensors,
        "quantized_tensor_types": sorted(quantized_types),
        "by_parameter": by_parameter,
    }


def _probe_native_quantized_save(
    quantized_model: Any, scratch_parent: Path
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="native-torchao-save-", dir=str(scratch_parent)
    ) as tmp:
        probe_dir = Path(tmp)
        try:
            quantized_model.save_pretrained(probe_dir, safe_serialization=True)
        except Exception as exc:  # noqa: BLE001 - compatibility boundary probe
            return {
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
            }
    return {"ok": True}


def main() -> None:
    args = _parse_args()
    baseline_dir = Path(args.baseline_dir)
    subject_dir = Path(args.subject_dir)
    fixture_dir = Path(args.fixture_dir)
    if baseline_dir.resolve() == subject_dir.resolve():
        raise SystemExit("--baseline-dir and --subject-dir must be different paths.")
    local_files_only = not bool(args.allow_network)
    (
        torch,
        auto_model,
        auto_tokenizer,
        llama_config,
        llama_model,
        int8_config,
        quantize,
    ) = _require_dependencies()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _prepare_output_dirs(
        [baseline_dir, subject_dir, fixture_dir], force=bool(args.force)
    )
    fixture = write_text_fixture(
        fixture_dir,
        model_id=str(baseline_dir),
        rows=int(args.rows),
        terms_per_row=int(args.terms_per_row),
        seq_len=int(args.seq_len),
        preview_n=int(args.preview_n),
        final_n=int(args.final_n),
    )

    tokenizer = auto_tokenizer.from_pretrained(
        args.tokenizer_source,
        local_files_only=local_files_only,
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
    baseline_model.save_pretrained(baseline_dir, safe_serialization=True)
    tokenizer.save_pretrained(baseline_dir)

    baseline_state = {
        name: value.detach().cpu().clone()
        for name, value in baseline_model.state_dict().items()
    }
    quantized_model = auto_model.from_pretrained(baseline_dir).eval()
    quantize(quantized_model, int8_config())
    native_save_probe = _probe_native_quantized_save(
        quantized_model,
        scratch_parent=subject_dir.parent,
    )

    exported_state, delta_summary = _dense_state_dict(quantized_model, baseline_state)
    if int(delta_summary["quantized_tensors"]) <= 0:
        raise SystemExit("torchao did not produce quantized tensor-backed weights.")
    if float(delta_summary["max_abs_delta"]) <= 0.0:
        raise SystemExit("Exported subject checkpoint did not change any weights.")

    export_model = llama_model(config).eval()
    export_model.load_state_dict(exported_state, strict=True)
    export_model.save_pretrained(subject_dir, safe_serialization=True)
    tokenizer.save_pretrained(subject_dir)

    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    baseline_files = {
        str(path.relative_to(baseline_dir)): _sha256(path)
        for path in _checkpoint_files(baseline_dir)
    }
    subject_files = {
        str(path.relative_to(subject_dir)): _sha256(path)
        for path in _checkpoint_files(subject_dir)
    }
    summary = {
        "format_version": "integration-example-edit-summary-v1",
        "created_at": timestamp,
        "baseline_checkpoint_path": str(baseline_dir),
        "subject_checkpoint_path": str(subject_dir),
        "external_edit_type": "torchao_int8_weight_only_export",
        "toolchain": "torchao",
        "toolchain_versions": {
            "torch": _version("torch"),
            "torchao": _version("torchao"),
            "transformers": _version("transformers"),
        },
        "model": {
            "architecture": "tiny-llama-causal-lm",
            "tokenizer_source": str(args.tokenizer_source),
            "hidden_size": int(args.hidden_size),
            "intermediate_size": int(args.intermediate_size),
            "num_hidden_layers": 1,
        },
        "fixture": fixture,
        "torchao": {
            "quantization": "Int8WeightOnlyConfig",
            "export_mode": "dequantized_hf_checkpoint",
            "native_quantized_save_probe": native_save_probe,
        },
        "delta_summary": delta_summary,
        "files": {
            "baseline": baseline_files,
            "subject": subject_files,
        },
    }
    checkpoint_refs = {
        "format_version": "checkpoint-refs-v1",
        "lane_id": "tiny-llama-torchao-int8-export",
        "created_at": timestamp,
        "baseline": {
            "kind": "byoe_checkpoint_ref",
            "path": str(baseline_dir),
            "purpose": "Deterministic tiny HF baseline for the torchao example.",
        },
        "subject": {
            "kind": "byoe_checkpoint_ref",
            "path": str(subject_dir),
            "edit_workflow": "External torchao int8 weight-only quantize/export",
            "external_edit_type": "torchao_int8_weight_only_export",
            "purpose": (
                "HF-loadable checkpoint exported from a torchao-quantized "
                "tiny model after dequantizing tensor-subclass weights."
            ),
        },
        "artifacts": {
            "external_edit_summary": "external_edit_summary.json",
            "files": {
                "baseline": baseline_files,
                "subject": subject_files,
            },
        },
    }
    _write_json(subject_dir / "external_edit_summary.json", summary)
    _write_json(subject_dir / "checkpoint_refs.json", checkpoint_refs)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
