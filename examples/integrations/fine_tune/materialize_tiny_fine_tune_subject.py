#!/usr/bin/env python3
"""Materialize a tiny fine-tuned subject checkpoint for integration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMS = (
    "invarlock",
    "fine-tune",
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
            "Run a deterministic one-step tiny fine-tune, save the subject "
            "checkpoint, and write fixture/provenance sidecars."
        )
    )
    parser.add_argument(
        "--baseline",
        default="sshleifer/tiny-gpt2",
        help="Baseline model ID or local path. Default: sshleifer/tiny-gpt2.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the fine-tuned subject checkpoint will be written.",
    )
    parser.add_argument(
        "--fixture-dir",
        required=True,
        help="Directory where the generated local JSONL fixture will be written.",
    )
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--rows", type=int, default=860)
    parser.add_argument("--terms-per-row", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--preview-n", type=int, default=400)
    parser.add_argument("--final-n", type=int, default=400)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face downloads instead of using local cache only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def _disable_torchvision_for_text_only_transformers() -> None:
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    import_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False


def _require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Missing example dependency. Install the Hugging Face stack in your "
            "example environment, for example: python -m pip install "
            "'invarlock[hf]'"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_tree(root: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in _checkpoint_files(root):
        rel = file_path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    model_id: str = "sshleifer/tiny-gpt2",
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
  seed: 20260627

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
"""
    preset_path.write_text(preset_text, encoding="utf-8")

    summary = {
        "format_version": "tiny-fine-tune-fixture-v1",
        "model_id": model_id,
        "rows": rows,
        "terms_per_row": terms_per_row,
        "seq_len": seq_len,
        "preview_n": preview_n,
        "final_n": final_n,
        "data_path": str(data_path),
        "preset_path": str(preset_path),
        "data_sha256": _sha256(data_path),
        "preset_sha256": _sha256(preset_path),
    }
    _write_json(summary_path, summary)
    return summary


def _selected_floating_state(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }


def _delta_summary(before: dict[str, Any], model: Any) -> dict[str, Any]:
    max_abs_delta = 0.0
    changed_tensors = 0
    checked_tensors = 0
    by_tensor: dict[str, float] = {}
    for name, tensor in model.state_dict().items():
        before_value = before.get(name)
        if before_value is None or not tensor.is_floating_point():
            continue
        checked_tensors += 1
        delta = (tensor.detach().cpu().to(before_value.dtype) - before_value).abs()
        tensor_delta = float(delta.max().item())
        by_tensor[name] = tensor_delta
        max_abs_delta = max(max_abs_delta, tensor_delta)
        if tensor_delta > 0.0:
            changed_tensors += 1
    return {
        "checked_tensors": checked_tensors,
        "changed_tensors": changed_tensors,
        "max_abs_delta": max_abs_delta,
        "by_tensor": by_tensor,
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer = _require_dependencies()
    _disable_torchvision_for_text_only_transformers()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    local_files_only = not args.allow_network

    tokenizer = AutoTokenizer.from_pretrained(
        args.baseline, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.baseline, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(args.output_dir)
    fixture_dir = Path(args.fixture_dir)
    _prepare_output_dir(output_dir, force=args.force)
    fixture_summary = write_text_fixture(
        fixture_dir,
        model_id=args.baseline,
        rows=args.rows,
        terms_per_row=args.terms_per_row,
        seq_len=args.seq_len,
        preview_n=args.preview_n,
        final_n=args.final_n,
    )

    before = _selected_floating_state(model)
    train_texts = [
        "InvarLock fine tune integration alpha.",
        "InvarLock fine tune integration beta.",
        "Evidence metadata remains descriptive.",
        "Tiny training creates a real subject checkpoint.",
    ]
    encoded = tokenizer(
        train_texts,
        max_length=32,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**encoded, labels=labels).loss
    loss.backward()
    optimizer.step()
    model.eval()

    delta = _delta_summary(before, model)
    if int(delta["changed_tensors"]) == 0:
        raise SystemExit("fine-tune materialization did not change any tensors")

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    checkpoint_entries = []
    for file_path in _checkpoint_files(output_dir):
        checkpoint_entries.append(
            {
                "path": file_path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(file_path),
                "bytes": file_path.stat().st_size,
            }
        )

    subject_digest = _sha256_tree(output_dir)
    checkpoint_refs = {
        "schema": "invarlock.integration.checkpoint_refs.v1",
        "baseline": {
            "type": "hf_model_id_or_path",
            "value": args.baseline,
        },
        "subject": {
            "type": "local_hf_checkpoint",
            "path": str(output_dir),
            "tree_digest": subject_digest,
            "files": checkpoint_entries,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    external_edit_summary = {
        "format_version": "tiny-fine-tune-subject-v1",
        "external_edit_type": "fine_tune",
        "model_id": args.baseline,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "loss": float(loss.detach().cpu().item()),
        "changed_tensors": delta["changed_tensors"],
        "checked_tensors": delta["checked_tensors"],
        "max_abs_delta": delta["max_abs_delta"],
        "fixture_summary": fixture_summary,
        "subject_digest": subject_digest,
        "notes": (
            "Tiny deterministic one-step fine-tune used to materialize a real "
            "HF-loadable subject checkpoint for baseline-vs-subject evidence."
        ),
    }
    _write_json(output_dir / "checkpoint_refs.json", checkpoint_refs)
    _write_json(output_dir / "external_edit_summary.json", external_edit_summary)
    return {
        "checkpoint_refs": checkpoint_refs,
        "external_edit_summary": external_edit_summary,
        "fixture_summary": fixture_summary,
    }


def main() -> int:
    args = _parse_args()
    result = _materialize(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
