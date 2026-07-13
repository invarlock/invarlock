#!/usr/bin/env python3
"""Create deterministic local evaluation data for training integrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TERMS = (
    "invarlock",
    "training",
    "baseline",
    "subject",
    "regression",
    "metric",
    "window",
    "evidence",
    "runtime",
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


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _row_text(row_index: int, *, terms_per_row: int) -> str:
    return " ".join(
        f"{TERMS[(row_index + offset) % len(TERMS)]}-{row_index}-{offset}"
        for offset in range(terms_per_row)
    )


def write_text_fixture(
    output_dir: Path,
    *,
    model_id: str,
    format_version: str,
    rows: int,
    terms_per_row: int,
    seq_len: int,
    preview_n: int,
    final_n: int,
    seed: int = 43,
) -> dict[str, Any]:
    """Write a deterministic JSONL dataset, preset, and hashed summary."""

    if rows < preview_n + final_n:
        raise ValueError("rows must be at least preview_n + final_n")
    if terms_per_row < 1:
        raise ValueError("terms_per_row must be positive")
    if seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    if not format_version.strip():
        raise ValueError("format_version must be non-empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tiny_causal_text.jsonl"
    preset_path = output_dir / "preset.yaml"
    summary_path = output_dir / "fixture_summary.json"

    with data_path.open("w", encoding="utf-8") as handle:
        for row_index in range(rows):
            payload = {"text": _row_text(row_index, terms_per_row=terms_per_row)}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    preset_text = f'''model:
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
  seed: {seed}

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
'''
    preset_path.write_text(preset_text, encoding="utf-8")

    summary: dict[str, Any] = {
        "format_version": format_version,
        "model_id": model_id,
        "rows": rows,
        "terms_per_row": terms_per_row,
        "seq_len": seq_len,
        "preview_n": preview_n,
        "final_n": final_n,
        "seed": seed,
        "data_path": str(data_path),
        "preset_path": str(preset_path),
        "data_sha256": _sha256(data_path),
        "preset_sha256": _sha256(preset_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the local evaluation fixture for a training integration."
    )
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--format-version", required=True)
    parser.add_argument("--rows", type=int, default=860)
    parser.add_argument("--terms-per-row", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--preview-n", type=int, default=400)
    parser.add_argument("--final-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = write_text_fixture(
        Path(args.fixture_dir),
        model_id=args.model_id,
        format_version=args.format_version,
        rows=args.rows,
        terms_per_row=args.terms_per_row,
        seq_len=args.seq_len,
        preview_n=args.preview_n,
        final_n=args.final_n,
        seed=args.seed,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
