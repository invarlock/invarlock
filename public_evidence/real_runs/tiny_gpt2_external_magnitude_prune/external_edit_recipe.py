#!/usr/bin/env python3
"""Materialize the external BYOE subject checkpoint for this public run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_subject_files(output_dir: Path) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            hashes.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256_file(path),
                }
            )
    return hashes


def materialize_subject(
    *,
    model_id: str,
    output_dir: Path,
    prune_fraction: float,
    local_files_only: bool,
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )

    edited_tensors: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2 or not torch.is_floating_point(parameter):
                continue
            flat = parameter.detach().abs().flatten()
            prune_count = max(1, int(flat.numel() * prune_fraction))
            if prune_count >= flat.numel():
                prune_count = flat.numel() - 1
            if prune_count <= 0:
                continue
            threshold = torch.kthvalue(flat, prune_count).values
            mask = parameter.detach().abs() <= threshold
            changed = int(mask.sum().item())
            parameter.masked_fill_(mask, 0.0)
            edited_tensors.append(
                {
                    "name": name,
                    "changed_values": changed,
                    "total_values": int(parameter.numel()),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "schema": "invarlock.public_evidence.external_edit_recipe.v1",
        "baseline_model_id": model_id,
        "external_edit_type": "magnitude_prune",
        "prune_fraction": prune_fraction,
        "subject_checkpoint_path": output_dir.as_posix(),
        "edited_tensors": edited_tensors,
        "weights_vendored": False,
        "subject_file_sha256": _hash_subject_files(output_dir),
    }
    (output_dir / "external_edit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["subject_file_sha256"] = _hash_subject_files(output_dir)
    (output_dir / "external_edit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument(
        "--output-dir",
        default="/private/tmp/invarlock-byoe-magnitude-prune-subject",
    )
    parser.add_argument("--prune-fraction", type=float, default=0.0025)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.prune_fraction < 1.0:
        raise SystemExit("--prune-fraction must be between 0 and 1")

    summary = materialize_subject(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        prune_fraction=args.prune_fraction,
        local_files_only=not args.allow_network,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
