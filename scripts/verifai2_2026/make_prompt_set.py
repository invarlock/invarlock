#!/usr/bin/env python3
"""
make_prompt_set.py
==================

Build a verifier-trace prompt_set block (trace_contract.prompt_set) from a JSONL
file containing prompts/tasks.

This is intentionally simple and release-safe:
- Supports `mode=hash_only` (default): stores only per-item sha256 + dataset refs.
- Supports `mode=embedded`: embeds prompt text (only if redistributable).

The prompt-set digest is computed per research/verifai2_2026/verifier_trace_contract.md
and is stable across embedded vs hash-only mode (it never hashes embedded text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    # Contract canonicalization: sorted keys, no whitespace, UTF-8.
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}")
            out.append(obj)
    return out


def _format_prompt(template: str, record: dict[str, Any], *, text_field: str) -> str:
    if text_field not in record:
        raise KeyError(
            f"Missing text field {text_field!r} in record id={record.get('id')!r}"
        )
    mapping = dict(record)
    mapping.setdefault("text", record.get(text_field))
    try:
        return str(template).format_map(mapping)
    except KeyError as exc:
        raise KeyError(
            f"Template references missing key {exc} for record id={record.get('id')!r}"
        ) from exc


def _compute_prompt_set_digest(
    dataset: dict[str, Any], items: list[dict[str, Any]]
) -> str:
    # Contract: digest hashes only dataset identifiers (name/config/split/revision)
    # plus the ordered (id, sha256) item list. It must be stable across other
    # dataset metadata such as manifest_sha256 or selection_script hashes.
    dataset_id: dict[str, Any] = {}
    for k in ("name", "config", "split", "revision"):
        if k in dataset:
            dataset_id[k] = dataset[k]
    payload = {
        "dataset": dataset_id,
        "items": [{"id": it["id"], "sha256": it["sha256"]} for it in items],
    }
    return _sha256_hex(_canonical_json_bytes(payload))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="in_path", type=Path, required=True, help="Input JSONL."
    )
    p.add_argument(
        "--id-field",
        type=str,
        default="id",
        help="Field name containing a stable task id (default: id).",
    )
    p.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing the prompt text (default: text).",
    )
    p.add_argument(
        "--template",
        type=str,
        default="{text}",
        help="Prompt template applied before hashing; use {text} or other record keys.",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["hash_only", "embedded"],
        default="hash_only",
        help="Whether to embed prompt text (only if redistributable).",
    )
    p.add_argument("--dataset-name", type=str, default="local")
    p.add_argument("--dataset-config", type=str, default="")
    p.add_argument("--dataset-split", type=str, default="test")
    p.add_argument(
        "--dataset-revision",
        type=str,
        default="",
        help="Dataset revision; default is sha256 of the input file bytes.",
    )
    p.add_argument(
        "--dataset-manifest-sha256",
        type=str,
        default="",
        help="Optional manifest sha256 for local datasets.",
    )
    p.add_argument(
        "--selection-script",
        type=Path,
        help="Optional path to a selection script to hash and record.",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="If >0, keep only the first N records."
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Output JSON file for prompt_set."
    )
    args = p.parse_args(argv)

    records = _read_jsonl(args.in_path)
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]
    if not records:
        print("No records found.", file=sys.stderr)
        return 2

    dataset_revision = args.dataset_revision.strip()
    if not dataset_revision:
        dataset_revision = _sha256_hex(args.in_path.read_bytes())

    dataset: dict[str, Any] = {
        "name": args.dataset_name,
        "split": args.dataset_split,
        "revision": dataset_revision,
    }
    if args.dataset_config:
        dataset["config"] = args.dataset_config
    if args.dataset_manifest_sha256:
        dataset["manifest_sha256"] = args.dataset_manifest_sha256

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for rec in records:
        raw_id = rec.get(args.id_field)
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError(
                f"Missing/invalid id field {args.id_field!r} in record: {rec!r}"
            )
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate id: {raw_id}")
        seen_ids.add(raw_id)

        prompt = _format_prompt(args.template, rec, text_field=args.text_field)
        prompt_sha = _sha256_hex(prompt.encode("utf-8"))
        item: dict[str, Any] = {"id": raw_id, "sha256": prompt_sha}
        if args.mode == "embedded":
            item["text"] = prompt
        items.append(item)

    digest = _compute_prompt_set_digest(dataset, items)
    prompt_set: dict[str, Any] = {
        "mode": args.mode,
        "dataset": dataset,
        "items": items,
        "digest_sha256": digest,
    }
    if args.selection_script is not None:
        prompt_set["selection_script_sha256"] = _sha256_hex(
            args.selection_script.read_bytes()
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(prompt_set, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
