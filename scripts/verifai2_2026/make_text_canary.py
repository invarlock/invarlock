#!/usr/bin/env python3
"""
make_text_canary.py
===================

Create a deterministic `local_jsonl` canary file for InvarLock BYOD workflows.

Outputs:
- JSONL where each line contains at least {"text": "..."}.
- A manifest JSON capturing selection rules and hashes for release-safe sharing.

Notes:
- The manifest is designed so you can publish the manifest + extraction script
  without necessarily redistributing the raw canary text (licensing dependent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    # Lossy decode is OK for a canary corpus; record source hash separately.
    return raw.decode("utf-8", errors="replace")


def _gather_files(root: Path, patterns: list[str]) -> list[Path]:
    out: set[Path] = set()
    for pat in patterns:
        out.update(root.glob(pat))
    return sorted(out)


def build_canary(
    *,
    input_dir: Path,
    patterns: list[str],
    n: int,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = _gather_files(input_dir, patterns)
    candidates: list[dict[str, Any]] = []
    for p in files:
        if not p.is_file():
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        src_sha = _sha256_hex(raw)
        text = raw.decode("utf-8", errors="replace")
        text = text.replace("\x00", "")
        text = text.strip()
        if len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        rel = str(p.relative_to(input_dir))
        candidates.append(
            {
                "id": rel,
                "source_path": rel,
                "source_sha256": src_sha,
                "text": text,
                "text_sha256": _sha256_hex(text.encode("utf-8")),
                "n_chars": len(text),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    manifest = {
        "schema_version": "text_canary_manifest.v1",
        "created_at": _utc_now_iso(),
        "input": {
            "kind": "directory",
            "root": str(input_dir),
            "patterns": patterns,
            "candidates": len(candidates),
        },
        "selection": {
            "n": n,
            "seed": seed,
            "min_chars": min_chars,
            "max_chars": max_chars,
        },
        "items": [
            {
                "id": it["id"],
                "source_sha256": it["source_sha256"],
                "text_sha256": it["text_sha256"],
                "n_chars": it["n_chars"],
            }
            for it in selected
        ],
    }
    return selected, manifest


def write_jsonl(items: list[dict[str, Any]], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            # Keep at least "text" for local_jsonl provider; extra fields are OK.
            rec = {
                "text": it["text"],
                "id": it["id"],
                "source_sha256": it["source_sha256"],
                "text_sha256": it["text_sha256"],
            }
            line = json.dumps(rec, ensure_ascii=True) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern(s) relative to input-dir (repeatable). Example: '**/*.py'",
    )
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path, required=True)
    args = p.parse_args(argv)

    patterns = args.glob or ["**/*.txt"]
    items, manifest = build_canary(
        input_dir=args.input_dir,
        patterns=patterns,
        n=args.n,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    if not items:
        print("No items selected; check --glob/--min-chars.", file=sys.stderr)
        return 2

    out_sha = write_jsonl(items, args.out)
    manifest["output"] = {
        "jsonl_path": str(args.out),
        "jsonl_sha256": out_sha,
        "items_written": len(items),
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote canary: {args.out} ({len(items)} items)")
    print(f"Wrote manifest: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
