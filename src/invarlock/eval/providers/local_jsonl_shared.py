from __future__ import annotations

import json
from collections.abc import Iterator
from glob import glob as _glob
from pathlib import Path
from typing import Any


def resolve_local_jsonl_files(
    *,
    file: str | None = None,
    path: str | None = None,
    data_files: str | list[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    if isinstance(file, str) and file:
        p = Path(file)
        if p.exists() and p.is_file():
            files.append(p)
    if isinstance(path, str) and path:
        p = Path(path)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
    if isinstance(data_files, str) and data_files:
        files.extend(Path(p) for p in _glob(data_files))
    elif isinstance(data_files, list):
        for item in data_files:
            try:
                pp = Path(str(item))
                if pp.exists() and pp.is_file():
                    files.append(pp)
            except (AttributeError, OSError, TypeError, ValueError):
                continue
    seen: set[str] = set()
    uniq: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve().as_posix()
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(file_path)
    return uniq


def iter_local_jsonl_objects(files: list[Path]) -> Iterator[dict[str, Any]]:
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except (OSError, UnicodeDecodeError):
            continue


def load_local_jsonl_texts(
    files: list[Path], *, text_field: str, max_samples: int
) -> list[str]:
    texts: list[str] = []
    for obj in iter_local_jsonl_objects(files):
        value = obj.get(text_field)
        if isinstance(value, str) and value.strip():
            texts.append(value)
            if len(texts) >= max_samples:
                return texts
    return texts


def load_local_jsonl_pairs(
    files: list[Path], *, src_field: str, tgt_field: str, max_samples: int
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for obj in iter_local_jsonl_objects(files):
        src = obj.get(src_field)
        tgt = obj.get(tgt_field)
        if (
            isinstance(src, str)
            and src.strip()
            and isinstance(tgt, str)
            and tgt.strip()
        ):
            pairs.append((src, tgt))
            if len(pairs) >= max_samples:
                return pairs
    return pairs


def local_jsonl_cache_key(
    files: list[Path], *, field_names: tuple[str, ...], max_samples: int
) -> tuple[Any, ...]:
    from invarlock.eval.data_support import _local_files_signature

    return (
        _local_files_signature(files),
        tuple(field_names),
        (int(max_samples),),
    )


__all__ = [
    "iter_local_jsonl_objects",
    "load_local_jsonl_pairs",
    "load_local_jsonl_texts",
    "local_jsonl_cache_key",
    "resolve_local_jsonl_files",
]
