"""Shared filesystem helpers for repository check scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root_from(path: str | Path) -> Path:
    return Path(path).resolve().parents[2]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_lines(path: Path) -> list[str]:
    return read_text(path).splitlines()


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def path_contains_all(path: Path, snippets: set[str] | list[str] | tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    return all(snippet in text for snippet in snippets)
