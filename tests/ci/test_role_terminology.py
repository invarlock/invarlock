from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("docs", "examples", "src", "tests")
SCANNED_FILES = ("CHANGELOG.md", "Makefile", "README.md", "SECURITY.md", "SUPPORT.md")
TEXT_SUFFIXES = {
    ".cue",
    ".json",
    ".md",
    ".pem",
    ".py",
    ".rego",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def test_release_tree_uses_explicit_trust_and_evaluation_roles() -> None:
    removed_role = b"pro" + b"ducer"
    word = re.compile(
        rb"(?<![A-Za-z0-9_])" + removed_role + rb"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    encoded = base64.b64encode(removed_role).rstrip(b"=")
    files = [ROOT / name for name in SCANNED_FILES]
    files.extend(
        path
        for root in SCANNED_ROOTS
        for path in (ROOT / root).rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )
    findings = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if word.search(path.read_bytes()) or encoded in path.read_bytes()
    ]
    names = [
        path.relative_to(ROOT).as_posix()
        for root in SCANNED_ROOTS
        for path in (ROOT / root).rglob("*")
        if path.is_file() and word.search(path.name.encode())
    ]

    assert findings == []
    assert names == []
