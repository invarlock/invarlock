from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.verifai2_2026 import make_text_canary


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_read_text_is_lossy(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\xff\xfe\x00")
    s = make_text_canary._read_text(p)
    assert isinstance(s, str)


def test_build_canary_filters_and_truncates_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "subdir").mkdir()

    long_bytes = (b"abc\x00" * 20) + b"TAIL"
    (root / "a.txt").write_bytes(long_bytes)
    (root / "exact.txt").write_text("Z" * 10, encoding="utf-8")
    (root / "short.txt").write_text("tiny", encoding="utf-8")
    (root / "subdir" / "b.txt").write_text("B" * 50, encoding="utf-8")

    # Glob includes directories and a broken symlink.
    os.symlink(str(root / "does_not_exist.txt"), str(root / "broken_link.txt"))
    unreadable = root / "unreadable.txt"
    unreadable.write_text("U" * 50, encoding="utf-8")

    # Root on Linux can ignore chmod bits; force the read failure deterministically.
    orig_read_bytes = Path.read_bytes

    def _patched_read_bytes(self: Path) -> bytes:
        if self == unreadable:
            raise PermissionError("synthetic unreadable")
        return orig_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _patched_read_bytes, raising=True)

    selected1, manifest1 = make_text_canary.build_canary(
        input_dir=root,
        patterns=["**/*"],
        n=2,
        seed=123,
        min_chars=10,
        max_chars=10,
    )
    selected2, manifest2 = make_text_canary.build_canary(
        input_dir=root,
        patterns=["**/*"],
        n=2,
        seed=123,
        min_chars=10,
        max_chars=10,
    )
    assert [it["id"] for it in selected1] == [it["id"] for it in selected2]
    assert manifest1["input"]["candidates"] == manifest2["input"]["candidates"]

    # Truncation + null stripping happened.
    assert all(len(it["text"]) <= 10 for it in selected1)
    assert "\x00" not in selected1[0]["text"]


def test_write_jsonl_hash_matches_file_bytes(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    items = [
        {
            "id": "x",
            "text": "hello",
            "source_sha256": "0" * 64,
            "text_sha256": _sha256_hex(b"hello"),
        }
    ]
    digest = make_text_canary.write_jsonl(items, out)
    assert digest == _sha256_hex(out.read_bytes())


def test_main_no_items_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("hi", encoding="utf-8")

    out = tmp_path / "canary.jsonl"
    manifest = tmp_path / "manifest.json"
    rc = make_text_canary.main(
        [
            "--input-dir",
            str(root),
            "--glob",
            "**/*.txt",
            "--min-chars",
            "1000",
            "--out",
            str(out),
            "--manifest-out",
            str(manifest),
        ]
    )
    assert rc == 2
    assert "No items selected" in capsys.readouterr().err


def test_main_success_default_glob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("A" * 50, encoding="utf-8")
    (root / "b.md").write_text("B" * 50, encoding="utf-8")

    out = tmp_path / "canary.jsonl"
    manifest = tmp_path / "manifest.json"
    rc = make_text_canary.main(
        [
            "--input-dir",
            str(root),
            "--n",
            "1",
            "--seed",
            "0",
            "--min-chars",
            "10",
            "--max-chars",
            "20",
            "--out",
            str(out),
            "--manifest-out",
            str(manifest),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "Wrote canary" in text and "Wrote manifest" in text

    man = json.loads(manifest.read_text(encoding="utf-8"))
    assert man["output"]["items_written"] == 1
    assert out.exists()
