from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release import runtime_release_evidence as evidence

SOURCE_COMMIT = "a" * 40
SOURCE_ARCHIVE_SHA256 = "b" * 64


def _gguf_summary(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "evidence_sha256": "d" * 64,
                "fixture_revision": "e" * 40,
                "format_version": evidence.GGUF_FORMAT,
                "image_digest": "sha256:" + "c" * 64,
                "runs": 2,
                "status": "ok",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_builder_cannot_overwrite_destination_created_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _gguf_summary(tmp_path / "gguf.json")
    output = tmp_path / "asset.tar.gz"
    real_link = evidence.os.link

    def inject_destination(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"concurrent publisher")
        real_link(source, destination)

    monkeypatch.setattr(evidence.os, "link", inject_destination)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="already exists"):
        evidence.build_asset(
            output=output,
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={"llama_cpp": summary},
            behavioral_receipts=[],
        )

    assert output.read_bytes() == b"concurrent publisher"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
