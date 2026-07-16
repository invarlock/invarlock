from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "checks"
    / "sync_packaged_public_evidence.py"
)


def _run(source: Path, packaged: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source),
            "--packaged-root",
            str(packaged),
            mode,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_sync_writes_and_checks_explicit_empty_index(tmp_path: Path) -> None:
    source = tmp_path / "public_evidence"
    source.mkdir()
    packaged = tmp_path / "packaged"

    written = _run(source, packaged, "--write")
    checked = _run(source, packaged, "--check")

    assert written.returncode == checked.returncode == 0
    payload = json.loads((packaged / "evidence_index.json").read_text(encoding="utf-8"))
    assert payload["entries"] == []
    assert payload["status_label"] == "Evidence not yet created"


def test_sync_indexes_only_canonical_pack_and_signed_receipt(tmp_path: Path) -> None:
    source = tmp_path / "public_evidence"
    entry = source / "evidence" / "demo"
    pack = entry / "pack"
    pack.mkdir(parents=True)
    for name in ("manifest.json", "manifest.signature.json", "checksums.sha256"):
        (pack / name).write_text("{}\n", encoding="utf-8")
    (entry / "verification-receipt.json").write_text("{}\n", encoding="utf-8")
    (entry / "evidence.meta.json").write_text(
        json.dumps(
            {
                "format_version": "invarlock/public-evidence-meta-v1",
                "summary": "Verified demo",
                "artifact_paths": {
                    "evidence_pack": "pack",
                    "verification_receipt": "verification-receipt.json",
                },
            }
        ),
        encoding="utf-8",
    )
    packaged = tmp_path / "packaged"

    result = _run(source, packaged, "--write")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((packaged / "evidence_index.json").read_text(encoding="utf-8"))
    assert payload["evidence_count"] == 1
    assert payload["entries"][0]["evidence_class"] == "signed_evidence_pack"
    assert set(payload["entries"][0]["artifacts"]) == {
        "evidence_pack",
        "verification_receipt",
    }
    assert (source / "evidence_index.json").read_bytes() == (
        packaged / "evidence_index.json"
    ).read_bytes()


def test_sync_preserves_external_entries_beside_derived_local_entries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "public_evidence"
    entry = source / "evidence" / "local"
    pack = entry / "pack"
    pack.mkdir(parents=True)
    for name in ("manifest.json", "manifest.signature.json", "checksums.sha256"):
        (pack / name).write_text("{}\n", encoding="utf-8")
    (entry / "verification-receipt.json").write_text("{}\n", encoding="utf-8")
    (entry / "evidence.meta.json").write_text(
        json.dumps(
            {
                "format_version": "invarlock/public-evidence-meta-v1",
                "summary": "Local evidence",
                "artifact_paths": {
                    "evidence_pack": "pack",
                    "verification_receipt": "verification-receipt.json",
                },
            }
        ),
        encoding="utf-8",
    )
    digest = "sha256:" + "a" * 64
    external = {
        "slug": "external",
        "path": "public_evidence/evidence/external",
        "evidence_class": "signed_evidence_pack",
        "summary": "External evidence",
        "artifacts": {
            "evidence_pack": {
                "kind": "directory",
                "path": "public_evidence/evidence/external/evidence",
                "file_count": 3,
                "size_bytes": 30,
                "control_hashes": {"manifest.json": digest},
                "external_asset": {
                    "url": "https://example.com/evidence.tar.zst",
                    "sha256": digest,
                },
            },
            "verification_receipt": {
                "kind": "file",
                "path": "public_evidence/evidence/external/verification.receipt.json",
                "size_bytes": 10,
                "sha256": digest,
                "external_asset": {
                    "url": "https://example.com/verification.receipt.json",
                    "sha256": digest,
                },
            },
        },
    }
    (source / "evidence_index.json").write_text(
        json.dumps(
            {
                "format_version": "invarlock/public-evidence-index-v1",
                "status": "available",
                "status_label": "Evidence available",
                "carrier_policy": {"installed_wheel": "compact_index_only"},
                "evidence_count": 1,
                "evidence_file_count": 4,
                "evidence_size_bytes": 40,
                "entries": [external],
            }
        ),
        encoding="utf-8",
    )
    packaged = tmp_path / "packaged"

    result = _run(source, packaged, "--write")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((source / "evidence_index.json").read_text(encoding="utf-8"))
    assert [entry["slug"] for entry in payload["entries"]] == ["external", "local"]
    assert payload["evidence_file_count"] == 8
    assert payload["evidence_size_bytes"] == 52
    assert (source / "evidence_index.json").read_bytes() == (
        packaged / "evidence_index.json"
    ).read_bytes()


def test_sync_rejects_legacy_metadata_shape(tmp_path: Path) -> None:
    source = tmp_path / "public_evidence"
    entry = source / "evidence" / "demo"
    entry.mkdir(parents=True)
    (entry / "evidence.meta.json").write_text(
        json.dumps({"evidence_class": "historical_archived_fixture"}),
        encoding="utf-8",
    )

    result = _run(source, tmp_path / "packaged", "--write")

    assert result.returncode == 1
    assert "unsupported metadata format" in result.stdout


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"note": "undeclared"}, "metadata fields are not closed"),
        ({"summary": ""}, "summary must be concise plain text"),
        (
            {
                "artifact_paths": {
                    "evidence_pack": "/tmp/other",
                    "verification_receipt": "receipt.json",
                }
            },
            "invalid direct-child evidence_pack path",
        ),
        (
            {
                "artifact_paths": {
                    "evidence_pack": "../other",
                    "verification_receipt": "receipt.json",
                }
            },
            "invalid direct-child evidence_pack path",
        ),
        (
            {
                "artifact_paths": {
                    "evidence_pack": "nested/pack",
                    "verification_receipt": "receipt.json",
                }
            },
            "invalid direct-child evidence_pack path",
        ),
    ],
)
def test_sync_rejects_open_or_escaping_metadata(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "public_evidence"
    entry = source / "evidence" / "demo"
    entry.mkdir(parents=True)
    metadata: dict[str, object] = {
        "format_version": "invarlock/public-evidence-meta-v1",
        "summary": "Verified demo",
        "artifact_paths": {
            "evidence_pack": "pack",
            "verification_receipt": "receipt.json",
        },
    }
    metadata.update(mutation)
    (entry / "evidence.meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = _run(source, tmp_path / "packaged", "--write")

    assert result.returncode == 1
    assert message in result.stdout
