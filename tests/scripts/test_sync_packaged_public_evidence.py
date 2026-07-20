from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "checks"
    / "sync_packaged_public_evidence.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sync_packaged_public_evidence_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_index(*, entries: list[object] | None = None) -> dict[str, object]:
    selected = [] if entries is None else entries
    return {
        "format_version": "invarlock/public-evidence-index-v1",
        "status": "not_created" if not selected else "available",
        "status_label": (
            "Evidence not yet created" if not selected else "Evidence available"
        ),
        "carrier_policy": {"installed_wheel": "compact_index_only"},
        "evidence_count": len(selected),
        "evidence_file_count": 0,
        "evidence_size_bytes": 0,
        "entries": selected,
    }


def _external_entry(slug: str = "external") -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    return {
        "slug": slug,
        "path": f"public_evidence/evidence/{slug}",
        "evidence_class": "signed_evidence_pack",
        "summary": "External evidence",
        "artifacts": {
            "evidence_pack": {
                "kind": "directory",
                "path": f"public_evidence/evidence/{slug}/evidence",
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
                "path": f"public_evidence/evidence/{slug}/verification.receipt.json",
                "size_bytes": 10,
                "sha256": digest,
                "external_asset": {
                    "url": "https://example.com/verification.receipt.json",
                    "sha256": digest,
                },
            },
        },
    }


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


def test_read_object_rejects_non_object_json(tmp_path: Path) -> None:
    module = _load()
    payload = tmp_path / "array.json"
    payload.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        module._read_object(payload)


def test_regular_files_rejects_symlinked_content(tmp_path: Path) -> None:
    module = _load()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    tree = tmp_path / "pack"
    tree.mkdir()
    (tree / "linked.json").symlink_to(outside)

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        module._regular_files(tree)


def test_artifact_summary_rejects_missing_or_unsafe_artifact(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    root.mkdir()

    with pytest.raises(ValueError, match="missing or unsafe"):
        module._artifact_summary(root / "missing", source_root=root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"format_version": "future"}, "unsupported public-evidence index format"),
        ({"entries": {}}, "entries must be a list"),
        ({"evidence_count": 1}, "evidence_count must match entries"),
        ({"evidence_file_count": True}, "must be a non-negative integer"),
        ({"evidence_size_bytes": -1}, "must be a non-negative integer"),
        (
            {"status": "available", "status_label": "Evidence available"},
            "empty index must say Evidence not yet created",
        ),
    ],
)
def test_validate_index_rejects_inconsistent_top_level_state(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    module = _load()
    value = _valid_index()
    value.update(mutation)

    with pytest.raises(ValueError, match=message):
        module._validate_index(tmp_path / "evidence_index.json", value)


@pytest.mark.parametrize(
    "entries",
    [
        ["not-an-object"],
        [{"slug": ""}],
        [{"slug": "duplicate"}, {"slug": "duplicate"}],
    ],
)
def test_validate_index_rejects_non_unique_or_invalid_slugs(
    tmp_path: Path, entries: list[object]
) -> None:
    module = _load()
    value = _valid_index(entries=entries)

    with pytest.raises(ValueError, match="entries must have unique slugs"):
        module._validate_index(tmp_path / "evidence_index.json", value)


def test_validate_index_rejects_nonempty_not_created_status(tmp_path: Path) -> None:
    module = _load()
    value = _valid_index(entries=[{"slug": "external"}])
    value.update(status="not_created", status_label="Evidence not yet created")

    with pytest.raises(
        ValueError, match="non-empty index must say evidence is available"
    ):
        module._validate_index(tmp_path / "evidence_index.json", value)


def test_external_entries_replace_a_local_slug(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "public_evidence"
    source.mkdir()
    entries = [_external_entry("replace-me"), _external_entry("keep-me")]
    (source / "evidence_index.json").write_text(
        json.dumps(_valid_index(entries=entries)), encoding="utf-8"
    )

    preserved = module._external_entries(source, replacing_slugs={"replace-me"})

    assert [entry["slug"] for entry in preserved] == ["keep-me"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"artifacts": []}, "has invalid artifacts"),
        (
            {
                "artifacts": {
                    "evidence_pack": {},
                    "verification_receipt": {},
                }
            },
            "must name an external_asset",
        ),
    ],
)
def test_external_entries_reject_malformed_carriers(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    module = _load()
    source = tmp_path / "public_evidence"
    source.mkdir()
    entry = _external_entry()
    entry.update(mutation)
    (source / "evidence_index.json").write_text(
        json.dumps(_valid_index(entries=[entry])), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        module._external_entries(source, replacing_slugs=set())


def test_external_entries_reject_local_bytes_without_metadata(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "public_evidence"
    source.mkdir()
    entry = _external_entry()
    artifacts = entry["artifacts"]
    assert isinstance(artifacts, dict)
    artifact = artifacts["verification_receipt"]
    assert isinstance(artifact, dict)
    logical_path = artifact["path"]
    assert isinstance(logical_path, str)
    local = source.parent / logical_path
    local.parent.mkdir(parents=True)
    local.write_text("{}\n", encoding="utf-8")
    (source / "evidence_index.json").write_text(
        json.dumps(_valid_index(entries=[entry])), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="local artifact.*requires evidence.meta.json"):
        module._external_entries(source, replacing_slugs=set())


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"artifacts": []}, "entry artifacts must be an object"),
        ({"artifacts": {"pack": []}}, "artifact summary must be an object"),
        (
            {"artifacts": {"pack": {"kind": "file", "size_bytes": -1}}},
            "artifact totals are invalid",
        ),
        (
            {
                "artifacts": {
                    "pack": {
                        "kind": "directory",
                        "file_count": True,
                        "size_bytes": 1,
                    }
                }
            },
            "artifact totals are invalid",
        ),
    ],
)
def test_artifact_totals_fail_closed(entry: dict[str, object], message: str) -> None:
    module = _load()

    with pytest.raises(ValueError, match=message):
        module._artifact_totals([entry])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"artifact_paths": []}, "artifact_paths must name only"),
        ({"summary": " leading"}, "summary must be concise plain text"),
        ({"summary": "bad\nsummary"}, "summary must be concise plain text"),
        ({"summary": "x" * 513}, "summary must be concise plain text"),
        (
            {
                "artifact_paths": {
                    "evidence_pack": "pack",
                    "verification_receipt": "evidence.meta.json",
                }
            },
            "invalid direct-child verification_receipt path",
        ),
        (
            {
                "artifact_paths": {
                    "evidence_pack": "pack",
                    "verification_receipt": "C:\\receipt.json",
                }
            },
            "invalid direct-child verification_receipt path",
        ),
    ],
)
def test_validate_metadata_rejects_ambiguous_publication_metadata(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    module = _load()
    value: dict[str, object] = {
        "format_version": "invarlock/public-evidence-meta-v1",
        "summary": "Verified evidence",
        "artifact_paths": {
            "evidence_pack": "pack",
            "verification_receipt": "receipt.json",
        },
    }
    value.update(mutation)

    with pytest.raises(ValueError, match=message):
        module._validate_metadata(tmp_path / "evidence.meta.json", value)


def test_build_rejects_symlinked_direct_child_escape(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "public_evidence"
    entry = source / "evidence" / "escape"
    entry.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (entry / "pack").symlink_to(outside, target_is_directory=True)
    (entry / "receipt.json").write_text("{}\n", encoding="utf-8")
    (entry / "evidence.meta.json").write_text(
        json.dumps(
            {
                "format_version": "invarlock/public-evidence-meta-v1",
                "summary": "Escaping pack",
                "artifact_paths": {
                    "evidence_pack": "pack",
                    "verification_receipt": "receipt.json",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a direct entry child"):
        module.build_public_evidence_index(source)


def test_check_reports_each_stale_index_and_obsolete_packaged_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "public_evidence"
    source.mkdir()
    packaged = tmp_path / "packaged"
    (packaged / "evidence").mkdir(parents=True)

    result = _run(source, packaged, "--check")

    assert result.returncode == 1
    assert "full packaged public evidence tree must be removed" in result.stdout
    assert "source public evidence index is out of sync" in result.stdout
    assert "packaged public evidence index is out of sync" in result.stdout


def test_write_removes_obsolete_packaged_evidence_tree(tmp_path: Path) -> None:
    source = tmp_path / "public_evidence"
    source.mkdir()
    packaged = tmp_path / "packaged"
    obsolete = packaged / "evidence"
    obsolete.mkdir(parents=True)
    (obsolete / "old-pack.json").write_text("{}\n", encoding="utf-8")

    result = _run(source, packaged, "--write")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not obsolete.exists()
