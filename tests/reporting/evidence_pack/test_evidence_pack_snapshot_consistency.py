from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_snapshot as snapshot_mod
from tests.reporting._support_evidence_pack_paths import (
    _build_pack,
    _sign_pack,
    _write_json,
)


def test_pack_snapshot_is_file_backed_digest_bound_and_cleans_up(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    payload = b"authenticated evidence bytes\n" * 1024
    source = pack_dir / "evidence.bin"
    source.write_bytes(payload)

    snapshot, errors = snapshot_mod.PackSnapshot.capture(pack_dir)

    assert errors == []
    assert snapshot is not None
    entry = snapshot.files.entry("evidence.bin")
    assert entry is not None
    assert not hasattr(entry, "payload")
    assert entry.snapshot_path != source
    assert entry.sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.files.digest_ledger == {"evidence.bin": entry.sha256}
    with pytest.raises(TypeError):
        snapshot.files.digest_ledger["evidence.bin"] = "0" * 64  # type: ignore[index]

    backing_root = entry.snapshot_path.parent
    with snapshot.files.materialized() as materialized:
        assert materialized == backing_root
        assert (materialized / "evidence.bin").read_bytes() == payload

    assert not backing_root.exists()


def test_materialized_snapshot_detects_backing_file_tampering(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "evidence.json").write_text('{"ok": true}\n', encoding="utf-8")
    snapshot, errors = snapshot_mod.PackSnapshot.capture(pack_dir)
    assert errors == []
    assert snapshot is not None
    entry = snapshot.files.entry("evidence.json")
    assert entry is not None
    entry.snapshot_path.chmod(0o600)
    entry.snapshot_path.write_text('{"ok": false}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable snapshot digest changed"):
        with snapshot.files.materialized():
            pass


def test_materialized_snapshot_rejects_backing_symlink_substitution(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    source = pack_dir / "evidence.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    snapshot, errors = snapshot_mod.PackSnapshot.capture(pack_dir)
    assert errors == []
    assert snapshot is not None
    entry = snapshot.files.entry("evidence.json")
    assert entry is not None
    entry.snapshot_path.unlink()
    entry.snapshot_path.symlink_to(source)

    with pytest.raises(RuntimeError, match="snapshot file became unsafe"):
        with snapshot.files.materialized():
            pass


def test_verify_authenticates_before_manifest_semantics(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_level"] = "not-a-level"
    _write_json(manifest_path, manifest)
    _sign_pack(pack_dir, tmp_path)
    signature_path = pack_dir / evidence_pack_mod.MANIFEST_SIGNATURE_FILENAME
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    signature["signature"]["value"] = "A" * 88
    _write_json(signature_path, signature)

    monkeypatch.setattr(
        evidence_pack_mod,
        "validate_manifest",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("manifest semantics ran before authentication")
        ),
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.SIGNATURE


def test_verify_parses_the_authenticated_manifest_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    manifest_path = pack_dir / "manifest.json"
    authenticated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authenticated_manifest["evidence_level"] = "not-a-level"
    _write_json(manifest_path, authenticated_manifest)
    _sign_pack(pack_dir, tmp_path)
    authenticated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_manifest = dict(authenticated_manifest)
    valid_manifest["evidence_level"] = "low"
    original_validate = evidence_pack_mod.validate_manifest

    def transient_substitution(path: Path) -> list[str]:
        if path == manifest_path:
            _write_json(path, valid_manifest)
            try:
                return original_validate(path)
            finally:
                _write_json(path, authenticated_manifest)
        return original_validate(path)

    monkeypatch.setattr(
        evidence_pack_mod,
        "validate_manifest",
        transient_substitution,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.FORMAT
    assert any("evidence_level" in error for error in result.payload["errors"])


def test_verify_reports_signed_non_json_manifest_as_format_error(
    tmp_path: Path,
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    (pack_dir / "manifest.json").write_text("{", encoding="utf-8")
    _sign_pack(pack_dir, tmp_path, record_manifest_fingerprint=False)

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.FORMAT
    assert any(
        "manifest is not valid JSON" in error for error in result.payload["errors"]
    )


def test_verify_rejects_pack_inventory_change_after_nested_verification(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    _sign_pack(pack_dir, tmp_path)

    def verify_reports(_snapshot_root: Path, **_kwargs):
        (pack_dir / "late-added.json").write_text("{}\n", encoding="utf-8")
        return [], {"ok": True}

    monkeypatch.setattr(evidence_pack_mod, "_verify_reports", verify_reports)

    result = evidence_pack_mod.verify_evidence_pack(
        pack_dir,
        report_assurance="off",
    )

    assert result.status is evidence_pack_mod.EvidencePackStatus.INTEGRITY
    assert any("pack snapshot changed" in error for error in result.payload["errors"])


def test_skip_verify_never_reports_integrity_after_pack_inventory_change(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    _sign_pack(pack_dir, tmp_path)

    def mutate_source_pack(_pack_dir: Path) -> list[str]:
        (pack_dir / "late-added.json").write_text("{}\n", encoding="utf-8")
        return []

    monkeypatch.setattr(
        evidence_pack_mod,
        "verify_manifest_provenance",
        mutate_source_pack,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.INTEGRITY
    assert result.payload["ok"] is False
    assert result.payload["integrity_ok"] is False
    assert result.payload["reports_verified"] is False
    assert result.payload["verification_scope"] == "not_verified"
    assert any("pack snapshot changed" in error for error in result.payload["errors"])
