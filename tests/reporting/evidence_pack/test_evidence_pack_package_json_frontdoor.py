from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_integrity as integrity_mod
from invarlock.cli.app import app
from invarlock.evidence_pack_json import (
    read_json_object_snapshot,
    sha256_prefixed,
)
from tests.reporting._support_evidence_pack_paths import _build_pack

_ALLOW_UNVERIFIED_PROVENANCE_ENV = {"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "1"}


def _pack(tmp_path: Path) -> Path:
    return _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )


@pytest.mark.parametrize(
    ("relative_path", "payload", "expected_fragment"),
    (
        (
            "metadata/scenarios.json",
            b'{"scenarios": [], "scenarios": []}',
            "duplicate key",
        ),
        (
            "metadata/environment.json",
            b'{"platform": NaN}',
            "non-standard constant",
        ),
        (
            "metadata/environment.json",
            b'{"platform": Infinity}',
            "non-standard constant",
        ),
        (
            "metadata/environment.json",
            b'{"platform": 1e999}',
            "non-finite number",
        ),
        (
            "results/final_verdict.json",
            b"[]",
            "must decode to a JSON object",
        ),
        (
            "reports/model/clean/noop/evaluation.report.json",
            b'{"value":"\xff"}',
            "not UTF-8 JSON",
        ),
        (
            "manifest.signature.json",
            b'{"format":"x", "format":"x"}',
            "duplicate key",
        ),
    ),
)
def test_package_verifier_rejects_ambiguous_structural_json_before_verification(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
    expected_fragment: str,
) -> None:
    pack_dir = _pack(tmp_path)
    target = pack_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.INTEGRITY
    assert result.payload["ok"] is False
    assert any(expected_fragment in error for error in result.payload["errors"])


def test_package_verifier_rejects_duplicate_manifest_keys_in_format_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = _pack(tmp_path)
    (pack_dir / "manifest.json").write_text(
        """
        {
          "format": "evidence-pack-v1",
          "format": "evidence-pack-v1",
          "checksums_sha256": "checksums.sha256",
          "checksums_sha256_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.FORMAT
    assert any("duplicate key" in error for error in result.payload["errors"])


def test_direct_cli_frontdoor_rejects_duplicate_scenario_manifest(
    tmp_path: Path,
) -> None:
    pack_dir = _pack(tmp_path)
    (pack_dir / "metadata/scenarios.json").write_text(
        '{"scenarios": [], "scenarios": []}', encoding="utf-8"
    )

    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "verify",
            str(pack_dir),
            "--json",
            "--skip-verify",
        ],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == evidence_pack_mod.EvidencePackStatus.INTEGRITY.value
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("duplicate key" in error for error in payload["errors"])


def test_direct_signature_reader_rejects_a_symlinked_bundle(tmp_path: Path) -> None:
    pack_dir = _pack(tmp_path)
    target = pack_dir / "signature-target.json"
    target.write_text("{}", encoding="utf-8")
    signature = pack_dir / integrity_mod.MANIFEST_SIGNATURE_FILENAME
    try:
        signature.symlink_to(target.name)
    except OSError as exc:  # pragma: no cover - platform-specific filesystem policy
        pytest.skip(f"symlinks unavailable in test filesystem: {exc}")

    errors, warnings, fingerprint = integrity_mod.verify_signature(
        pack_dir, strict=False
    )

    assert warnings == []
    assert fingerprint is None
    assert any("must not be a symlink" in error for error in errors)


def test_direct_signature_reader_rejects_a_nonregular_bundle(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable on this platform")
    pack_dir = _pack(tmp_path)
    signature = pack_dir / integrity_mod.MANIFEST_SIGNATURE_FILENAME
    try:
        os.mkfifo(signature)
    except OSError as exc:  # pragma: no cover - filesystem policy varies
        pytest.skip(f"named pipes unavailable in test filesystem: {exc}")

    errors, warnings, fingerprint = integrity_mod.verify_signature(
        pack_dir, strict=False
    )

    assert warnings == []
    assert fingerprint is None
    assert errors == ["manifest.signature.json must be a regular file."]


def test_package_verifier_rejects_nonregular_pack_entries(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable on this platform")
    pack_dir = _pack(tmp_path)
    fifo = pack_dir / "metadata" / "unsafe.fifo"
    try:
        os.mkfifo(fifo)
    except OSError as exc:  # pragma: no cover - filesystem policy varies
        pytest.skip(f"named pipes unavailable in test filesystem: {exc}")

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.INTEGRITY
    assert any(
        "only regular files and directories" in error
        for error in result.payload["errors"]
    )


def test_package_verifier_rejects_ambiguous_provenance_path_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = _pack(tmp_path)
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["path"] = "./metadata/environment.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status is evidence_pack_mod.EvidencePackStatus.INTEGRITY
    assert any(
        "path escapes the pack root" in error for error in result.payload["errors"]
    )


def test_strict_json_snapshot_returns_the_exact_bytes_used_for_parsing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.json"
    expected_bytes = b'{"candidate_id":"candidate-1","quality_loss":0.01}\n'
    path.write_bytes(expected_bytes)

    snapshot, payload = read_json_object_snapshot(path, label="candidate evidence")

    assert snapshot == expected_bytes
    assert payload == {"candidate_id": "candidate-1", "quality_loss": 0.01}
    assert sha256_prefixed(snapshot) == (
        "sha256:fdb9c181c5b948bd59c5b4e57c95f86467b9f4bf6f4530ecd4e60877cadc8129"
    )
