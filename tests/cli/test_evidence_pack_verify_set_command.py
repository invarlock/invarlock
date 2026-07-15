from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.cli.app import app


def test_evidence_pack_verify_set_forwards_exact_catalog_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_verify_set(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            payload={"ok": True, "warnings": [], "errors": []},
            status=evidence_pack_mod.EvidencePackStatus.OK,
        )

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.verify_evidence_pack_set",
        fake_verify_set,
        raising=True,
    )
    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "verify-set",
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--expected-catalog-digest",
            "sha256:" + ("c" * 64),
            "--expected-source-commit",
            "d" * 40,
            "--expected-source-bundle-digest",
            "sha256:" + ("e" * 64),
            "--expected-runtime-image-digest",
            "sha256:" + ("a" * 64),
            "--pack",
            str(tmp_path / "pack-a"),
            "--pack",
            str(tmp_path / "pack-b"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--expected-fingerprint",
            "sha256:" + ("f" * 64),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["catalog_path"] == tmp_path / "catalog.json"
    assert seen["pack_dirs"] == [tmp_path / "pack-a", tmp_path / "pack-b"]
    assert seen["receipt_path"] == tmp_path / "receipt.json"
    assert seen["expected_catalog_digest"] == "sha256:" + ("c" * 64)
    assert seen["expected_source_commit"] == "d" * 40
    assert seen["expected_source_bundle_digest"] == "sha256:" + ("e" * 64)
    assert seen["expected_runtime_image_digest"] == "sha256:" + ("a" * 64)
    assert seen["expected_fingerprint"] == "sha256:" + ("f" * 64)
