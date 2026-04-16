from __future__ import annotations

import json
from pathlib import Path

from invarlock.public_contracts import published_basis_lanes
from invarlock.reporting.report_schema import validate_report

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_published_basis_lanes_ship_public_evidence_references() -> None:
    for lane in published_basis_lanes():
        evidence = lane.get("evidence", {})
        assert isinstance(evidence, dict)
        report_fixture = evidence.get("evaluation_report_fixture")
        proof_pack_recipe = evidence.get("proof_pack_recipe")
        assert isinstance(report_fixture, str) and report_fixture
        assert isinstance(proof_pack_recipe, str) and proof_pack_recipe
        assert (REPO_ROOT / report_fixture).is_file(), report_fixture
        assert (REPO_ROOT / proof_pack_recipe).is_file(), proof_pack_recipe


def test_packaged_public_evidence_matches_repo_public_evidence() -> None:
    pairs = [
        (
            REPO_ROOT / "public_evidence" / "published_basis" / "gpt2",
            REPO_ROOT
            / "src"
            / "invarlock"
            / "_data"
            / "public_evidence"
            / "published_basis"
            / "gpt2",
        ),
        (
            REPO_ROOT / "public_evidence" / "published_basis" / "bert",
            REPO_ROOT
            / "src"
            / "invarlock"
            / "_data"
            / "public_evidence"
            / "published_basis"
            / "bert",
        ),
    ]

    for source_dir, packaged_dir in pairs:
        assert source_dir.is_dir(), source_dir
        assert packaged_dir.is_dir(), packaged_dir
        source_files = sorted(
            path.relative_to(source_dir)
            for path in source_dir.rglob("*")
            if path.is_file()
        )
        packaged_files = sorted(
            path.relative_to(packaged_dir)
            for path in packaged_dir.rglob("*")
            if path.is_file()
        )
        assert packaged_files == source_files
        for rel_path in source_files:
            assert (packaged_dir / rel_path).read_bytes() == (
                source_dir / rel_path
            ).read_bytes()


def test_offline_golden_runs_public_fixtures() -> None:
    manifest_path = REPO_ROOT / "tests/artifacts/golden_runs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["published_basis"] == ["gpt2", "bert"]

    for lane in manifest["lanes"]:
        report_path = REPO_ROOT / lane["report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert validate_report(report) is True
        assert report["meta"]["model_id"] == lane["model_id"]
        assert report["primary_metric"]["kind"] == lane["primary_metric_kind"]
        assert report["validation"]["primary_metric_acceptable"] is True
