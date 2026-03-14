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
        assert Path(report_fixture).is_file(), report_fixture
        assert Path(proof_pack_recipe).is_file(), proof_pack_recipe


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
