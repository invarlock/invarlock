from __future__ import annotations

from pathlib import Path

from invarlock.public_contracts import published_basis_lanes


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
