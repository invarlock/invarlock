from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.public_contracts import load_support_matrix


def _parse_docs_evidence_labels() -> dict[str, str]:
    text = Path("docs/README.md").read_text(encoding="utf-8")
    section = text.split("## Support Matrix", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 4 or parts[0] in {"Surface", "---"}:
            continue
        rows[parts[1].strip("`")] = parts[3].strip("*")
    return rows


def _lane_tiers(payload: dict) -> dict[str, str]:
    lanes = payload.get("support_matrix", {}).get("lanes", [])
    return {
        lane["lane_id"]: lane["support_tier"]
        for lane in lanes
        if isinstance(lane, dict)
        and isinstance(lane.get("lane_id"), str)
        and isinstance(lane.get("support_tier"), str)
    }


def test_support_matrix_contract_matches_docs_and_cli_json_surfaces() -> None:
    contract = load_support_matrix()
    lanes = contract["lanes"]
    docs_labels = _parse_docs_evidence_labels()

    runner = CliRunner()
    plugins = runner.invoke(app, ["advanced", "plugins", "adapters", "--json"])
    assert plugins.exit_code == 0, plugins.output
    plugins_payload = json.loads(plugins.stdout.strip().splitlines()[-1])

    doctor = runner.invoke(app, ["doctor", "--json"])
    assert doctor.exit_code in (0, 1), doctor.output
    doctor_payload = json.loads(doctor.stdout.strip().splitlines()[-1])

    assert (
        plugins_payload["support_matrix"]["format_version"]
        == contract["format_version"]
    )
    assert (
        doctor_payload["support_matrix"]["format_version"] == contract["format_version"]
    )

    contract_tiers = {lane["lane_id"]: lane["support_tier"] for lane in lanes}
    assert _lane_tiers(plugins_payload) == contract_tiers
    assert _lane_tiers(doctor_payload) == contract_tiers

    assert len(lanes) == 39
    assert len({lane["family"] for lane in lanes}) == 39
    assert {lane["support_tier"] for lane in lanes} == {"published_basis"}
    assert {lane["evidence_status"] for lane in lanes} == {"not_created"}
    assert {lane["evidence_status_label"] for lane in lanes} == {
        "Evidence not yet created"
    }
    assert all("evidence" not in lane for lane in lanes)
    assert all(lane["support_groups"] for lane in lanes)

    expected_docs = {lane["lane_id"]: lane["evidence_status_label"] for lane in lanes}
    assert docs_labels == expected_docs
