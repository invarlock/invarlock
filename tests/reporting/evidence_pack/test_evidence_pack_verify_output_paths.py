from __future__ import annotations

import json
from pathlib import Path

from invarlock.evidence_pack import EvidencePackStatus, verify_evidence_pack
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from tests.reporting._support_evidence_pack_paths import (
    _build_pack,
    _write_json,
    evidence_pack_mod,
)


def test_evidence_pack_verify_writes_nested_verify_json(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    json_out = tmp_path / "verify.json"

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={
                "format_version": "verify-v1",
                "ok": True,
                "reports": [str(path) for path in reports],
            },
            diagnostics=(),
        ),
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
        raising=True,
    )

    result = verify_evidence_pack(pack_dir, json_out_path=json_out)
    payload = result.payload
    exit_code = result.status

    assert exit_code == EvidencePackStatus.OK
    assert payload["verify"]["format_version"] == "verify-v1"
    assert json.loads(json_out.read_text(encoding="utf-8"))["ok"] is True


def test_scenario_strictness_and_report_scenario_helpers(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    metadata = pack_dir / "metadata"
    metadata.mkdir(parents=True)

    assert evidence_pack_mod._scenario_strictness_by_id(pack_dir) == {}

    scenarios = metadata / "scenarios.json"
    scenarios.write_text("{not-json", encoding="utf-8")
    assert evidence_pack_mod._scenario_strictness_by_id(pack_dir) == {}

    _write_json(scenarios, {"scenarios": "not-a-list"})
    assert evidence_pack_mod._scenario_strictness_by_id(pack_dir) == {}

    _write_json(
        scenarios,
        {
            "scenarios": [
                "skip",
                {"id": "clean", "strictness": "strict"},
                {"id": 7, "strictness": "ignored"},
            ]
        },
    )
    assert evidence_pack_mod._scenario_strictness_by_id(pack_dir) == {"clean": "strict"}

    report = (
        pack_dir / "reports" / "model" / "clean" / "noop" / "evaluation.report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    assert evidence_pack_mod._report_scenario_id(pack_dir, report) == "clean"
    assert (
        evidence_pack_mod._report_scenario_id(pack_dir, tmp_path / "outside.json")
        is None
    )
