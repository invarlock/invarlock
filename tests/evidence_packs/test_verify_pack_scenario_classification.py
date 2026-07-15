from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_packs.python import verify_pack_report_classification


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reports_require_current_scenario_classification(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    declared = pack / "reports/model/clean/evaluation.report.json"
    undeclared = pack / "reports/model/errors/unknown/evaluation.report.json"
    _write_json(declared, {})
    _write_json(undeclared, {})

    assert verify_pack_report_classification.unclassified_reports(
        pack, [declared, undeclared]
    ) == [
        declared,
        undeclared,
    ]

    _write_json(
        pack / "metadata/scenarios.json",
        {"scenarios": [{"id": "clean", "strictness": "must_pass"}]},
    )
    assert verify_pack_report_classification.unclassified_reports(
        pack, [declared, undeclared]
    ) == [undeclared]

    _write_json(
        pack / "metadata/scenarios.json",
        {
            "scenarios": [
                {"id": "clean", "strictness": "must_pass"},
                {"id": "unknown", "strictness": "must_fail"},
            ]
        },
    )
    assert (
        verify_pack_report_classification.unclassified_reports(
            pack, [declared, undeclared]
        )
        == []
    )


def test_invalid_strictness_is_not_classification(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report = pack / "reports/model/clean/evaluation.report.json"
    _write_json(report, {})
    _write_json(
        pack / "metadata/scenarios.json",
        {"scenarios": [{"id": "clean", "strictness": "optional"}]},
    )

    assert verify_pack_report_classification.unclassified_reports(pack, [report]) == [
        report
    ]
    assert not verify_pack_report_classification.report_expects_verify_failure(
        pack, report
    )


def test_informational_and_must_detect_strictness_are_classified(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    informational = (
        pack / "reports/model/quant_8bit_deployable/run_1/evaluation.report.json"
    )
    must_detect = pack / "reports/model/errors/rmt_norm_noise/evaluation.report.json"
    _write_json(informational, {})
    _write_json(must_detect, {})
    _write_json(
        pack / "metadata/scenarios.json",
        {
            "scenarios": [
                {"id": "quant_8bit_deployable", "strictness": "informational"},
                {"id": "rmt_norm_noise", "strictness": "must_detect"},
            ]
        },
    )

    assert (
        verify_pack_report_classification.unclassified_reports(
            pack, [informational, must_detect]
        )
        == []
    )
    assert verify_pack_report_classification.report_is_informational(
        pack, informational
    )
    assert not verify_pack_report_classification.report_expects_verify_failure(
        pack, must_detect
    )


def test_standalone_builder_report_can_be_explicitly_classified(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report = pack / "reports/report-001/evaluation.report.json"
    _write_json(report, {})
    _write_json(
        pack / "metadata/scenarios.json",
        {"scenarios": [{"id": "report-001", "strictness": "must_pass"}]},
    )

    assert verify_pack_report_classification.unclassified_reports(pack, [report]) == []
