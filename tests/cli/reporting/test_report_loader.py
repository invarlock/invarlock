from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.commands.report import _load_run_report
from invarlock.core.report_inputs import (
    load_evaluation_report_input_json,
    load_run_report_input_json,
    resolve_run_reports_from_evaluation_input,
)


def _run_report_payload() -> dict[str, object]:
    return {
        "meta": {"seed": 1},
        "data": {},
        "edit": {},
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 1.0}},
        "artifacts": {},
        "flags": {},
    }


def test_load_run_report_from_file(tmp_path: Path):
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"ok": True}), encoding="utf-8")
    out = _load_run_report(str(p))
    assert out == {"ok": True}


def test_load_run_report_from_dir_requires_canonical_report_name(tmp_path: Path):
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "my_report.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical run report file"):
        _load_run_report(str(tmp_path))


def test_load_run_report_from_dir_prefers_exact_canonical_report(tmp_path: Path):
    (tmp_path / "my_report.json").write_text(
        json.dumps({"hello": "fuzzy"}), encoding="utf-8"
    )
    (tmp_path / "report.json").write_text(
        json.dumps({"hello": "canonical"}), encoding="utf-8"
    )

    out = _load_run_report(str(tmp_path))

    assert out == {"hello": "canonical"}


def test_load_run_report_from_dir_rejects_ambiguous_canonical_names(
    tmp_path: Path,
):
    (tmp_path / "report.json").write_text(json.dumps({"kind": "run"}), encoding="utf-8")
    (tmp_path / "evaluation.report.json").write_text(
        json.dumps({"kind": "evaluation"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ambiguous report directory"):
        _load_run_report(str(tmp_path))


def test_load_run_report_dir_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="canonical run report file"):
        _load_run_report(str(tmp_path))


def test_load_evaluation_report_rejects_run_payload(tmp_path: Path):
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(json.dumps({"kind": "run"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected an evaluation report payload"):
        load_evaluation_report_input_json(str(report_path))


def test_load_run_report_error_uses_canonical_evaluate_wording(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"validation": {"primary_metric_acceptable": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_run_report_input_json(str(report_path))

    message = str(excinfo.value)
    assert "invarlock evaluate" in message
    assert "evaluate/run" not in message


def test_resolve_run_reports_from_evaluation_input_uses_provenance(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    subject_dir = runs_dir / "subject"
    baseline_dir = runs_dir / "baseline"
    subject_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)
    subject_report = subject_dir / "report.json"
    baseline_report = baseline_dir / "report.json"
    subject_report.write_text(json.dumps(_run_report_payload()), encoding="utf-8")
    baseline_report.write_text(json.dumps(_run_report_payload()), encoding="utf-8")

    report_dir = tmp_path / "reports" / "eval"
    report_dir.mkdir(parents=True)
    evaluation_path = report_dir / "evaluation.report.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "validation": {},
                "provenance": {
                    "edited": {"report_path": "../../runs/subject/report.json"},
                    "baseline": {"report_path": "../../runs/baseline/report.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    resolved_eval, resolved_subject, resolved_baseline = (
        resolve_run_reports_from_evaluation_input(str(evaluation_path))
    )

    assert resolved_eval == evaluation_path.resolve()
    assert resolved_subject == subject_report.resolve()
    assert resolved_baseline == baseline_report.resolve()


def test_resolve_run_reports_from_evaluation_input_requires_linked_paths(
    tmp_path: Path,
) -> None:
    evaluation_path = tmp_path / "evaluation.report.json"
    evaluation_path.write_text(
        json.dumps({"schema_version": "v1", "validation": {}, "provenance": {}}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation report bundle does not expose both linked run reports",
    ):
        resolve_run_reports_from_evaluation_input(str(evaluation_path))
