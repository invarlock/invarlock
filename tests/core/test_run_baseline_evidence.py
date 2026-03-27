from __future__ import annotations

import json
from pathlib import Path

from invarlock.core.run_baseline_evidence import load_baseline_pairing_evidence


def _extract_pairing_schedule(report: dict | None) -> dict | None:
    if not isinstance(report, dict):
        return None
    return report.get("pairing_schedule")


def test_load_baseline_pairing_evidence_missing_path(tmp_path: Path) -> None:
    result = load_baseline_pairing_evidence(
        baseline_path=tmp_path / "missing.json",
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "missing_path"
    assert result.report_data is None
    assert result.pairing_schedule is None
    assert "PAIRING-EVIDENCE-MISSING" in str(result.message)


def test_load_baseline_pairing_evidence_parse_failure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not-json", encoding="utf-8")

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "parse_failed"
    assert result.report_data is None
    assert result.pairing_schedule is None
    assert "JSON parse failed" in str(result.message)


def test_load_baseline_pairing_evidence_invalid_schedule(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"meta": {"tokenizer_hash": "tok"}}), encoding="utf-8"
    )

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "missing_schedule"
    assert result.report_data is None
    assert result.pairing_schedule is None
    assert "missing or invalid evaluation_windows" in str(result.message)


def test_load_baseline_pairing_evidence_merges_schedule_and_harvests_hash(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"logloss": [0.1]},
                    "final": {"token_counts": [3]},
                },
                "pairing_schedule": {
                    "preview": {"window_ids": [1], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [2], "input_ids": [[4, 5, 6]]},
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "loaded"
    assert result.tokenizer_hash == "tokhash123"
    assert result.pairing_schedule == {
        "preview": {"window_ids": [1], "input_ids": [[1, 2, 3]]},
        "final": {"window_ids": [2], "input_ids": [[4, 5, 6]]},
    }
    assert result.report_data is not None
    assert result.report_data["evaluation_windows"]["preview"]["window_ids"] == [1]
    assert result.report_data["evaluation_windows"]["preview"]["logloss"] == [0.1]
    assert result.report_data["evaluation_windows"]["final"]["token_counts"] == [3]
