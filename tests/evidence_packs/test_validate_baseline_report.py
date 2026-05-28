from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.evidence_packs.python import validate_baseline_report


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _baseline_payload() -> dict[str, Any]:
    return {
        "edit": {"name": "noop"},
        "meta": {"adapter": "hf_causal"},
        "context": {"profile": "ci", "auto": {"tier": "balanced"}},
        "evaluation_windows": {
            "preview": {"window_ids": ["p1"], "input_ids": [[1, 2]]},
            "final": {"window_ids": ["f1"], "input_ids": [[3, 4]]},
        },
    }


def _validate(path: Path) -> int:
    return validate_baseline_report.main([str(path), "hf_causal", "ci", "balanced"])


def test_validate_baseline_report_accepts_expected_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline.report.json"
    _write_report(report_path, _baseline_payload())

    assert _validate(report_path) == 0


def test_validate_baseline_report_requires_expected_metadata(
    tmp_path: Path,
) -> None:
    for field_path in (
        ("meta", "adapter"),
        ("context", "profile"),
        ("context", "auto"),
        ("context", "auto", "tier"),
    ):
        payload: dict[str, Any] = _baseline_payload()
        node: Any = payload
        for key in field_path[:-1]:
            assert isinstance(node, dict)
            node = node[key]
        assert isinstance(node, dict)
        node.pop(field_path[-1], None)

        report_path = tmp_path / ("baseline-" + "-".join(field_path) + ".json")
        _write_report(report_path, payload)

        assert _validate(report_path) == 1
