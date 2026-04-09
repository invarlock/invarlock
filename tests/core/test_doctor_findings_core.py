from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from invarlock.core import auto_tuning
from invarlock.core import doctor_findings as mod
from invarlock.core.report_inputs import ReportInputError


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_explicit_report_input_formats_directory_and_json_errors(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    _, _, findings, had_error = mod.load_explicit_report_input(
        str(missing), label="Baseline", field="baseline_report"
    )
    assert had_error is True
    assert "not found" in findings[0].message

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    _write_json(ambiguous / "report.json", {})
    _write_json(ambiguous / "evaluation.report.json", {})

    _, _, findings, had_error = mod.load_explicit_report_input(
        str(ambiguous), label="Baseline", field="baseline_report"
    )

    assert had_error is True
    assert findings[0].code == "D014"
    assert "ambiguous" in findings[0].message

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    _, _, findings, had_error = mod.load_explicit_report_input(
        str(invalid), label="Subject", field="subject_report"
    )
    assert had_error is True
    assert "not valid JSON" in findings[0].message

    non_object = tmp_path / "list.json"
    non_object.write_text("[1, 2, 3]", encoding="utf-8")
    _, _, findings, had_error = mod.load_explicit_report_input(
        str(non_object), label="Subject", field="subject_report"
    )
    assert had_error is True
    assert "must decode to a JSON object" in findings[0].message


def test_load_explicit_report_input_formats_missing_canonical_directory(
    tmp_path: Path,
) -> None:
    non_canonical = tmp_path / "non-canonical"
    non_canonical.mkdir()
    _write_json(non_canonical / "my_report.json", {})

    _, _, findings, had_error = mod.load_explicit_report_input(
        str(non_canonical), label="Baseline", field="baseline_report"
    )

    assert had_error is True
    assert "does not contain a canonical report file" in findings[0].message


def test_build_provider_kind_findings_supports_mapping_and_object_inputs() -> None:
    findings, had_error = mod.build_provider_kind_findings({"kind": "bogus"})
    assert had_error is True
    assert findings[0].code == "D001"

    class ProviderObject:
        kind = "bogus"

    findings, had_error = mod.build_provider_kind_findings(ProviderObject())
    assert had_error is True
    assert findings[0].code == "D001"

    findings, had_error = mod.build_provider_kind_findings("wikitext2")
    assert findings == []
    assert had_error is False


def test_build_provider_schema_findings_cover_missing_path_and_blank_text_field() -> (
    None
):
    class BadPath:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mod.build_provider_schema_findings(
            {"kind": "local_jsonl", "file": BadPath(), "text_field": ""}
        )

    class HFProvider:
        kind = "hf_text"
        text_field = ""

    findings, had_error = mod.build_provider_schema_findings(HFProvider())
    assert had_error is False
    assert [finding.code for finding in findings] == ["D012"]


def test_build_provider_schema_findings_accepts_existing_local_jsonl_and_hf_text(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text": "hi"}\n', encoding="utf-8")

    findings, had_error = mod.build_provider_schema_findings(
        {"kind": "local_jsonl", "file": dataset, "text_field": "text"}
    )
    assert findings == []
    assert had_error is False

    findings, had_error = mod.build_provider_schema_findings(
        {"kind": "hf_text", "text_field": "body"}
    )
    assert findings == []
    assert had_error is False


def test_build_capacity_findings_handles_import_failure(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "invarlock.core.auto_tuning":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    findings, insufficient, policy_meta = mod.build_capacity_findings(
        cap={}, tier="dev"
    )
    assert findings == []
    assert insufficient is False
    assert policy_meta is None


def test_build_capacity_findings_reports_effective_floors_and_insufficiency() -> None:
    findings, insufficient, policy_meta = mod.build_capacity_findings(
        cap={"tokens_available": 10, "examples_available": 1},
        tier="balanced",
    )

    codes = [finding.code for finding in findings]
    assert "D007" in codes
    assert "D008" in codes
    assert insufficient is True
    assert policy_meta is not None
    assert policy_meta["tier"] == "balanced"


def test_build_capacity_findings_can_skip_floor_note_when_policy_is_zero(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        auto_tuning,
        "get_tier_policies",
        lambda: {
            "balanced": {
                "metrics": {
                    "pm_ratio": {"min_tokens": 0, "min_token_fraction": 0.0},
                    "accuracy": {"min_examples": 0, "min_examples_fraction": 0.0},
                }
            }
        },
    )

    findings, insufficient, policy_meta = mod.build_capacity_findings(
        cap={"tokens_available": 10, "examples_available": 10},
        tier="balanced",
    )

    assert findings == []
    assert insufficient is False
    assert policy_meta is not None


def test_build_capacity_findings_handles_examples_only_shortfall(monkeypatch) -> None:
    monkeypatch.setattr(
        auto_tuning,
        "get_tier_policies",
        lambda: {
            "balanced": {
                "metrics": {
                    "pm_ratio": {"min_tokens": 0, "min_token_fraction": 0.0},
                    "accuracy": {"min_examples": 2, "min_examples_fraction": 0.0},
                }
            }
        },
    )

    findings, insufficient, policy_meta = mod.build_capacity_findings(
        cap={"examples_available": 1},
        tier="balanced",
    )

    codes = [finding.code for finding in findings]
    assert codes == ["D007", "D008"]
    assert insufficient is True
    assert policy_meta is not None


def test_build_doctor_result_sorts_findings_and_summarizes() -> None:
    payload = mod.build_doctor_result(
        format_version="v1",
        findings=[
            {"code": "D013", "severity": "note", "message": "note"},
            {"code": "D001", "severity": "error", "message": "error"},
            {"code": "D005", "severity": "warning", "message": "warning"},
        ],
        contracts={},
        support_matrix={},
        model_family_catalog={},
        adapter_capabilities={},
        plugin_compatibility={},
        policy={},
    )

    assert [item["code"] for item in payload["findings"]] == ["D001", "D005", "D013"]
    assert payload["summary"] == {"errors": 1, "warnings": 1, "notes": 1}
    assert "resolution" not in payload


def test_doctor_accumulator_marks_errors_and_sorts() -> None:
    acc = mod.DoctorAccumulator()

    acc.add("D100", "warning", "warn")
    acc.add("D001", "error", "boom")
    acc.sort()

    assert acc.had_error is True
    assert [item["code"] for item in acc.findings] == ["D001", "D100"]


def test_mapping_get_handles_getter_exceptions() -> None:
    class BadGetter:
        def get(self, _key: str) -> object:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mod._mapping_get(BadGetter(), "field")


def test_mapping_get_returns_none_for_recoverable_getter_errors() -> None:
    class BadGetter:
        def get(self, _key: str) -> object:
            raise ValueError("boom")

    assert mod._mapping_get(BadGetter(), "field") is None


def test_format_report_input_error_covers_remaining_reasons(tmp_path: Path) -> None:
    unreadable = mod._format_report_input_error(
        label="Subject",
        exc=ReportInputError("unreadable", tmp_path / "report.json", detail="boom"),
    )
    assert "not readable" in unreadable

    non_regular = mod._format_report_input_error(
        label="Subject",
        exc=ReportInputError("non_regular", tmp_path / "pipe"),
    )
    assert "regular JSON file or canonical report directory" in non_regular

    fallback = mod._format_report_input_error(
        label="Subject",
        exc=ReportInputError("unknown_reason", tmp_path / "mystery.json"),
    )
    assert "input is invalid" in fallback


def test_build_cross_check_findings_marks_strict_split_mismatch_as_error(
    tmp_path: Path,
) -> None:
    baseline = _write_json(
        tmp_path / "baseline.json",
        {"provenance": {"dataset_split": "validation"}},
    )
    subject = _write_json(
        tmp_path / "subject.json",
        {"provenance": {"dataset_split": "train"}},
    )

    findings, had_error = mod.build_cross_check_findings(
        str(baseline),
        str(subject),
        cfg_metric_kind=None,
        strict=True,
        profile="ci",
    )

    assert had_error is True
    assert findings[0].code == "D011"
    assert findings[0].severity == "error"


def test_build_cross_check_findings_marks_dev_split_mismatch_as_warning(
    tmp_path: Path,
) -> None:
    baseline = _write_json(
        tmp_path / "baseline-dev.json",
        {"provenance": {"dataset_split": "validation"}},
    )
    subject = _write_json(
        tmp_path / "subject-dev.json",
        {"provenance": {"dataset_split": "train"}},
    )

    findings, had_error = mod.build_cross_check_findings(
        str(baseline),
        str(subject),
        cfg_metric_kind=None,
        strict=False,
        profile="dev",
    )

    assert had_error is False
    assert findings[0].code == "D011"
    assert findings[0].severity == "warning"


def test_build_provider_schema_findings_handles_exists_probe_errors_and_blank_text_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text": "hi"}\n', encoding="utf-8")

    monkeypatch.setattr(
        mod.Path,
        "exists",
        lambda self: (_ for _ in ()).throw(OSError("boom")),
    )
    findings, had_error = mod.build_provider_schema_findings(
        {"kind": "vision_text", "file": dataset}
    )
    assert had_error is True
    assert findings[0].code == "D011"

    monkeypatch.undo()
    findings, had_error = mod.build_provider_schema_findings(
        {"kind": "local_jsonl", "file": dataset, "text_field": "   "}
    )
    assert had_error is False
    assert [finding.code for finding in findings] == ["D012"]
