from __future__ import annotations

from invarlock.reporting import dataset_hashing, report_normalization
from invarlock.reporting.report_make import make_report
from tests.reporting._support_report_make_paths import (
    _base_baseline,
    _base_report,
    _patch_common,
    _stub_evaluation_report_extractors,
)


def test_make_evaluation_report_handles_missing_dataset_section(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"].pop("tokenizer_hash", None)
    report["data"] = None

    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization, "normalize_baseline", lambda value: value, raising=False
    )
    monkeypatch.setattr(
        dataset_hashing,
        "_extract_dataset_info",
        lambda *_: {"hash": {}, "windows": {}},
    )

    evaluation_report = make_report(report, baseline)
    assert "tokenizer_hash" not in evaluation_report["meta"]


def test_make_evaluation_report_preserves_nullable_provenance(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"]["model_id"] = None
    report["meta"]["adapter"] = ""
    report["meta"]["device"] = None
    report["edit"]["name"] = None
    baseline["meta"].pop("model_id", None)
    baseline.pop("model_id", None)
    baseline.pop("run_id", None)

    _patch_common(monkeypatch, report, baseline)
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info={"hash": {}, "windows": {"stats": {}}},
        resolved_policy={"spectral": {}, "variance": {}},
    )

    evaluation_report = make_report(report, baseline)

    assert evaluation_report["meta"]["model_id"] is None
    assert evaluation_report["meta"]["adapter"] is None
    assert evaluation_report["meta"]["device"] is None
    assert "edit_name" not in evaluation_report
    assert "name" not in evaluation_report["edit"]
    assert evaluation_report["baseline_ref"]["model_id"] is None
    assert evaluation_report["baseline_ref"]["run_id"] is None
    diagnostics = evaluation_report["meta"].get("build_diagnostics", [])
    codes = {entry["code"] for entry in diagnostics}
    assert {
        "meta.model_id_unavailable",
        "meta.adapter_unavailable",
        "meta.device_unavailable",
    }.issubset(codes)
