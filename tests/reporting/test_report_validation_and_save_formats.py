from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.reporting import report as report_mod
from invarlock.reporting import report_files as report_files_mod
from invarlock.reporting.report_files import save_report
from invarlock.reporting.report_types import create_empty_report


def _valid_run_report():
    report = create_empty_report()
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
        "ratio_vs_baseline": 1.0,
    }
    return report


def test_to_markdown_raises_on_invalid_primary_report() -> None:
    with pytest.raises(ValueError, match="Invalid primary RunReport structure"):
        report_mod.to_markdown({})


def test_to_markdown_raises_on_invalid_comparison_report() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        report_mod.to_markdown(rp, compare={})


def test_to_html_raises_on_invalid_comparison_report() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        report_mod.to_html(rp, compare={})


def test_to_html_raises_on_invalid_primary_report() -> None:
    with pytest.raises(ValueError, match="Invalid primary RunReport structure"):
        report_mod.to_html({})


def test_to_evaluation_report_raises_on_unsupported_format() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Unsupported evaluation report format"):
        report_mod.to_evaluation_report(rp, rp, format="yaml")


def test_to_evaluation_report_raises_on_invalid_primary_report() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Invalid primary RunReport structure"):
        report_mod.to_evaluation_report({}, rp)


def test_save_report_defaults_to_json_markdown_html(tmp_path: Path) -> None:
    rp = _valid_run_report()
    saved = save_report(rp, tmp_path, formats=None)
    assert set(saved) == {"json", "markdown", "html"}


def test_save_report_cert_manifest_skips_non_dict_guards_and_empty_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    rp["guards"] = [
        "not-a-dict",
        {"policy": "bad"},
    ]
    baseline = _valid_run_report()
    monkeypatch.setattr(
        report_files_mod, "to_evaluation_report", lambda *_a, **_k: "{}"
    )
    saved = save_report(rp, tmp_path, formats=["report"], baseline=baseline)
    assert "report" in saved
    assert "report_md" in saved
    assert "manifest" in saved


def test_save_report_manifest_best_effort_handles_non_object_eval_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    baseline = _valid_run_report()
    monkeypatch.setattr(
        report_files_mod, "to_evaluation_report", lambda *_a, **_k: "[]"
    )
    saved = save_report(rp, tmp_path, formats=["report"], baseline=baseline)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["manifest"].name == "manifest.json"
    assert manifest["summary"]["run_model"] == rp["meta"]["model_id"]
    assert "overall_status" not in manifest["summary"]


def test_save_report_manifest_failures_do_not_abort_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    baseline = _valid_run_report()
    monkeypatch.setattr(
        report_files_mod.json,
        "loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    saved = save_report(rp, tmp_path, formats=["report"], baseline=baseline)
    assert "report" in saved
    assert "manifest" not in saved


def test_build_guard_evidence_payload_handles_guard_access_errors() -> None:
    class BadReport:
        def get(self, _key: str):
            raise RuntimeError("boom")

    payload = report_files_mod._build_guard_evidence_payload(BadReport())  # type: ignore[arg-type]
    assert payload == {"guards_decisions": []}
