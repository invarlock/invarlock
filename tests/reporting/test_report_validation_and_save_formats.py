from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.core import guard_evidence as report_evidence_mod
from invarlock.reporting import report_bundle as report_bundle_mod
from invarlock.reporting import report_summary as report_summary_mod
from invarlock.reporting.report_bundle import save_evaluation_bundle
from invarlock.reporting.report_files import save_report
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.run_report_formatters import to_html, to_markdown


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
        to_markdown({})


def test_to_markdown_raises_on_invalid_comparison_report() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        to_markdown(rp, compare={})


def test_to_html_raises_on_invalid_comparison_report() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        to_html(rp, compare={})


def test_to_html_raises_on_invalid_primary_report() -> None:
    with pytest.raises(ValueError, match="Invalid primary RunReport structure"):
        to_html({})


def test_save_report_defaults_to_json_markdown_html(tmp_path: Path) -> None:
    rp = _valid_run_report()
    saved = save_report(rp, tmp_path, formats=None)
    assert set(saved) == {"json", "markdown", "html"}


def test_save_report_comparison_suffix(tmp_path: Path) -> None:
    rp = _valid_run_report()
    compare = _valid_run_report()
    saved = save_report(rp, tmp_path, formats=["json", "markdown"], compare=compare)
    assert saved["json"].name == "report_comparison.json"
    assert saved["markdown"].name == "report_comparison.md"


def test_save_report_allows_markdown_only(tmp_path: Path) -> None:
    rp = _valid_run_report()
    saved = save_report(rp, tmp_path, formats=["markdown"])
    assert set(saved) == {"markdown"}
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "report.html").exists()


def test_save_report_cert_manifest_skips_non_dict_guards_and_empty_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    rp["guards"] = [
        "not-a-dict",
        {"policy": "bad"},
    ]
    monkeypatch.setattr(report_bundle_mod, "validate_report", lambda *_a, **_k: True)
    monkeypatch.setattr(
        report_bundle_mod, "render_report_markdown", lambda *_a, **_k: "{}"
    )
    saved = save_evaluation_bundle(
        run_report=rp,
        output_dir=tmp_path,
        evaluation_report={},
    )
    assert "report" in saved
    assert "report_md" in saved
    assert "manifest" in saved


def test_save_report_manifest_best_effort_handles_non_object_eval_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    monkeypatch.setattr(report_bundle_mod, "validate_report", lambda *_a, **_k: True)
    monkeypatch.setattr(
        report_bundle_mod, "render_report_markdown", lambda *_a, **_k: "[]"
    )
    saved = save_evaluation_bundle(
        run_report=rp,
        output_dir=tmp_path,
        evaluation_report=[],
    )
    assert "report" in saved
    assert "manifest" not in saved


def test_save_report_manifest_failures_do_not_abort_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    monkeypatch.setattr(report_bundle_mod, "validate_report", lambda *_a, **_k: True)
    monkeypatch.setattr(
        report_bundle_mod,
        "build_report_manifest_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(
        report_bundle_mod, "render_report_markdown", lambda *_a, **_k: "{}"
    )
    saved = save_evaluation_bundle(
        run_report=rp,
        output_dir=tmp_path,
        evaluation_report={"schema_version": "v1", "run_id": "x", "primary_metric": {}},
    )
    assert "report" in saved
    assert "manifest" not in saved


def test_save_evaluation_bundle_uses_manifest_summary_view_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rp = _valid_run_report()
    summary = report_summary_mod.ReportManifestSummary(
        run_model="m",
        device="cpu",
        seed=1,
        overall_status="PASS",
        primary_metric_ratio=1.0,
        gates_passed=3,
        gates_total=4,
    )
    seen: dict[str, object] = {}

    def fake_build_manifest_summary(
        run_report: dict[str, object], evaluation_report: dict[str, object]
    ) -> report_summary_mod.ReportManifestSummary:
        seen["run_report"] = run_report
        seen["evaluation_report"] = evaluation_report
        return summary

    monkeypatch.setattr(
        report_bundle_mod,
        "build_report_manifest_summary",
        fake_build_manifest_summary,
    )
    monkeypatch.setattr(report_bundle_mod, "validate_report", lambda *_a, **_k: True)
    monkeypatch.setattr(
        report_bundle_mod, "render_report_markdown", lambda *_a, **_k: "{}"
    )

    saved = save_evaluation_bundle(
        run_report=rp,
        output_dir=tmp_path,
        evaluation_report={"schema_version": "v1", "run_id": "x", "primary_metric": {}},
    )
    assert seen["run_report"] == rp
    assert seen["evaluation_report"]["schema_version"] == "v1"
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert '"overall_status": "PASS"' in manifest
    assert '"gates_passed": 3' in manifest
    assert '"evidence_level": "medium"' in manifest
    assert '"reviewer_summary_txt"' in manifest
    assert "manifest" in saved
    assert "reviewer_summary" in saved


def test_build_guard_evidence_payload_handles_guard_access_errors() -> None:
    class BadReport:
        def get(self, _key: str):
            raise RuntimeError("boom")

    payload = report_evidence_mod.build_guard_evidence_payload(BadReport())
    assert payload == {"guards_decisions": []}
