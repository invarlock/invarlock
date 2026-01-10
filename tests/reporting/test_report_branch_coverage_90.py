from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.reporting import report as report_mod
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


def test_to_certificate_raises_on_unsupported_format() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Unsupported certificate format"):
        report_mod.to_certificate(rp, rp, format="yaml")


def test_to_certificate_raises_on_invalid_primary_report() -> None:
    rp = _valid_run_report()
    with pytest.raises(ValueError, match="Invalid primary RunReport structure"):
        report_mod.to_certificate({}, rp)


def test_save_report_defaults_to_json_markdown_html(tmp_path: Path) -> None:
    rp = _valid_run_report()
    saved = report_mod.save_report(rp, tmp_path, formats=None)
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
    monkeypatch.setattr(report_mod, "to_certificate", lambda *_a, **_k: "{}")
    saved = report_mod.save_report(rp, tmp_path, formats=["cert"], baseline=baseline)
    assert "cert" in saved
    assert "cert_md" in saved
    assert "manifest" in saved
