from __future__ import annotations

from copy import deepcopy

from invarlock.reporting import certificate as cert
from tests.reporting.test_certificate_full_context import _rich_run_report


def test_make_certificate_marks_tiny_relax(monkeypatch):
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    certificate = cert.make_certificate(report, baseline)
    assert certificate["auto"]["tiny_relax"] is True
    stats = certificate["dataset"]["windows"]["stats"]
    assert "coverage" in stats and "window_match_fraction" in stats
    qo = certificate.get("quality_overhead")
    if qo:
        assert qo["basis"] in {"ratio", "delta_pp"}


def test_make_certificate_emits_telemetry_summary(monkeypatch, capsys):
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    certificate = cert.make_certificate(report, baseline)
    out = capsys.readouterr().out
    assert "INVARLOCK_TELEMETRY run_id" in out
    assert certificate["telemetry"]["summary_line"].startswith("INVARLOCK_TELEMETRY")
