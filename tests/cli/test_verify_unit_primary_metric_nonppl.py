from __future__ import annotations

from invarlock.cli.commands import verify as V


def test_validate_primary_metric_nonppl_requires_ratio() -> None:
    cert = {"primary_metric": {"kind": "accuracy", "final": 0.8, "preview": 0.8}}
    errs = V._validate_primary_metric(cert)
    assert any("missing primary_metric.ratio_vs_baseline" in e for e in errs)
