from __future__ import annotations

from invarlock.reporting import verify_contract as V


def test_validate_primary_metric_accuracy_requires_delta_pp() -> None:
    cert = {"primary_metric": {"kind": "accuracy", "final": 0.8, "preview": 0.8}}
    errs = V._validate_primary_metric(cert)
    assert any("delta_vs_baseline_pp must be finite" in e for e in errs)


def test_validate_primary_metric_accuracy_rejects_ratio_alias() -> None:
    cert = {
        "primary_metric": {
            "kind": "accuracy",
            "final": 0.8,
            "preview": 0.8,
            "delta_vs_baseline_pp": 0.0,
            "ratio_vs_baseline": 0.0,
        }
    }
    errs = V._validate_primary_metric(cert)
    assert any("ratio_vs_baseline is not allowed for accuracy" in e for e in errs)
