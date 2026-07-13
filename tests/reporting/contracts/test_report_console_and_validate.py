from __future__ import annotations

from invarlock.reporting import report_schema as schema_mod
from invarlock.reporting.report_summary import compute_console_validation_block


def test_compute_console_validation_block_guard_omitted_and_included():
    # Guard not evaluated → row omitted; overall pass computed from others
    cert = {
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "guard_metric_impact": {"evaluated": False},
    }
    blk = compute_console_validation_block(cert)
    labels = blk["labels"]
    assert all("Guard Metric Impact" not in lab for lab in labels)
    assert blk["overall_pass"] is True

    # Guard evaluated and passing → row included
    cert2 = {
        "validation": {**cert["validation"], "guard_metric_impact_acceptable": True},
        "guard_metric_impact": {"evaluated": True},
    }
    blk2 = compute_console_validation_block(cert2)
    assert any("Guard Metric Impact" in lab for lab in blk2["labels"])
    assert blk2["overall_pass"] is True

    # Guard evaluated and failing → overall fail
    cert3 = {
        "validation": {**cert["validation"], "guard_metric_impact_acceptable": False},
        "guard_metric_impact": {"evaluated": True},
    }
    blk3 = compute_console_validation_block(cert3)
    assert blk3["overall_pass"] is False


def test_validate_evaluation_report_rejects_payload_when_schema_validation_fails(
    monkeypatch,
):
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda c: False)
    good = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r1",
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
        },
    }
    assert schema_mod.validate_report(good) is False

    bad = {
        **good,
        "validation": {"primary_metric_acceptable": "not-bool"},
    }
    assert schema_mod.validate_report(bad) is False
