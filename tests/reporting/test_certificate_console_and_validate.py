from __future__ import annotations

from invarlock.reporting import certificate as C
from invarlock.reporting.render import compute_console_validation_block


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
        "guard_overhead": {"evaluated": False},
    }
    blk = compute_console_validation_block(cert)
    labels = blk["labels"]
    assert all("Guard Overhead" not in lab for lab in labels)
    assert blk["overall_pass"] is True

    # Guard evaluated and passing → row included
    cert2 = {
        "validation": {**cert["validation"], "guard_overhead_acceptable": True},
        "guard_overhead": {"evaluated": True},
    }
    blk2 = compute_console_validation_block(cert2)
    assert any("Guard Overhead" in lab for lab in blk2["labels"])
    assert blk2["overall_pass"] is True

    # Guard evaluated and failing → overall fail
    cert3 = {
        "validation": {**cert["validation"], "guard_overhead_acceptable": False},
        "guard_overhead": {"evaluated": True},
    }
    blk3 = compute_console_validation_block(cert3)
    assert blk3["overall_pass"] is False


def test_validate_certificate_fallback_and_flag_types(monkeypatch):
    # Force JSON schema validator to fail to exercise fallback path
    monkeypatch.setattr(C, "_validate_with_jsonschema", lambda c: False)
    good = {
        "schema_version": C.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r1",
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
        },
    }
    assert C.validate_certificate(good) is True

    bad = {
        **good,
        "validation": {"primary_metric_acceptable": "not-bool"},
    }
    assert C.validate_certificate(bad) is False
