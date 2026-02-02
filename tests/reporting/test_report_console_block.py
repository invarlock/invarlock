from __future__ import annotations

from invarlock.reporting.render import compute_console_validation_block


def test_console_block_without_guard_overhead_is_pass():
    cert = {
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        }
    }
    block = compute_console_validation_block(cert)
    assert block["overall_pass"] is True
    rows = block["rows"]
    keys = {r["label"].strip().lower().replace(" ", "_") for r in rows}
    assert "guard_overhead_acceptable" not in keys  # row omitted when not evaluated


def test_console_block_with_guard_overhead_included_and_fail():
    cert = {
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": False,
        },
        "guard_overhead": {"evaluated": True},
    }
    block = compute_console_validation_block(cert)
    assert block["overall_pass"] is False
    rows = block["rows"]
    # Guard row present and marked evaluated
    assert any(
        r.get("evaluated") and "Guard Overhead" in r.get("label", "") for r in rows
    )
