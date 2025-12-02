from __future__ import annotations

from invarlock.reporting.render import (
    build_console_summary_pack,
    compute_console_validation_block,
)


def test_console_summary_pack_basic():
    cert = {
        "schema_version": "v1",
        "run_id": "r-1",
        "artifacts": {"generated_at": "now"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "x",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            # guard overhead may be omitted (treated as not evaluated)
        },
        "guard_overhead": {"evaluated": False},
    }

    # Block should compute and overall pass should be True
    block = compute_console_validation_block(cert)
    assert block["overall_pass"] is True

    pack = build_console_summary_pack(cert)
    assert pack["overall_pass"] is True
    assert "PASS" in pack["overall_line"]
    # Should include one line per effective label (guard omitted in both)
    assert isinstance(pack["gate_lines"], list)
    assert len(pack["gate_lines"]) == len(block["labels"])  # guard omitted
