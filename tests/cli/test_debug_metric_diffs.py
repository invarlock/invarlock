from __future__ import annotations

from invarlock.cli.commands.run import _format_debug_metric_diffs


def test_debug_metric_diffs_logs_small_deltas():
    # Arrange: pm and legacy ppl_* are identical → deltas near zero
    pm = {
        "kind": "ppl_causal",
        "preview": 50.0,
        "final": 50.0,
        "ratio_vs_baseline": 1.0,
    }
    metrics = {"primary_metric": pm}

    # Act
    line = _format_debug_metric_diffs(pm, metrics, baseline_report_data=None)

    # Assert: expected keys present and tiny deltas
    assert "final:" in line or "preview:" in line or "ratio_vs_baseline" in line
    # Extract all numbers after '=' signs and ensure they are tiny
    pieces = [seg.strip() for seg in line.split(";") if "=" in seg]
    for seg in pieces:
        num = float(seg.split("=")[-1])
        assert abs(num) < 1e-6
