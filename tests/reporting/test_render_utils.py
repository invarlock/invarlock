from __future__ import annotations

from invarlock.reporting import render as render_mod


def test_compute_console_validation_block_handles_guard(monkeypatch):
    labels = [
        "Primary Metric Acceptable",
        "Guard Overhead Acceptable",
        "Invariants Pass",
    ]
    monkeypatch.setattr(
        render_mod, "_load_console_labels", lambda: labels, raising=False
    )

    certificate = {
        "validation": {
            "primary_metric_acceptable": True,
            "invariants_pass": False,
            "guard_overhead_acceptable": True,
        },
        "guard_overhead": {"evaluated": False, "ok": True},
    }
    block = render_mod.compute_console_validation_block(certificate)
    assert "Guard Overhead Acceptable" not in block["labels"]
    assert block["overall_pass"] is False

    certificate["guard_overhead"]["evaluated"] = True
    block2 = render_mod.compute_console_validation_block(certificate)
    guard_rows = [row for row in block2["rows"] if "Guard Overhead" in row["label"]]
    assert guard_rows and guard_rows[0]["ok"] is True
    assert block2["overall_pass"] is False


def test_formatting_helpers_cover_variants():
    plugin = {"name": "demo", "version": "1.0", "module": "invarlock.plugins.demo"}
    formatted = render_mod._format_plugin(plugin)
    assert "**demo**" in formatted and "`invarlock.plugins.demo`" in formatted
    long_value = "abcd" * 5
    expected_short = long_value[:8] + "…" + long_value[-8:]
    assert render_mod._short_digest(long_value) == expected_short
    assert render_mod._fmt_by_kind(0.75, "accuracy") == "75.0"
    assert render_mod._fmt_by_kind(12.3456, "ppl_mlm") == "12.3"
    assert render_mod._fmt_by_kind("oops", "other") == "N/A"
    assert render_mod._fmtv("latency_ms_p95", 10.2) == "10"
    assert render_mod._fmtv("throughput_sps", 4.56) == "4.6"
    assert render_mod._fmtv("delta", 1.234) == "1.234"
    assert render_mod._fmtv("delta", None) == "-"
    assert render_mod._p(0.125) == "12.5%"
    assert render_mod._p("bad") == "N/A"


def test_append_system_overhead_section(tmp_path):
    lines: list[str] = []
    sys_over = {
        "latency_ms_p50": {"baseline": 10, "edited": 12, "delta": 2, "ratio": 1.2},
        "throughput_sps": {"baseline": 0, "edited": 0},
    }
    render_mod._append_system_overhead_section(lines, sys_over)
    joined = "\n".join(lines)
    assert "System Overhead" in joined
    assert "Latency p50" in joined
    assert "12.000" not in joined  # formatted without decimals
    assert "Throughput" in joined and "N/A" in joined


def test_append_accuracy_subgroups():
    lines: list[str] = []
    subgroups = {
        "Group A": {
            "n_preview": 5,
            "n_final": 6,
            "accuracy_preview": 0.7,
            "accuracy_final": 0.8,
        }
    }
    render_mod._append_accuracy_subgroups(lines, subgroups)
    out = "\n".join(lines)
    assert "Accuracy Subgroups" in out
    assert "Group A" in out
