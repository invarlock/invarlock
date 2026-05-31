from __future__ import annotations

import json

import pytest

from invarlock.reporting.report_builder_support import (
    build_telemetry_payload,
    save_telemetry_report,
)


def test_build_telemetry_payload_includes_timings_and_memory(tmp_path) -> None:
    report = {
        "meta": {"model_id": "m", "adapter": "a", "device": "cpu", "run_id": "r1"},
        "metrics": {
            "timings": {"prepare": 1.0, "eval": 2.0},
            "guard_timings": {"spectral": 0.4},
            "memory_snapshots": [{"phase": "eval", "rss_mb": 12.0}],
            "memory_mb_peak": 12.0,
            "gpu_memory_mb_peak": 3.0,
            "latency_ms_per_tok": 1.2,
            "total_tokens": 10,
        },
    }

    payload = build_telemetry_payload(report)
    assert payload["meta"]["model_id"] == "m"
    assert payload["timings"]["prepare"] == 1.0
    assert payload["guard_timings"]["spectral"] == 0.4
    assert payload["memory"]["memory_mb_peak"] == 12.0
    assert payload["performance"]["total_tokens"] == 10.0

    path = save_telemetry_report(report, tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["timings"]["eval"] == 2.0


def test_save_telemetry_report_rejects_path_escape(tmp_path) -> None:
    report = {"meta": {}, "metrics": {}}

    with pytest.raises(ValueError, match="plain file name"):
        save_telemetry_report(report, tmp_path / "out", filename="../telemetry.json")


def test_save_telemetry_report_rejects_empty_filename(tmp_path) -> None:
    report = {"meta": {}, "metrics": {}}

    with pytest.raises(ValueError, match="plain file name"):
        save_telemetry_report(report, tmp_path / "out", filename="")
