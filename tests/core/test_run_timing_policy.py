from __future__ import annotations

from invarlock.core.run_timing_policy import build_timing_summary_payload


def test_build_timing_summary_payload_prefers_breakdown_order_and_peak_lines() -> None:
    payload = build_timing_summary_payload(
        timings={
            "load_model": 1.0,
            "prepare": 2.0,
            "prepare_guards": 3.0,
            "edit": 4.0,
            "guards": 5.0,
            "eval": 6.0,
            "finalize": 7.0,
            "ignored": "x",
        },
        total_duration=9.5,
        report={
            "metrics": {
                "memory_mb_peak": 123.456,
                "gpu_memory_mb_peak": 78.9,
            }
        },
    )

    assert payload is not None
    assert payload.timings["total"] == 9.5
    assert payload.order == (
        ("Load model", "load_model"),
        ("Prepare", "prepare"),
        ("Prep guards", "prepare_guards"),
        ("Edit", "edit"),
        ("Guards", "guards"),
        ("Eval", "eval"),
        ("Finalize", "finalize"),
        ("Total", "total"),
    )
    assert payload.extra_lines == (
        "  Peak Memory : 123.46 MB",
        "  Peak GPU Mem: 78.90 MB",
    )


def test_build_timing_summary_payload_uses_execute_without_breakdown() -> None:
    payload = build_timing_summary_payload(
        timings={"load_dataset": 1.0, "execute": 4.0},
        total_duration=None,
        report={"metrics": {}},
    )

    assert payload is not None
    assert payload.order == (
        ("Load data", "load_dataset"),
        ("Execute", "execute"),
    )


def test_build_timing_summary_payload_returns_none_without_printable_timings() -> None:
    assert (
        build_timing_summary_payload(
            timings={"ignored": "x"},
            total_duration=None,
            report={"metrics": {"memory_mb_peak": "x"}},
        )
        is None
    )
