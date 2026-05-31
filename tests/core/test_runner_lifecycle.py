from __future__ import annotations

from typing import Any

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import (
    finalize_run_report,
    initialize_run_report,
    merge_execution_metrics,
)


def test_initialize_run_report_merges_context_and_auto_config() -> None:
    cfg = RunConfig(
        context={
            "run_id": "run-1",
            "plugins": {"adapters": ["hf"]},
            "auto": {"tier": "balanced", "keep": True},
        }
    )

    report = initialize_run_report(
        config=cfg,
        serialized_config={"profile": "ci"},
        cuda_flags={"deterministic_algorithms": True},
        auto_config={"tier": "aggressive"},
        start_time=123.0,
    )

    assert report.meta["start_time"] == 123.0
    assert report.meta["config"] == {"profile": "ci"}
    assert report.meta["run_id"] == "run-1"
    assert report.meta["plugins"] == {"adapters": ["hf"]}
    assert report.meta["auto"] == {"tier": "aggressive"}
    assert cfg.context["auto"] == {"tier": "aggressive", "keep": True}
    assert report.context["auto"] == {"tier": "aggressive", "keep": True}


def test_initialize_run_report_handles_context_update_failure() -> None:
    class _BadContext(dict):
        def update(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    class _BadReport(RunReport):
        def __init__(self) -> None:
            super().__init__()
            self.context = _BadContext()

    report = initialize_run_report(
        config=RunConfig(context={"run_id": "x"}),
        serialized_config={},
        cuda_flags={},
        report_factory=_BadReport,
        start_time=1.0,
    )

    assert isinstance(report.context, dict)
    assert report.context["run_id"] == "x"


def test_initialize_run_report_skips_auto_merge_for_non_dict_context() -> None:
    class _Mapping:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = dict(data)

        def __iter__(self):
            return iter(self._data.items())

        def get(self, key: str, default: Any = None) -> Any:
            return self._data.get(key, default)

    cfg = RunConfig(context=_Mapping({"run_id": "x"}))
    report = initialize_run_report(
        config=cfg,
        serialized_config={},
        cuda_flags={},
        auto_config={"tier": "aggressive"},
        start_time=2.0,
    )

    assert report.meta["auto"] == {"tier": "aggressive"}
    assert "auto" not in report.context


def test_initialize_run_report_replaces_non_mapping_report_context_and_bad_auto() -> (
    None
):
    class _ListContextReport(RunReport):
        def __init__(self) -> None:
            super().__init__()
            self.context = []

    cfg = RunConfig(
        context={"run_id": "", "plugins": {"guards": ["spectral"]}, "auto": "bad"}
    )
    report = initialize_run_report(
        config=cfg,
        serialized_config={"profile": "dev"},
        cuda_flags={"cuda": False},
        auto_config={"tier": "balanced"},
        report_factory=_ListContextReport,
        start_time=5.0,
    )

    assert report.context == {
        "run_id": "",
        "plugins": {"guards": ["spectral"]},
        "auto": {"tier": "balanced"},
    }
    assert report.meta["plugins"] == {"guards": ["spectral"]}
    assert cfg.context["auto"] == {"tier": "balanced"}


def test_finalize_run_report_records_duration_when_start_time_exists() -> None:
    report = RunReport()
    report.meta["start_time"] = 10.0

    finalize_run_report(report, final_status="success", end_time=13.5)

    assert report.status == "success"
    assert report.meta["end_time"] == 13.5
    assert report.meta["duration"] == 3.5


def test_finalize_run_report_skips_duration_without_numeric_start_time() -> None:
    report = RunReport()
    report.meta["start_time"] = "bad"

    finalize_run_report(report, final_status="failed", end_time=15.0)

    assert report.status == "failed"
    assert report.meta["end_time"] == 15.0
    assert "duration" not in report.meta


def test_merge_execution_metrics_merges_timings_and_memory_summary() -> None:
    report = RunReport()
    report.metrics = {"memory_mb_peak": 4.0}

    merge_execution_metrics(
        report,
        timings={"prepare": 1.0},
        guard_timings={"variance": 2.0},
        memory_snapshots=[{"phase": "prepare", "rss_mb": 1.0}],
        memory_summary={"memory_mb_peak": 3.0, "memory_delta_mb": 0.5},
    )

    assert report.metrics["timings"] == {"prepare": 1.0}
    assert report.metrics["guard_timings"] == {"variance": 2.0}
    assert report.metrics["memory_snapshots"] == [{"phase": "prepare", "rss_mb": 1.0}]
    assert report.metrics["memory_mb_peak"] == 4.0
    assert report.metrics["memory_delta_mb"] == 0.5


def test_merge_execution_metrics_handles_empty_memory_snapshots() -> None:
    report = RunReport()
    report.metrics = "bad"

    merge_execution_metrics(
        report,
        timings={},
        guard_timings={},
        memory_snapshots=[],
        memory_summary={"memory_mb_peak": 1.0},
    )

    assert report.metrics == {}


def test_merge_execution_metrics_preserves_non_numeric_existing_peak() -> None:
    report = RunReport()
    report.metrics = {"memory_mb_peak": "bad"}

    merge_execution_metrics(
        report,
        timings={"finalize": 0.5},
        guard_timings={},
        memory_snapshots=[{"phase": "finalize", "rss_mb": 3.0}],
        memory_summary={"memory_mb_peak": 2.0},
    )

    assert report.metrics["timings"] == {"finalize": 0.5}
    assert report.metrics["memory_mb_peak"] == 2.0
