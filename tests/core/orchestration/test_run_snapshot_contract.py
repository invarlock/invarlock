from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.run_snapshot_contract import (
    SnapshotDiagnostic,
    build_snapshot_execution_plan,
    resolve_snapshot_retry_transition,
)


def test_build_snapshot_execution_plan_direct_reuse_sets_skip_model_load() -> None:
    plan = build_snapshot_execution_plan(
        adapter=SimpleNamespace(),
        model=object(),
        cfg_snapshot=None,
        direct_reuse_loaded_model=True,
        skip_guard_metric_impact_source="config:context.run.skip_guard_metric_impact_check",
        choose_snapshot_mode_fn=lambda **kwargs: "disabled",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={},
        disk_usage_fn=lambda path: SimpleNamespace(free=0),
        free_model_memory_fn=lambda model: None,
    )

    assert plan.skip_model_load is True
    assert plan.snapshot_enabled is None
    assert plan.emitted_skip_guard_metric_impact_warning is True
    assert plan.diagnostics == (
        SnapshotDiagnostic(
            code="snapshot.overhead_check_skipped",
            summary="Guard metric impact check skipped via config policy (config:context.run.skip_guard_metric_impact_check)",
            context={"source": "config:context.run.skip_guard_metric_impact_check"},
        ),
        SnapshotDiagnostic(
            code="snapshot.loaded_model_reused",
            summary="Reusing initially loaded model for guarded execution.",
        ),
    )


def test_build_snapshot_execution_plan_bytes_falls_back_to_chunked() -> None:
    calls: list[str] = []

    class Adapter:
        def snapshot(self, model):  # noqa: ANN001
            calls.append("snapshot")
            raise ValueError("bytes failed")

        def snapshot_chunked(self, model):  # noqa: ANN001
            calls.append("snapshot_chunked")
            return "/tmp/snap"

        def restore_chunked(self, model, path):  # noqa: ANN001
            calls.append(f"restore_chunked:{path}")

    plan = build_snapshot_execution_plan(
        adapter=Adapter(),
        model=object(),
        cfg_snapshot={},
        direct_reuse_loaded_model=False,
        skip_guard_metric_impact_source=None,
        choose_snapshot_mode_fn=lambda **kwargs: "bytes",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={},
        disk_usage_fn=lambda path: SimpleNamespace(free=0),
        free_model_memory_fn=lambda model: None,
    )

    assert plan.snapshot_enabled is True
    assert plan.diagnostics == (
        SnapshotDiagnostic(
            code="snapshot.bytes_failed_chunked_fallback",
            summary="Byte snapshot failed; falling back to chunked snapshot.",
            context={
                "error_type": "ValueError",
                "error": "bytes failed",
            },
        ),
    )
    assert plan.restore_fn is not None
    plan.restore_fn()
    assert calls == ["snapshot", "snapshot_chunked", "restore_chunked:/tmp/snap"]


def test_build_snapshot_execution_plan_records_prepare_failure() -> None:
    freed: list[object] = []

    class Adapter:
        def snapshot(self, model):  # noqa: ANN001
            raise ValueError("snapshot failed")

    plan = build_snapshot_execution_plan(
        adapter=Adapter(),
        model=object(),
        cfg_snapshot={},
        direct_reuse_loaded_model=False,
        skip_guard_metric_impact_source=None,
        choose_snapshot_mode_fn=lambda **kwargs: "bytes",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={},
        disk_usage_fn=lambda path: SimpleNamespace(free=0),
        free_model_memory_fn=freed.append,
        non_fatal_exceptions=(ValueError,),
    )

    assert len(freed) == 1
    assert plan.snapshot_enabled is False
    assert plan.diagnostics == (
        SnapshotDiagnostic(
            code="snapshot.prepare_failed",
            summary="Snapshot preparation failed; falling back to reload-per-attempt execution.",
            level="error",
            context={
                "error_type": "ValueError",
                "error": "snapshot failed",
            },
        ),
    )


def test_build_snapshot_execution_plan_propagates_runtime_errors() -> None:
    class Adapter:
        def snapshot(self, model):  # noqa: ANN001
            raise RuntimeError("snapshot runtime failed")

    with pytest.raises(RuntimeError, match="snapshot runtime failed"):
        build_snapshot_execution_plan(
            adapter=Adapter(),
            model=object(),
            cfg_snapshot={},
            direct_reuse_loaded_model=False,
            skip_guard_metric_impact_source=None,
            choose_snapshot_mode_fn=lambda **kwargs: "bytes",
            estimate_model_bytes_fn=lambda model: 0,
            psutil_module=None,
            environ={},
            disk_usage_fn=lambda path: SimpleNamespace(free=0),
            free_model_memory_fn=lambda model: None,
        )


def test_resolve_snapshot_retry_transition_reuses_loaded_model() -> None:
    transition = resolve_snapshot_retry_transition(
        skip_guard_metric_impact=True,
        profile_normalized="release",
        emitted_skip_guard_metric_impact_warning=False,
        skip_guard_metric_impact_source="config:context.run.skip_guard_metric_impact_check",
        retry_controller=None,
        model=object(),
        restore_fn=None,
        skip_model_load=False,
    )

    assert transition.skip_model_load is True
    assert transition.emitted_skip_guard_metric_impact_warning is True
    assert transition.diagnostics == (
        SnapshotDiagnostic(
            code="snapshot.overhead_check_skipped",
            summary="Guard metric impact check skipped via config policy (config:context.run.skip_guard_metric_impact_check)",
            context={"source": "config:context.run.skip_guard_metric_impact_check"},
        ),
        SnapshotDiagnostic(
            code="snapshot.restore_unavailable_reuse_loaded_model",
            summary="Snapshot restore unavailable; reusing initially loaded model for guarded execution.",
        ),
    )


def test_build_snapshot_execution_plan_uses_env_tmpdir_when_config_temp_dir_missing() -> (
    None
):
    seen: list[str] = []

    plan = build_snapshot_execution_plan(
        adapter=SimpleNamespace(),
        model=object(),
        cfg_snapshot={"temp_dir": ""},
        direct_reuse_loaded_model=False,
        skip_guard_metric_impact_source=None,
        choose_snapshot_mode_fn=lambda **kwargs: "disabled",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={"TMPDIR": "/tmp/custom-snapshot"},
        disk_usage_fn=lambda path: seen.append(path) or SimpleNamespace(free=0),
        free_model_memory_fn=lambda model: None,
    )

    assert plan.snapshot_enabled is False
    assert seen == ["/tmp/custom-snapshot"]


def test_build_snapshot_execution_plan_uses_env_tmpdir_when_snapshot_cfg_is_not_mapping() -> (
    None
):
    seen: list[str] = []

    plan = build_snapshot_execution_plan(
        adapter=SimpleNamespace(),
        model=object(),
        cfg_snapshot=SimpleNamespace(temp_dir="/ignored"),
        direct_reuse_loaded_model=False,
        skip_guard_metric_impact_source=None,
        choose_snapshot_mode_fn=lambda **kwargs: "disabled",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={"TMPDIR": "/tmp/non-mapping-snapshot"},
        disk_usage_fn=lambda path: seen.append(path) or SimpleNamespace(free=0),
        free_model_memory_fn=lambda model: None,
    )

    assert plan.snapshot_enabled is False
    assert seen == ["/tmp/non-mapping-snapshot"]
