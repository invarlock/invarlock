from __future__ import annotations

from types import SimpleNamespace

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
        skip_overhead_source="config:context.run.skip_overhead_check",
        choose_snapshot_mode_fn=lambda **kwargs: "disabled",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={},
        disk_usage_fn=lambda path: SimpleNamespace(free=0),
        free_model_memory_fn=lambda model: None,
    )

    assert plan.skip_model_load is True
    assert plan.snapshot_enabled is None
    assert plan.emitted_skip_overhead_warning is True
    assert plan.diagnostics == (
        SnapshotDiagnostic(
            code="snapshot.overhead_check_skipped",
            message="Overhead check skipped via config policy (config:context.run.skip_overhead_check)",
            details={"source": "config:context.run.skip_overhead_check"},
        ),
        SnapshotDiagnostic(
            code="snapshot.loaded_model_reused",
            message="Reusing initially loaded model for guarded execution.",
        ),
    )


def test_build_snapshot_execution_plan_bytes_falls_back_to_chunked() -> None:
    calls: list[str] = []

    class Adapter:
        def snapshot(self, model):  # noqa: ANN001
            calls.append("snapshot")
            raise RuntimeError("bytes failed")

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
        skip_overhead_source=None,
        choose_snapshot_mode_fn=lambda **kwargs: "bytes",
        estimate_model_bytes_fn=lambda model: 0,
        psutil_module=None,
        environ={},
        disk_usage_fn=lambda path: SimpleNamespace(free=0),
        free_model_memory_fn=lambda model: None,
    )

    assert plan.snapshot_enabled is True
    assert plan.restore_fn is not None
    plan.restore_fn()
    assert calls == ["snapshot", "snapshot_chunked", "restore_chunked:/tmp/snap"]


def test_resolve_snapshot_retry_transition_reuses_loaded_model() -> None:
    transition = resolve_snapshot_retry_transition(
        skip_overhead=True,
        profile_normalized="release",
        emitted_skip_overhead_warning=False,
        skip_overhead_source="config:context.run.skip_overhead_check",
        retry_controller=None,
        model=object(),
        restore_fn=None,
        skip_model_load=False,
    )

    assert transition.skip_model_load is True
    assert transition.emitted_skip_overhead_warning is True
    assert transition.diagnostics == (
        SnapshotDiagnostic(
            code="snapshot.overhead_check_skipped",
            message="Overhead check skipped via config policy (config:context.run.skip_overhead_check)",
            details={"source": "config:context.run.skip_overhead_check"},
        ),
        SnapshotDiagnostic(
            code="snapshot.restore_unavailable_reuse_loaded_model",
            message="Snapshot restore unavailable; reusing initially loaded model for guarded execution.",
        ),
    )
