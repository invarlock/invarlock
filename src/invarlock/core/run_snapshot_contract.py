from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any

_NON_FATAL_SNAPSHOT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)


@dataclass(frozen=True)
class SnapshotDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotExecutionPlan:
    model: Any
    restore_fn: Any | None
    skip_model_load: bool
    snapshot_tmpdir: str | None
    snapshot_provenance: dict[str, bool]
    emitted_skip_overhead_warning: bool
    snapshot_enabled: bool | None
    diagnostics: tuple[SnapshotDiagnostic, ...]


def build_snapshot_execution_plan(
    *,
    adapter: Any,
    model: Any,
    cfg_snapshot: dict[str, Any] | None,
    direct_reuse_loaded_model: bool,
    skip_overhead_source: str | None,
    choose_snapshot_mode_fn: Any,
    estimate_model_bytes_fn: Any,
    psutil_module: Any | None,
    environ: dict[str, str],
    tempfile_gettempdir_fn: Any = tempfile.gettempdir,
    disk_usage_fn: Any,
    free_model_memory_fn: Any,
    non_fatal_exceptions: tuple[
        type[BaseException], ...
    ] = _NON_FATAL_SNAPSHOT_EXCEPTIONS,
) -> SnapshotExecutionPlan:
    snapshot_provenance = {
        "restore_failed": False,
        "reload_path_used": False,
    }
    if direct_reuse_loaded_model:
        source_note = f" ({skip_overhead_source})" if skip_overhead_source else ""
        return SnapshotExecutionPlan(
            model=model,
            restore_fn=None,
            skip_model_load=True,
            snapshot_tmpdir=None,
            snapshot_provenance=snapshot_provenance,
            emitted_skip_overhead_warning=True,
            snapshot_enabled=None,
            diagnostics=(
                SnapshotDiagnostic(
                    code="snapshot.overhead_check_skipped",
                    message=f"Overhead check skipped via config policy{source_note}",
                    details={"source": skip_overhead_source},
                ),
                SnapshotDiagnostic(
                    code="snapshot.loaded_model_reused",
                    message="Reusing initially loaded model for guarded execution.",
                ),
            ),
        )

    supports_chunked = hasattr(adapter, "snapshot_chunked") and hasattr(
        adapter, "restore_chunked"
    )
    supports_bytes = hasattr(adapter, "snapshot") and hasattr(adapter, "restore")
    est_mb = estimate_model_bytes_fn(model) / (1024.0 * 1024.0)
    try:
        if psutil_module is None:
            raise AttributeError("psutil unavailable")
        ram = psutil_module.virtual_memory()
        avail_mb = float(getattr(ram, "available", 0)) / (1024.0 * 1024.0)
    except (AttributeError, RuntimeError, OSError, TypeError, ValueError):
        avail_mb = 0.0
    try:
        tmpdir = None
        if isinstance(cfg_snapshot, dict):
            tmpdir = cfg_snapshot.get("temp_dir") or None
        if not tmpdir:
            tmpdir = (
                environ.get("TMPDIR") or environ.get("TMP") or tempfile_gettempdir_fn()
            )
        du = disk_usage_fn(tmpdir)
        free_mb = float(du.free) / (1024.0 * 1024.0)
    except (OSError, TypeError, ValueError):
        free_mb = 0.0

    mode = choose_snapshot_mode_fn(
        snapshot_config=cfg_snapshot or {},
        env_mode=environ.get("INVARLOCK_SNAPSHOT_MODE", "auto"),
        supports_bytes=supports_bytes,
        supports_chunked=supports_chunked,
        estimated_model_mb=est_mb,
        available_ram_mb=avail_mb,
        disk_free_mb=free_mb,
        env_ram_fraction=environ.get("INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION"),
        env_threshold_mb=environ.get("INVARLOCK_SNAPSHOT_THRESHOLD_MB"),
    )

    try:
        if mode == "chunked":
            snapshot_tmpdir = adapter.snapshot_chunked(model)  # type: ignore[attr-defined]

            def _restore() -> None:
                adapter.restore_chunked(model, snapshot_tmpdir)  # type: ignore[attr-defined]

            return SnapshotExecutionPlan(
                model=model,
                restore_fn=_restore,
                skip_model_load=False,
                snapshot_tmpdir=snapshot_tmpdir,
                snapshot_provenance=snapshot_provenance,
                emitted_skip_overhead_warning=False,
                snapshot_enabled=True,
                diagnostics=(),
            )
        if mode == "bytes":
            try:
                base_blob = adapter.snapshot(model)  # type: ignore[attr-defined]
            except non_fatal_exceptions as exc:
                if not supports_chunked:
                    raise
                snapshot_tmpdir = adapter.snapshot_chunked(model)  # type: ignore[attr-defined]

                def _restore_fallback_chunked() -> None:
                    adapter.restore_chunked(model, snapshot_tmpdir)  # type: ignore[attr-defined]

                return SnapshotExecutionPlan(
                    model=model,
                    restore_fn=_restore_fallback_chunked,
                    skip_model_load=False,
                    snapshot_tmpdir=snapshot_tmpdir,
                    snapshot_provenance=snapshot_provenance,
                    emitted_skip_overhead_warning=False,
                    snapshot_enabled=True,
                    diagnostics=(
                        SnapshotDiagnostic(
                            code="snapshot.bytes_failed_chunked_fallback",
                            message="Byte snapshot failed; falling back to chunked snapshot.",
                            details={
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        ),
                    ),
                )

            def _restore_bytes() -> None:
                adapter.restore(model, base_blob)  # type: ignore[attr-defined]

            return SnapshotExecutionPlan(
                model=model,
                restore_fn=_restore_bytes,
                skip_model_load=False,
                snapshot_tmpdir=None,
                snapshot_provenance=snapshot_provenance,
                emitted_skip_overhead_warning=False,
                snapshot_enabled=True,
                diagnostics=(),
            )

        free_model_memory_fn(model)
        return SnapshotExecutionPlan(
            model=None,
            restore_fn=None,
            skip_model_load=False,
            snapshot_tmpdir=None,
            snapshot_provenance=snapshot_provenance,
            emitted_skip_overhead_warning=False,
            snapshot_enabled=False,
            diagnostics=(),
        )
    except non_fatal_exceptions as exc:
        free_model_memory_fn(model)
        return SnapshotExecutionPlan(
            model=None,
            restore_fn=None,
            skip_model_load=False,
            snapshot_tmpdir=None,
            snapshot_provenance=snapshot_provenance,
            emitted_skip_overhead_warning=False,
            snapshot_enabled=False,
            diagnostics=(
                SnapshotDiagnostic(
                    code="snapshot.prepare_failed",
                    message="Snapshot preparation failed; falling back to reload-per-attempt execution.",
                    severity="error",
                    details={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            ),
        )


@dataclass(frozen=True)
class SnapshotRetryTransition:
    skip_model_load: bool
    emitted_skip_overhead_warning: bool
    diagnostics: tuple[SnapshotDiagnostic, ...]


def resolve_snapshot_retry_transition(
    *,
    skip_overhead: bool,
    profile_normalized: str | None,
    emitted_skip_overhead_warning: bool,
    skip_overhead_source: str | None,
    retry_controller: Any,
    model: Any,
    restore_fn: Any | None,
    skip_model_load: bool,
) -> SnapshotRetryTransition:
    diagnostics: list[SnapshotDiagnostic] = []
    updated_skip_model_load = skip_model_load
    warned = emitted_skip_overhead_warning

    if skip_overhead and profile_normalized in {"ci", "release"}:
        if not warned:
            source_note = f" ({skip_overhead_source})" if skip_overhead_source else ""
            diagnostics.append(
                SnapshotDiagnostic(
                    code="snapshot.overhead_check_skipped",
                    message=f"Overhead check skipped via config policy{source_note}",
                    details={"source": skip_overhead_source},
                )
            )
            warned = True
        if (
            retry_controller is None
            and model is not None
            and restore_fn is None
            and not updated_skip_model_load
        ):
            updated_skip_model_load = True
            diagnostics.append(
                SnapshotDiagnostic(
                    code="snapshot.restore_unavailable_reuse_loaded_model",
                    message="Snapshot restore unavailable; reusing initially loaded model for guarded execution.",
                )
            )

    return SnapshotRetryTransition(
        skip_model_load=updated_skip_model_load,
        emitted_skip_overhead_warning=warned,
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "SnapshotDiagnostic",
    "SnapshotExecutionPlan",
    "SnapshotRetryTransition",
    "build_snapshot_execution_plan",
    "resolve_snapshot_retry_transition",
]
