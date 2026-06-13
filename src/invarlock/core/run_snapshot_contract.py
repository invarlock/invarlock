from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

_NON_FATAL_SNAPSHOT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    OSError,
)


def resolve_snapshot_config(
    context: object | None,
    *,
    to_serialisable_dict_fn: Callable[[object], Any],
) -> dict[str, Any]:
    """Extract a plain snapshot policy mapping from run context."""
    try:
        context_map = to_serialisable_dict_fn(context or {})
    except (AttributeError, TypeError, ValueError):
        return {}
    if not isinstance(context_map, dict):
        return {}
    try:
        snapshot_map = to_serialisable_dict_fn(context_map.get("snapshot", {}))
    except (AttributeError, TypeError, ValueError):
        return {}
    return snapshot_map if isinstance(snapshot_map, dict) else {}


def estimate_model_bytes(model: Any) -> int:
    """Best-effort estimate of parameter + buffer footprint."""
    total = 0
    try:
        for _, param in getattr(model, "named_parameters", lambda: [])():
            try:
                total += int(param.element_size() * param.nelement())
            except (AttributeError, TypeError, ValueError):
                pass
        for _, buffer in getattr(model, "named_buffers", lambda: [])():
            try:
                total += int(buffer.element_size() * buffer.nelement())
            except (AttributeError, TypeError, ValueError):
                pass
    except (AttributeError, TypeError):
        return 0
    return total


def _parse_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed


def _requested_snapshot_mode(
    requested_mode: str,
    *,
    supports_bytes: bool,
    supports_chunked: bool,
) -> str:
    if requested_mode == "bytes" and supports_bytes:
        return "bytes"
    if requested_mode == "chunked" and supports_chunked:
        return "chunked"
    if supports_bytes:
        return "bytes"
    if supports_chunked:
        return "chunked"
    return "reload"


def choose_snapshot_mode(
    *,
    snapshot_config: Mapping[str, Any] | None,
    env_mode: str | None,
    supports_bytes: bool,
    supports_chunked: bool,
    estimated_model_mb: float,
    available_ram_mb: float,
    disk_free_mb: float,
    env_ram_fraction: str | None = None,
    env_threshold_mb: str | None = None,
) -> str:
    """Choose bytes/chunked/reload snapshot mode deterministically."""
    cfg_snapshot = dict(snapshot_config or {})
    cfg_mode = str(cfg_snapshot.get("mode", "")).lower()
    mode_env = str(env_mode or "auto").lower()

    if cfg_mode in {"bytes", "chunked"}:
        return _requested_snapshot_mode(
            cfg_mode,
            supports_bytes=supports_bytes,
            supports_chunked=supports_chunked,
        )
    if mode_env in {"bytes", "chunked"}:
        return _requested_snapshot_mode(
            mode_env,
            supports_bytes=supports_bytes,
            supports_chunked=supports_chunked,
        )

    frac = 0.4
    if cfg_snapshot.get("ram_fraction") is not None:
        frac = _parse_float(cfg_snapshot.get("ram_fraction"), frac)
    elif env_ram_fraction is not None:
        frac = _parse_float(env_ram_fraction, frac)
    frac = max(0.0, min(frac, 1.0))

    if cfg_snapshot.get("threshold_mb") is not None:
        absolute_threshold_mb = _parse_float(cfg_snapshot.get("threshold_mb"), 768.0)
    else:
        absolute_threshold_mb = _parse_float(env_threshold_mb, 768.0)
    absolute_threshold_mb = max(0.0, float(absolute_threshold_mb))

    if available_ram_mb > 0:
        ram_threshold_mb = float(available_ram_mb) * frac
        threshold_mb = (
            min(ram_threshold_mb, absolute_threshold_mb)
            if absolute_threshold_mb > 0
            else ram_threshold_mb
        )
    else:
        threshold_mb = absolute_threshold_mb

    margin = 1.2
    if cfg_snapshot.get("disk_free_margin_ratio") is not None:
        margin = _parse_float(cfg_snapshot.get("disk_free_margin_ratio"), margin)

    disk_has_room = disk_free_mb <= 0.0 or estimated_model_mb * margin <= disk_free_mb

    if supports_chunked and estimated_model_mb >= threshold_mb and disk_has_room:
        return "chunked"

    if supports_bytes:
        if (
            supports_chunked
            and available_ram_mb > 0
            and estimated_model_mb >= max(64.0, float(available_ram_mb) * 0.8)
            and disk_has_room
        ):
            return "chunked"
        return "bytes"

    if supports_chunked:
        return "chunked"

    return "reload"


@dataclass(frozen=True)
class SnapshotDiagnostic:
    code: str
    summary: str
    level: str = "warning"
    context: dict[str, Any] = field(default_factory=dict)


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
                    summary=f"Overhead check skipped via config policy{source_note}",
                    context={"source": skip_overhead_source},
                ),
                SnapshotDiagnostic(
                    code="snapshot.loaded_model_reused",
                    summary="Reusing initially loaded model for guarded execution.",
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
    bytes_fallback_exceptions = (
        non_fatal_exceptions
        if RuntimeError in non_fatal_exceptions
        else non_fatal_exceptions + (RuntimeError,)
    )

    try:
        if mode == "chunked":
            snapshot_tmpdir = adapter.snapshot_chunked(model)

            def _restore() -> None:
                adapter.restore_chunked(model, snapshot_tmpdir)

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
                base_blob = adapter.snapshot(model)
            except bytes_fallback_exceptions as exc:
                if not supports_chunked:
                    raise
                snapshot_tmpdir = adapter.snapshot_chunked(model)

                def _restore_fallback_chunked() -> None:
                    adapter.restore_chunked(model, snapshot_tmpdir)

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
                            summary="Byte snapshot failed; falling back to chunked snapshot.",
                            context={
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        ),
                    ),
                )

            def _restore_bytes() -> None:
                adapter.restore(model, base_blob)

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
                    summary="Snapshot preparation failed; falling back to reload-per-attempt execution.",
                    level="error",
                    context={
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
                    summary=f"Overhead check skipped via config policy{source_note}",
                    context={"source": skip_overhead_source},
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
                    summary="Snapshot restore unavailable; reusing initially loaded model for guarded execution.",
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
    "choose_snapshot_mode",
    "estimate_model_bytes",
    "resolve_snapshot_config",
    "resolve_snapshot_retry_transition",
]
