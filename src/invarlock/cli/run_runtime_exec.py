"""Runtime-only execution helpers for config-driven run commands."""

from __future__ import annotations

import copy
import inspect
import math
import os
import shutil
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from invarlock.cli.run_config import extract_model_load_kwargs
from invarlock.cli.run_runtime import free_model_memory, get_psutil
from invarlock.cli.run_shell_output import _event
from invarlock.cli.run_warning_filters import suppress_noisy_warnings
from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_policy import GUARD_OVERHEAD_THRESHOLD
from invarlock.core.run_snapshot_contract import (
    build_snapshot_execution_plan as _build_snapshot_execution_plan_impl,
)
from invarlock.core.run_snapshot_policy import (
    choose_snapshot_mode as _choose_snapshot_mode_impl,
)
from invarlock.core.run_snapshot_policy import (
    estimate_model_bytes as _estimate_model_bytes_impl,
)


class SnapshotRestoreFailed(RuntimeError):
    """Internal signal for snapshot restore failures during retries."""


def build_snapshot_execution_plan(
    *,
    adapter: Any,
    model: Any,
    cfg_snapshot: dict[str, Any] | None,
    direct_reuse_loaded_model: bool,
    skip_overhead_source: str | None,
) -> Any:
    return _build_snapshot_execution_plan_impl(
        adapter=adapter,
        model=model,
        cfg_snapshot=cfg_snapshot,
        direct_reuse_loaded_model=direct_reuse_loaded_model,
        skip_overhead_source=skip_overhead_source,
        choose_snapshot_mode_fn=_choose_snapshot_mode_impl,
        estimate_model_bytes_fn=_estimate_model_bytes_impl,
        psutil_module=get_psutil(),
        environ=os.environ,
        disk_usage_fn=shutil.disk_usage,
        free_model_memory_fn=free_model_memory,
    )


def load_model_with_cfg(
    adapter: Any,
    cfg: Any,
    device: str,
    *,
    profile: str | None = None,
    event_path: Any | None = None,
    warning_context: dict[str, Any] | None = None,
    prefer_local_files_only: bool = False,
) -> Any:
    """Load a model with config-provided kwargs, filtering for strict adapters."""
    try:
        model_id = cfg.model.id
    except Exception:
        try:
            model_id = (cfg.model_dump().get("model") or {}).get("id")
        except Exception:
            model_id = None
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Missing model.id in config")

    extra = extract_model_load_kwargs(cfg, invarlock_error_cls=InvarlockError)
    with suppress_noisy_warnings(
        profile,
        event_path=event_path,
        context=warning_context,
    ):
        strict_accepts_local_files_only = False
        try:
            sig = inspect.signature(adapter.load_model)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_var_kw:
                allowed = dict(extra)
                if prefer_local_files_only:
                    allowed["prefer_local_files_only"] = True
                return adapter.load_model(model_id, device=device, **allowed)
            allowed = {k: v for k, v in extra.items() if k in sig.parameters}
            strict_accepts_local_files_only = (
                "prefer_local_files_only" in sig.parameters
            )
            if prefer_local_files_only and strict_accepts_local_files_only:
                allowed["prefer_local_files_only"] = True
            if allowed:
                return adapter.load_model(model_id, device=device, **allowed)
        except Exception:
            pass
        if prefer_local_files_only and strict_accepts_local_files_only:
            return adapter.load_model(
                model_id, device=device, prefer_local_files_only=True
            )
        return adapter.load_model(model_id, device=device)


def init_retry_controller(
    *,
    until_pass: bool,
    max_attempts: int,
    timeout: int | None,
    baseline: str | None,
    console: Console,
) -> Any:
    """Initialize RetryController with consistent shell events."""
    retry_controller = None
    if until_pass:
        from invarlock.core.retry import RetryController

        retry_controller = RetryController(
            max_attempts=max_attempts, timeout=timeout, verbose=True
        )
        _event(
            console,
            "INIT",
            f"Retry mode enabled: max {max_attempts} attempts",
            emoji="🔄",
        )
        if baseline:
            _event(console, "DATA", f"Using baseline: {baseline}", emoji="📋")
    elif baseline:
        _event(console, "DATA", f"Using baseline: {baseline}", emoji="📋")
    return retry_controller


def run_bare_control(
    *,
    adapter: Any,
    edit_op: Any,
    cfg: Any,
    model: Any,
    run_config: Any,
    calibration_data: list[Any],
    auto_config: Any,
    edit_config: Any,
    preview_count: int,
    final_count: int,
    seed_bundle: dict[str, int | None],
    resolved_device: str,
    restore_fn: Any | None,
    console: Console,
    resolved_loss_type: str,
    overhead_threshold: float = GUARD_OVERHEAD_THRESHOLD,
    profile_normalized: str | None = None,
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
    prefer_local_files_only: bool = False,
) -> dict[str, Any] | None:
    """Execute the bare-control run for overhead estimation and return payload."""
    from invarlock.cli.overhead_utils import _extract_pm_snapshot_for_overhead
    from invarlock.core.api import EditRuntime
    from invarlock.core.runner import CoreRunner
    from invarlock.model_utils import set_seed

    _event(
        console,
        "EXEC",
        "Running bare control (guards disabled) for overhead check",
        emoji="🧪",
        profile=profile_normalized,
    )
    set_seed(seed_bundle["python"])  # type: ignore[arg-type]

    bare_runner = CoreRunner()
    bare_config = copy.deepcopy(run_config)
    bare_config.event_path = None
    bare_context = copy.deepcopy(run_config.context)
    bare_context.setdefault("validation", {})["guard_overhead_mode"] = "bare"
    bare_config.context = bare_context
    edit_runtime = EditRuntime()

    private_model_loaded = False
    bare_target_model = None
    try:
        if restore_fn and model is not None:
            try:
                restore_fn()
            except Exception as exc:
                raise SnapshotRestoreFailed(str(exc)) from exc
            bare_target_model = model
        elif skip_model_load:
            bare_target_model = model or SimpleNamespace(name="bare_stub_model")
        else:
            bare_target_model = load_model_with_cfg(
                adapter,
                cfg,
                resolved_device,
                profile=profile_normalized,
                prefer_local_files_only=prefer_local_files_only,
            )
            private_model_loaded = True
            if snapshot_provenance is not None:
                snapshot_provenance["reload_path_used"] = True

        with suppress_noisy_warnings(
            profile_normalized,
            event_path=getattr(run_config, "event_path", None),
            context={"phase": "guard_overhead_bare"},
        ):
            bare_report = bare_runner.execute(
                model=bare_target_model,
                adapter=adapter,
                edit=edit_op,
                guards=[],
                config=bare_config,
                calibration_data=calibration_data,
                auto_config=auto_config,
                edit_config=dict(edit_config or {}),
                edit_runtime=edit_runtime,
                preview_n=preview_count,
                final_n=final_count,
            )
    finally:
        if private_model_loaded:
            free_model_memory(bare_target_model)

    bare_ppl_final = None
    bare_ppl_preview = None
    if hasattr(bare_report, "metrics") and bare_report.metrics:
        bare_pm = bare_report.metrics.get("primary_metric", {})
        bare_ppl_final = bare_pm.get("final") if isinstance(bare_pm, dict) else None
        bare_ppl_preview = bare_pm.get("preview") if isinstance(bare_pm, dict) else None

    if profile_normalized in {"ci", "release"}:

        def _finite(x: Any) -> bool:
            try:
                return isinstance(x, int | float) and math.isfinite(float(x))
            except Exception:
                return False

        if not (_finite(bare_ppl_preview) and _finite(bare_ppl_final)):
            _event(
                console,
                "WARN",
                "Primary metric non-finite during bare control; continuing with diagnostics.",
                emoji="⚠️",
                profile=profile_normalized,
            )

    payload: dict[str, Any] = {
        "overhead_threshold": float(overhead_threshold),
        "messages": [],
        "warnings": [],
        "errors": [],
        "checks": {},
        "source": f"{profile_normalized or 'ci'}_profile",
        "mode": "bare",
    }

    if getattr(bare_report, "status", "").lower() not in {"success", "completed", "ok"}:
        payload["warnings"].append(
            f"Bare run status: {getattr(bare_report, 'status', 'unknown')}"
        )

    try:
        lk = str(resolved_loss_type or "causal").lower()
        if lk == "mlm":
            pm_kind_bare = "ppl_mlm"
        elif lk in {"seq2seq", "s2s", "t5"}:
            pm_kind_bare = "ppl_seq2seq"
        else:
            pm_kind_bare = "ppl_causal"
        pm_bare = _extract_pm_snapshot_for_overhead(bare_report, kind=pm_kind_bare)
        if isinstance(pm_bare, dict) and pm_bare:
            payload["bare_report"] = {"metrics": {"primary_metric": pm_bare}}
    except Exception:
        pass

    set_seed(seed_bundle["python"])  # type: ignore[arg-type]
    return payload


def execute_guarded_run(
    *,
    runner: Any,
    adapter: Any,
    model: Any,
    cfg: Any,
    edit_op: Any,
    run_config: Any,
    guards: list[Any],
    calibration_data: list[Any],
    auto_config: Any,
    edit_config: Any,
    preview_count: int,
    final_count: int,
    restore_fn: Any | None,
    resolved_device: str,
    profile_normalized: str | None = None,
    console: Console,
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
    prefer_local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Restore or load model and execute the guarded CoreRunner."""
    from invarlock.core.api import EditRuntime

    if restore_fn and model is not None:
        try:
            restore_fn()
        except Exception as exc:
            raise SnapshotRestoreFailed(str(exc)) from exc
    elif skip_model_load:
        model = model or SimpleNamespace(name="guarded_stub_model")
    else:
        _event(
            console,
            "INIT",
            f"Loading model: {cfg.model.id} (attempt 1)",
            emoji="🔧",
            profile=profile_normalized,
        )
        warning_context: dict[str, Any] = {"phase": "load_model"}
        try:
            if hasattr(run_config, "context") and isinstance(run_config.context, dict):
                rid = run_config.context.get("run_id")
                if isinstance(rid, str) and rid:
                    warning_context["run_id"] = rid
        except Exception:
            pass
        model = load_model_with_cfg(
            adapter,
            cfg,
            resolved_device,
            profile=profile_normalized,
            event_path=getattr(run_config, "event_path", None),
            warning_context=warning_context,
            prefer_local_files_only=prefer_local_files_only,
        )
        if snapshot_provenance is not None:
            snapshot_provenance["reload_path_used"] = True

    edit_runtime = EditRuntime()

    with suppress_noisy_warnings(
        profile_normalized,
        event_path=getattr(run_config, "event_path", None),
        context={"phase": "core_runner_execute"},
    ):
        core_report = runner.execute(
            model=model,
            adapter=adapter,
            edit=edit_op,
            guards=guards,
            config=run_config,
            calibration_data=calibration_data,
            auto_config=auto_config,
            edit_config=dict(edit_config or {}),
            edit_runtime=edit_runtime,
            preview_n=preview_count,
            final_n=final_count,
        )
    return core_report, model


__all__ = [
    "GUARD_OVERHEAD_THRESHOLD",
    "SnapshotRestoreFailed",
    "build_snapshot_execution_plan",
    "execute_guarded_run",
    "init_retry_controller",
    "load_model_with_cfg",
    "run_bare_control",
    "suppress_noisy_warnings",
]
