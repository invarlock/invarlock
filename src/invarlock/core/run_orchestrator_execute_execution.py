"""Execution preparation helpers for run orchestration execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.core import run_orchestrator_execute_helpers as _execute_helpers_module
from invarlock.core.run_orchestrator_execute_helpers import (
    _coerce_float,
    _RunComponentState,
    _RunExecutionState,
)
from invarlock.core.run_orchestrator_types import (
    RunAdapterSelectedEvent,
    RunEditSelectedEvent,
    RunExecutePipelineEvent,
    RunGuardChainResolvedEvent,
    RunLoadModelOnceEvent,
    RunSnapshotModeEvent,
)


def _resolve_run_components(
    *,
    cfg: Any,
    profile: str | None,
    eval_device_override: str | None,
    pairing_schedule: dict[str, Any] | None,
    seed_bundle: dict[str, int | None],
    run_id: str,
    baseline_report_data: dict[str, Any] | None,
    model_profile: Any,
    resolved_loss_type: str,
    tiny_relax_enabled: bool,
    resolved_device: Any,
    eval_section: Any,
    run_dir: Path,
    get_registry_fn: Any,
    run_config_type: Any,
    to_serialisable_dict_fn: Any,
    cfg_value: Any,
    emit: Any,
    emit_diagnostic: Any,
    halt: Any,
) -> _RunComponentState:
    registry = get_registry_fn()
    adapter = registry.get_adapter(cfg.model.adapter)
    edit_name = getattr(getattr(cfg, "edit", None), "name", None)
    if not isinstance(edit_name, str) or not edit_name.strip():
        halt("edit_name_missing")
        raise AssertionError("unreachable after edit_name_missing halt")
    edit_name_clean = edit_name.strip()
    try:
        edit_op = registry.get_edit(edit_name_clean)
    except (AttributeError, KeyError) as exc:
        halt("unknown_edit", error=exc, edit_name=edit_name_clean)
    adapter_meta = registry.get_plugin_metadata(cfg.model.adapter, "adapters")
    try:
        from invarlock.core.adapter_provenance import extract_adapter_provenance

        adapter_meta["provenance"] = extract_adapter_provenance(
            cfg.model.adapter
        ).to_dict()
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        TypeError,
        ValueError,
    ):
        pass
    try:
        edit_meta = registry.get_plugin_metadata(edit_name_clean, "edits")
    except KeyError:
        edit_meta = {
            "name": edit_name_clean,
            "module": "edits.unknown",
            "version": "unknown",
        }
    guards: list[Any] = []
    guard_metadata: list[dict[str, Any]] = []
    guards_section = cfg_value(cfg, "guards") or {}
    guard_order = (
        guards_section.get("order", [])
        if isinstance(guards_section, dict)
        else getattr(guards_section, "order", [])
    )
    for guard_name in guard_order:
        if guard_name == "noop":
            continue
        try:
            guard = registry.get_guard(guard_name)
        except KeyError:
            emit_diagnostic(code="guard_missing", guard_name=guard_name)
            continue
        guards.append(guard)
        guard_metadata.append(registry.get_plugin_metadata(guard_name, "guards"))
    plugin_provenance = {
        "adapter": adapter_meta,
        "edit": edit_meta,
        "guards": guard_metadata,
    }
    resolve_pm_acceptance_range_fn = (
        _execute_helpers_module._resolve_pm_acceptance_range_impl
    )
    resolve_pm_drift_band_fn = _execute_helpers_module._resolve_pm_drift_band_impl
    resolve_guard_overhead_threshold_fn = (
        _execute_helpers_module._resolve_guard_overhead_threshold_impl
    )
    build_run_context_payload_fn = (
        _execute_helpers_module._build_run_context_payload_impl
    )
    if (
        resolve_pm_acceptance_range_fn is None
        or resolve_pm_drift_band_fn is None
        or resolve_guard_overhead_threshold_fn is None
        or build_run_context_payload_fn is None
    ):
        raise RuntimeError("Run execution helper dependencies are not initialized.")
    pm_acceptance_range = resolve_pm_acceptance_range_fn(cfg)
    pm_drift_band = resolve_pm_drift_band_fn(cfg)
    guard_overhead_threshold = resolve_guard_overhead_threshold_fn(cfg)
    emit(RunAdapterSelectedEvent(adapter_name=str(adapter.name)))
    run_context = build_run_context_payload_fn(
        cfg=cfg,
        profile=profile,
        pairing_schedule=pairing_schedule,
        seed_bundle=seed_bundle,
        plugin_provenance=plugin_provenance,
        run_id=run_id,
        baseline_report_data=baseline_report_data,
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        guard_overhead_threshold=guard_overhead_threshold,
        model_profile=model_profile,
        resolved_loss_type=resolved_loss_type,
        tiny_relax_enabled=bool(tiny_relax_enabled),
        to_serialisable_dict_fn=to_serialisable_dict_fn,
    )
    eval_context = run_context.get("eval")
    if isinstance(eval_context, dict) and isinstance(eval_device_override, str):
        eval_context["device_override"] = eval_device_override
    run_config = run_config_type(
        device=resolved_device,
        max_pm_ratio=_coerce_float(
            eval_section.get("max_pm_ratio")
            if isinstance(eval_section, dict)
            else getattr(eval_section, "max_pm_ratio", None)
            if eval_section is not None
            else None,
            1.5,
        ),
        event_path=run_dir / "events.jsonl",
        context=run_context,
    )
    return _RunComponentState(
        adapter=adapter,
        edit_op=edit_op,
        guards=guards,
        run_context=run_context,
        run_config=run_config,
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        guard_overhead_threshold=guard_overhead_threshold,
    )


def _prepare_execution_state(
    *,
    cfg: Any,
    model_profile: Any,
    profile_normalized: str,
    resolved_device: Any,
    run_dir: Path,
    run_id: str,
    adapter: Any,
    edit_op: Any,
    guards: list[Any],
    prefer_local_files_only: bool,
    skip_overhead: bool,
    skip_overhead_source: str | None,
    direct_reuse_loaded_model: bool,
    emitted_skip_overhead_warning: bool,
    retry_controller: Any | None,
    cfg_value: Any,
    emit: Any,
    emit_transition: Any,
    record_timed_step: Any,
    load_model_with_cfg_fn: Any,
    build_snapshot_execution_plan_fn: Any,
    resolve_snapshot_config_fn: Any,
    resolve_snapshot_retry_transition_fn: Any,
    free_model_memory_fn: Any,
    core_runner_type: Any,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> _RunExecutionState:
    emit(RunExecutePipelineEvent(guard_count=len(guards)))
    runner = core_runner_type()
    build_run_execution_config_payloads_fn = (
        _execute_helpers_module._build_run_execution_config_payloads_impl
    )
    if build_run_execution_config_payloads_fn is None:
        raise RuntimeError("Run execution config payload builder is not initialized.")
    execution_payloads = build_run_execution_config_payloads_fn(
        cfg=cfg,
        model_profile=model_profile,
    )
    auto_config = execution_payloads.auto_config
    edit_config = execution_payloads.edit_config
    emit(RunEditSelectedEvent(edit_name=str(edit_op.name)))
    emit(
        RunGuardChainResolvedEvent(
            guard_names=tuple(
                str(getattr(guard, "name", "unknown")) for guard in guards
            )
        )
    )
    model = None
    restore_fn = None
    snapshot_tmpdir = None
    snapshot_provenance: dict[str, bool] = {
        "restore_failed": False,
        "reload_path_used": False,
    }
    skip_model_load = False
    loaded_model: Any | None = None
    supports_snapshot_restore = (
        hasattr(adapter, "snapshot")
        and hasattr(adapter, "restore")
        or hasattr(adapter, "snapshot_chunked")
        and hasattr(adapter, "restore_chunked")
    )
    try:
        emit(RunLoadModelOnceEvent(model_id=str(cfg.model.id)))
        with record_timed_step("load_model"):
            model = load_model_with_cfg_fn(
                adapter,
                cfg,
                resolved_device,
                profile=profile_normalized,
                event_path=run_dir / "events.jsonl",
                warning_context={"phase": "load_model", "run_id": run_id},
                prefer_local_files_only=prefer_local_files_only,
            )
        loaded_model = model
        if direct_reuse_loaded_model:
            snapshot_plan = build_snapshot_execution_plan_fn(
                adapter=adapter,
                model=model,
                cfg_snapshot=None,
                direct_reuse_loaded_model=True,
                skip_overhead_source=skip_overhead_source,
            )
        else:
            try:
                cfg_snapshot = resolve_snapshot_config_fn(
                    cfg_value(cfg, "context") or {}
                )
            except optional_runtime_exceptions:
                cfg_snapshot = {}
            snapshot_plan = build_snapshot_execution_plan_fn(
                adapter=adapter,
                model=model,
                cfg_snapshot=cfg_snapshot,
                direct_reuse_loaded_model=False,
                skip_overhead_source=skip_overhead_source,
            )
        model = snapshot_plan.model
        restore_fn = snapshot_plan.restore_fn
        skip_model_load = snapshot_plan.skip_model_load
        snapshot_tmpdir = snapshot_plan.snapshot_tmpdir
        snapshot_provenance = snapshot_plan.snapshot_provenance
        emitted_skip_overhead_warning = snapshot_plan.emitted_skip_overhead_warning
        if snapshot_plan.snapshot_enabled is not None:
            emit(RunSnapshotModeEvent(enabled=bool(snapshot_plan.snapshot_enabled)))
        for diagnostic in snapshot_plan.diagnostics:
            emit_transition("snapshot_plan", diagnostic)
    except optional_runtime_exceptions:
        free_model_memory_fn(model)
        model = None
        restore_fn = None
    snapshot_retry_transition = resolve_snapshot_retry_transition_fn(
        skip_overhead=skip_overhead,
        profile_normalized=profile_normalized,
        emitted_skip_overhead_warning=emitted_skip_overhead_warning,
        skip_overhead_source=skip_overhead_source,
        retry_controller=retry_controller,
        model=model,
        restore_fn=restore_fn,
        skip_model_load=skip_model_load,
    )
    skip_model_load = snapshot_retry_transition.skip_model_load
    emitted_skip_overhead_warning = (
        snapshot_retry_transition.emitted_skip_overhead_warning
    )
    for diagnostic in snapshot_retry_transition.diagnostics:
        emit_transition("snapshot_retry", diagnostic)
    if (
        loaded_model is not None
        and retry_controller is None
        and restore_fn is None
        and not direct_reuse_loaded_model
        and profile_normalized not in {"ci", "release"}
        and supports_snapshot_restore
    ):
        if model is None:
            model = loaded_model
        if model is loaded_model:
            skip_model_load = True
    return _RunExecutionState(
        runner=runner,
        auto_config=auto_config,
        edit_config=edit_config,
        model=model,
        restore_fn=restore_fn,
        snapshot_tmpdir=snapshot_tmpdir,
        snapshot_provenance=snapshot_provenance,
        skip_model_load=skip_model_load,
        emitted_skip_overhead_warning=emitted_skip_overhead_warning,
    )
