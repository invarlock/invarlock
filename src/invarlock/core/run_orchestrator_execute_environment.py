"""Environment preparation helpers for run orchestration execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from invarlock.core import run_orchestrator_execute_helpers as _execute_helpers_module
from invarlock.core.exceptions import ConfigError
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _RunEnvironmentState,
)
from invarlock.core.run_orchestrator_types import (
    RunConfigLoadedEvent,
    RunConfigLoadingEvent,
    RunDeviceResolvedEvent,
    RunOutputDirectoryReadyEvent,
    RunPipelineStartedEvent,
)
from invarlock.core.run_policy import (
    should_measure_overhead as _should_measure_overhead_default,
)

from .run_orchestrator_execute_seed import (
    _apply_determinism_preset,
    _load_baseline_evidence_state,
    _resolve_loss_seed_and_determinism_state,
)


def _validate_removed_edit_keys(
    cfg: Any,
    *,
    config_value_exceptions: tuple[type[BaseException], ...],
) -> None:
    edit_payload: dict[str, Any] = {}
    try:
        cfg_dump = cfg.model_dump()
    except (AttributeError, TypeError, ValueError):
        cfg_dump = None
    if isinstance(cfg_dump, dict):
        edit_section = cfg_dump.get("edit")
        if isinstance(edit_section, dict):
            edit_payload.update(edit_section)
    try:
        edit_obj = getattr(cfg, "edit", None)
    except config_value_exceptions:
        edit_obj = None
    edit_dict = getattr(edit_obj, "__dict__", None)
    if isinstance(edit_dict, dict):
        edit_payload.update(edit_dict)
    removed_edit_kind = edit_payload.get("kind")
    if removed_edit_kind is not None:
        raise ConfigError(
            code="E007",
            message=(
                "CONFIG-KEY-REMOVED: edit.kind. Use edit.name with a canonical "
                "edit plugin name."
            ),
            details={"removed_keys": ["edit.kind"]},
        )
    removed_edit_parameters = edit_payload.get("parameters")
    if removed_edit_parameters is not None:
        raise ConfigError(
            code="E007",
            message="CONFIG-KEY-REMOVED: edit.parameters. Use edit.plan.",
            details={"removed_keys": ["edit.parameters"]},
        )


def _prepare_run_environment(
    *,
    config: str,
    profile: str | None,
    profile_normalized: str,
    edit: str | None,
    tier: str | None,
    probes: int | None,
    device: str | None,
    out: str,
    until_pass: bool,
    max_attempts: int,
    timeout: float | None,
    baseline: str | None,
    determinism_mode: str | None,
    determinism_warn_only: bool,
    prepare_config_for_run_fn: Any,
    detect_model_profile_fn: Any,
    resolve_device_and_output_fn: Any,
    init_retry_controller_fn: Any,
    load_baseline_pairing_evidence_fn: Any,
    safe_int_fn: Any,
    optional_torch: Any,
    require_torch: Any,
    cfg_value: Any,
    emit: RunEventEmitter,
    emit_diagnostic: Any,
    config_value_exceptions: tuple[type[BaseException], ...],
    numeric_exceptions: tuple[type[BaseException], ...],
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> _RunEnvironmentState:
    require_torch()
    emit(RunConfigLoadingEvent(config_path=config))
    cfg = prepare_config_for_run_fn(
        config_path=config,
        profile=profile,
        edit=edit,
        tier=tier,
        probes=probes,
    )
    emit(RunConfigLoadedEvent())
    _validate_removed_edit_keys(
        cfg,
        config_value_exceptions=config_value_exceptions,
    )
    adapter_name = str(getattr(cfg.model, "adapter", "")).lower()
    model_id_raw = str(getattr(cfg.model, "id", ""))
    model_profile = detect_model_profile_fn(model_id=model_id_raw, adapter=adapter_name)
    tokenizer_hash: str | None = None
    tokenizer: Any | None = None
    loss_seed_state = _resolve_loss_seed_and_determinism_state(
        cfg,
        model_profile=model_profile,
        profile_normalized=profile_normalized,
        determinism_mode=determinism_mode,
        determinism_warn_only=determinism_warn_only,
        optional_torch=optional_torch,
        emit=emit,
        cfg_value=cfg_value,
        config_value_exceptions=config_value_exceptions,
        numeric_exceptions=numeric_exceptions,
        optional_runtime_exceptions=optional_runtime_exceptions,
    )
    resolved_device, output_dir = resolve_device_and_output_fn(
        cfg,
        device=device,
        out=out,
    )
    emit(
        RunDeviceResolvedEvent(
            requested_device=str(device or "auto"),
            resolved_device=str(resolved_device),
        )
    )
    seed_bundle = dict(loss_seed_state.seed_bundle)
    profile_label = profile_normalized or None
    determinism_meta = _apply_determinism_preset(
        profile_label=profile_label,
        resolved_device=resolved_device,
        seed_bundle=seed_bundle,
        seed_value=loss_seed_state.seed_value,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{output_dir.name}-{timestamp}" if output_dir.name else timestamp
    emit(RunOutputDirectoryReadyEvent(run_dir=str(run_dir), run_id=run_id))
    retry_controller = init_retry_controller_fn(
        until_pass=until_pass,
        max_attempts=max_attempts,
        timeout=timeout,
        baseline=baseline,
    )
    should_measure_overhead_fn = getattr(
        _execute_helpers_module,
        "_should_measure_overhead_impl",
        _should_measure_overhead_default,
    )
    (
        measure_guard_overhead,
        skip_overhead,
        skip_overhead_source,
    ) = should_measure_overhead_fn(profile_normalized, cfg)
    direct_reuse_loaded_model = (
        skip_overhead
        and profile_normalized in {"ci", "release"}
        and retry_controller is None
    )
    baseline_report_data, pairing_schedule, tokenizer_hash = (
        _load_baseline_evidence_state(
            baseline=baseline,
            profile_normalized=profile_normalized,
            tokenizer_hash=tokenizer_hash,
            load_baseline_pairing_evidence_fn=load_baseline_pairing_evidence_fn,
            emit=emit,
            emit_diagnostic=emit_diagnostic,
        )
    )
    requested_preview = safe_int_fn(getattr(cfg.dataset, "preview_n", 0), 0)
    requested_final = safe_int_fn(getattr(cfg.dataset, "final_n", 0), 0)
    effective_preview = requested_preview
    effective_final = requested_final
    preview_count = effective_preview
    final_count = effective_final
    try:
        resolved_split = getattr(cfg.dataset, "split", None) or "validation"
    except (AttributeError, TypeError):
        resolved_split = "validation"
    emit(RunPipelineStartedEvent())
    return _RunEnvironmentState(
        cfg=cfg,
        model_profile=model_profile,
        eval_section=loss_seed_state.eval_section,
        resolved_loss_type=loss_seed_state.resolved_loss_type,
        use_mlm=loss_seed_state.use_mlm,
        mask_prob=loss_seed_state.mask_prob,
        mask_seed=loss_seed_state.mask_seed,
        random_token_prob=loss_seed_state.random_token_prob,
        original_token_prob=loss_seed_state.original_token_prob,
        seed_value=loss_seed_state.seed_value,
        seed_bundle=seed_bundle,
        profile_label=profile_label,
        resolved_device=resolved_device,
        output_dir=output_dir,
        determinism_meta=determinism_meta,
        run_dir=run_dir,
        run_id=run_id,
        retry_controller=retry_controller,
        measure_guard_overhead=measure_guard_overhead,
        skip_overhead=skip_overhead,
        skip_overhead_source=skip_overhead_source,
        direct_reuse_loaded_model=direct_reuse_loaded_model,
        emitted_skip_overhead_warning=False,
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        baseline_report_data=baseline_report_data,
        pairing_schedule=pairing_schedule,
        requested_preview=requested_preview,
        requested_final=requested_final,
        effective_preview=effective_preview,
        effective_final=effective_final,
        preview_count=preview_count,
        final_count=final_count,
        resolved_split=resolved_split,
        used_fallback_split=False,
    )
