"""Environment preparation helpers for run orchestration execution."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from invarlock.core import run_orchestrator_execute_helpers as _execute_helpers_module
from invarlock.core.exceptions import ConfigError, InvarlockError
from invarlock.core.run_orchestrator import (
    RunBaselineScheduleLoadedEvent,
    RunConfigLoadedEvent,
    RunConfigLoadingEvent,
    RunDeterministicSeedsEvent,
    RunDeviceResolvedEvent,
    RunOutputDirectoryReadyEvent,
    RunPipelineStartedEvent,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _RunEnvironmentState,
    _RunLossAndSeedState,
)
from invarlock.core.run_policy import (
    should_measure_overhead as _should_measure_overhead_default,
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


def _resolve_loss_seed_and_determinism_state(
    cfg: Any,
    *,
    model_profile: Any,
    profile_normalized: str,
    determinism_mode: str | None,
    determinism_warn_only: bool,
    optional_torch: Any,
    emit: RunEventEmitter,
    cfg_value: Any,
    config_value_exceptions: tuple[type[BaseException], ...],
    numeric_exceptions: tuple[type[BaseException], ...],
    optional_runtime_exceptions: tuple[type[BaseException], ...],
    set_seed_fn: Callable[[int], None] | None = None,
    numpy_module: Any = np,
    execute_helpers_module: Any = _execute_helpers_module,
) -> _RunLossAndSeedState:
    resolved_set_seed: Callable[[int], None]
    if set_seed_fn is None:
        from invarlock.core.determinism_policy import set_seed as resolved_set_seed
    else:
        resolved_set_seed = set_seed_fn

    eval_section = cfg_value(cfg, "eval") or {}
    loss_cfg = eval_section.get("loss") if isinstance(eval_section, dict) else None
    if loss_cfg is None and not isinstance(eval_section, dict):
        loss_cfg = getattr(eval_section, "loss", None)

    resolved_loss_type = (
        str(loss_cfg.get("type", "auto")).lower()
        if isinstance(loss_cfg, dict)
        else str(getattr(loss_cfg, "type", "auto")).lower()
        if loss_cfg
        else "auto"
    )
    if resolved_loss_type == "auto":
        resolved_loss_type = model_profile.default_loss
    use_mlm = resolved_loss_type == "mlm"

    mask_prob = loss_cfg.get("mask_prob") if isinstance(loss_cfg, dict) else None
    mask_prob = float(mask_prob) if isinstance(mask_prob, int | float) else 0.15
    if not isinstance(loss_cfg, dict):
        mask_prob = _coerce_float(getattr(loss_cfg, "mask_prob", None), mask_prob)

    mask_seed = _coerce_int(
        loss_cfg.get("seed") if isinstance(loss_cfg, dict) else None,
        42,
    )
    if not isinstance(loss_cfg, dict):
        mask_seed = _coerce_int(getattr(loss_cfg, "seed", None), mask_seed)

    random_token_prob = _coerce_float(
        loss_cfg.get("random_token_prob") if isinstance(loss_cfg, dict) else None,
        0.1,
    )
    if not isinstance(loss_cfg, dict):
        random_token_prob = _coerce_float(
            getattr(loss_cfg, "random_token_prob", None),
            random_token_prob,
        )

    original_token_prob = _coerce_float(
        loss_cfg.get("original_token_prob") if isinstance(loss_cfg, dict) else None,
        0.1,
    )
    if not isinstance(loss_cfg, dict):
        original_token_prob = _coerce_float(
            getattr(loss_cfg, "original_token_prob", None),
            original_token_prob,
        )
    if isinstance(loss_cfg, dict) and loss_cfg.get("type") == "auto":
        loss_cfg["type"] = resolved_loss_type

    raw_seed_value = 42
    if hasattr(cfg, "dataset"):
        try:
            raw_seed_value = getattr(cfg.dataset, "seed", 42)
        except config_value_exceptions:
            raw_seed_value = 42
    try:
        seed_value = int(raw_seed_value)
    except numeric_exceptions:
        seed_value = 42
    resolved_set_seed(seed_value)

    profile_label = profile_normalized or None
    torch_mod = optional_torch()
    if torch_mod is not None and profile_label in {"ci", "release"}:
        try:
            resolved_determinism_mode = determinism_mode or "throughput"
            warn_only = resolved_determinism_mode.lower() != "strict"
            if bool(determinism_warn_only):
                warn_only = True
            if hasattr(torch_mod, "use_deterministic_algorithms"):
                torch_mod.use_deterministic_algorithms(True, warn_only=warn_only)
            if hasattr(torch_mod.backends, "cudnn"):
                torch_mod.backends.cudnn.benchmark = False
                try:
                    torch_mod.backends.cudnn.deterministic = True
                except (AttributeError, TypeError, RuntimeError):
                    pass
        except optional_runtime_exceptions:
            pass

    try:
        numpy_state = cast(tuple[Any, ...], numpy_module.random.get_state())
        numpy_seed_state = numpy_state[1]
        numpy_seed = (
            int(numpy_seed_state[0]) if len(numpy_seed_state) > 0 else int(seed_value)
        )
    except (AttributeError, IndexError, OverflowError, TypeError, ValueError):
        numpy_seed = seed_value

    torch_seed = None
    if torch_mod is not None:
        try:
            torch_seed = int(torch_mod.initial_seed())
        except (AttributeError, OverflowError, RuntimeError, TypeError, ValueError):
            torch_seed = seed_value

    seed_bundle = {
        "python": int(seed_value),
        "numpy": int(numpy_seed),
        "torch": int(torch_seed) if torch_seed is not None else None,
    }
    emit(
        RunDeterministicSeedsEvent(
            python_seed=int(seed_bundle["python"] or seed_value),
            numpy_seed=int(seed_bundle["numpy"] or seed_value),
            torch_seed=seed_bundle["torch"],
        )
    )
    return _RunLossAndSeedState(
        eval_section=eval_section,
        resolved_loss_type=resolved_loss_type,
        use_mlm=use_mlm,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        seed_value=seed_value,
        seed_bundle=seed_bundle,
    )


def _apply_determinism_preset(
    *,
    profile_label: str | None,
    resolved_device: Any,
    seed_bundle: dict[str, int | None],
    seed_value: int,
) -> dict[str, Any] | None:
    try:
        from invarlock.core.determinism_policy import apply_determinism_preset

        preset = apply_determinism_preset(
            profile=profile_label,
            device=resolved_device,
            seed=int(seed_bundle.get("python") or seed_value),
            threads=int(os.environ.get("INVARLOCK_OMP_THREADS", 1) or 1),
        )
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return None
    if not isinstance(preset, dict) or not preset:
        return None
    preset_seeds = preset.get("seeds")
    if isinstance(preset_seeds, dict) and preset_seeds:
        for key in ("python", "numpy", "torch"):
            if key in preset_seeds:
                seed_bundle[key] = preset_seeds.get(key)
    return preset


def _load_baseline_evidence_state(
    *,
    baseline: str | None,
    profile_normalized: str,
    tokenizer_hash: str | None,
    load_baseline_pairing_evidence_fn: Any,
    emit: RunEventEmitter,
    emit_diagnostic: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    if not baseline:
        return None, None, tokenizer_hash
    baseline_path = Path(baseline)
    strict_baseline = profile_normalized in {"ci", "release"}
    baseline_evidence = load_baseline_pairing_evidence_fn(
        baseline_path=baseline_path,
        tokenizer_hash=tokenizer_hash,
    )
    updated_tokenizer_hash = baseline_evidence.tokenizer_hash
    if baseline_evidence.status == "loaded":
        emit(RunBaselineScheduleLoadedEvent())
    elif baseline_evidence.message:
        if strict_baseline:
            raise InvarlockError(code="E001", message=baseline_evidence.message)
        emit_diagnostic(
            code="baseline_schedule_fallback",
            summary=baseline_evidence.message,
            level="warning",
        )
    return (
        baseline_evidence.report_data,
        baseline_evidence.pairing_schedule,
        updated_tokenizer_hash,
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


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
