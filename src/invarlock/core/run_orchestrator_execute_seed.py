"""Seed and determinism helpers for run orchestration execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from invarlock.core import run_orchestrator_execute_helpers as _execute_helpers_module
from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _RunLossAndSeedState,
)
from invarlock.core.run_orchestrator_types import (
    RunBaselineScheduleLoadedEvent,
    RunDeterministicSeedsEvent,
)
from invarlock.model_utils import set_seed


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
    set_seed_fn: Any = set_seed,
    numpy_module: Any = np,
    execute_helpers_module: Any = _execute_helpers_module,
) -> _RunLossAndSeedState:
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
    mask_prob = float(mask_prob) if isinstance(mask_prob, (int, float)) else 0.15
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
    set_seed_fn(seed_value)

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
