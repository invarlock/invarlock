"""Shared helpers for config-driven run orchestration execution."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_build_run_context_payload_impl: Any | None = None
_build_run_execution_config_payloads_impl: Any | None = None
_resolve_pm_acceptance_range_impl: Any | None = None
_resolve_pm_drift_band_impl: Any | None = None
_resolve_guard_overhead_threshold_impl: Any | None = None
_should_measure_overhead_impl: Any | None = None


def _coerce_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return float(default)
    return coerced if math.isfinite(coerced) else float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return int(default)


RunEventEmitter = Callable[[Any], None]


@dataclass(frozen=True)
class _RunLossAndSeedState:
    eval_section: Any
    resolved_loss_type: str
    use_mlm: bool
    mask_prob: float
    mask_seed: int
    random_token_prob: float
    original_token_prob: float
    seed_value: int
    seed_bundle: dict[str, int | None]


@dataclass(frozen=True)
class _RunEnvironmentState:
    cfg: Any
    model_profile: Any
    eval_section: Any
    resolved_loss_type: str
    use_mlm: bool
    mask_prob: float
    mask_seed: int
    random_token_prob: float
    original_token_prob: float
    seed_value: int
    seed_bundle: dict[str, int | None]
    profile_label: str | None
    resolved_device: Any
    output_dir: Path
    determinism_meta: dict[str, Any] | None
    run_dir: Path
    run_id: str
    retry_controller: Any | None
    measure_guard_overhead: bool
    skip_overhead: bool
    skip_overhead_source: str | None
    direct_reuse_loaded_model: bool
    emitted_skip_overhead_warning: bool
    tokenizer: Any | None
    tokenizer_hash: str | None
    baseline_report_data: dict[str, Any] | None
    pairing_schedule: dict[str, Any] | None
    requested_preview: int
    requested_final: int
    effective_preview: int
    effective_final: int
    preview_count: int
    final_count: int
    resolved_split: str
    used_fallback_split: bool


@dataclass(frozen=True)
class _RunComponentState:
    adapter: Any
    edit_op: Any
    guards: list[Any]
    run_context: dict[str, Any]
    run_config: Any
    pm_acceptance_range: Any
    pm_drift_band: Any
    guard_overhead_threshold: float


@dataclass(frozen=True)
class _RunDatasetState:
    tokenizer: Any | None
    tokenizer_hash: str | None
    calibration_data: list[dict[str, Any]]
    dataset_meta: dict[str, Any]
    window_plan: dict[str, Any] | None
    preview_count: int
    final_count: int
    effective_preview: int
    effective_final: int
    preview_mask_counts: list[int]
    final_mask_counts: list[int]
    preview_records: list[dict[str, Any]]
    final_records: list[dict[str, Any]]
    resolved_split: str
    used_fallback_split: bool


@dataclass
class _RunExecutionState:
    runner: Any
    auto_config: Any
    edit_config: Any
    model: Any | None
    restore_fn: Any | None
    snapshot_tmpdir: Any | None
    snapshot_provenance: dict[str, bool]
    skip_model_load: bool
    emitted_skip_overhead_warning: bool


@dataclass(frozen=True)
class _AttemptExecutionState:
    attempt: int
    edit_config: Any
    guard_overhead_payload: dict[str, Any] | None
    core_report: Any | None
    model: Any | None
    should_continue: bool


@dataclass(frozen=True)
class _AttemptDecision:
    report: dict[str, Any]
    timings: dict[str, float]
    report_path_out: str | None
    edit_config: Any
    attempt: int
    should_continue: bool


def _cfg_section_value(
    cfg_obj: Any,
    name: str,
    config_value_exceptions: tuple[type[BaseException], ...],
) -> Any:
    section_fn = getattr(cfg_obj, "section", None)
    if callable(section_fn):
        try:
            section = section_fn(name)
        except config_value_exceptions:
            section = None
        if section is not None:
            return section
    try:
        return getattr(cfg_obj, name)
    except config_value_exceptions:
        return None
