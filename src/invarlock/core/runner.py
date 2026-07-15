"""
InvarLock Core Runner
=================

Main pipeline execution orchestrator: prepare → edit → guards → eval → finalize/rollback.
Torch-independent coordination with proper event logging and checkpoint management.
"""

from __future__ import annotations

import os
import time
from typing import Any

from invarlock.eval.tail_stats import evaluate_metric_tail
from invarlock.observability.metrics import (
    capture_memory_snapshot,
)

from .api import (
    EditLike,
    EditRuntime,
    Guard,
    ModelAdapter,
    ModelEdit,
    RunConfig,
    RunReport,
)
from .auto_tuning import resolve_tier_policies
from .bootstrap import compute_independent_delta_log_ci, logspace_to_ratio_ci
from .checkpoint import CheckpointManager
from .events import EventLogger
from .exceptions import InvarlockError
from .runner_runtime.eval_metrics import (
    compute_real_metrics,
    measure_latency,
    samples_to_dataloader,
)
from .runner_runtime.eval_phase import eval_phase
from .runner_runtime.finalize import finalize_phase, handle_error
from .runner_runtime.guards import (
    apply_guard_policy,
    guard_phase,
    prepare_guards_phase,
    resolve_guard_policies,
)
from .runner_runtime.pairing import BOOTSTRAP_COVERAGE_REQUIREMENTS
from .types import LogLevel

__all__ = ["CoreRunner"]

_RUNNER_EXECUTION_ERRORS = (
    AssertionError,
    InvarlockError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_CUDA_FLAG_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    RuntimeError,
    TypeError,
    ValueError,
)
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    return None


def env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return coerce_bool(raw)


def collect_cuda_flags() -> dict[str, Any]:
    """Capture deterministic CUDA configuration for provenance."""
    flags: dict[str, Any] = {}
    try:
        import torch

        flags["deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        if hasattr(torch.backends, "cudnn"):
            flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                flags["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            matmul = torch.backends.cuda.matmul
            if hasattr(matmul, "allow_tf32"):
                flags["cuda_matmul_allow_tf32"] = bool(matmul.allow_tf32)
    except _CUDA_FLAG_ERRORS:  # pragma: no cover - fallback when torch missing
        pass

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace:
        flags["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return flags


def serialize_config(config: RunConfig) -> dict[str, Any]:
    """Serialize RunConfig for storage in report."""
    return {
        "device": config.device,
        "max_pm_ratio": config.max_pm_ratio,
        "checkpoint_interval": config.checkpoint_interval,
        "dry_run": config.dry_run,
        "verbose": config.verbose,
        "guards": config.context.get("guards", {}) if config.context else {},
    }


def _resolve_policy_flag(
    run_ctx: dict[str, Any],
    eval_ctx: dict[str, Any],
    *,
    run_key: str,
    eval_keys: tuple[str, ...],
    env_key: str | None,
    default: bool,
) -> bool:
    val = coerce_bool(run_ctx.get(run_key))
    if val is None:
        for key in eval_keys:
            val = coerce_bool(eval_ctx.get(key))
            if val is not None:
                break
    if env_key:
        env_val = env_flag(env_key)
        if env_val is not None:
            val = env_val
    return default if val is None else bool(val)


def resolve_policy_flags(config: RunConfig | None) -> dict[str, bool]:
    run_ctx: dict[str, Any] = {}
    eval_ctx: dict[str, Any] = {}
    if config and isinstance(config.context, dict):
        run_ctx = (
            config.context.get("run", {})
            if isinstance(config.context.get("run"), dict)
            else {}
        )
        eval_ctx = (
            config.context.get("eval", {})
            if isinstance(config.context.get("eval"), dict)
            else {}
        )

    return {
        "strict_eval": _resolve_policy_flag(
            run_ctx,
            eval_ctx,
            run_key="strict_eval",
            eval_keys=("strict_errors", "strict"),
            env_key=None,
            default=True,
        ),
        "strict_guard_prepare": _resolve_policy_flag(
            run_ctx,
            eval_ctx,
            run_key="strict_guard_prepare",
            eval_keys=(),
            env_key=None,
            default=True,
        ),
        "allow_calibration_materialize": _resolve_policy_flag(
            run_ctx,
            eval_ctx,
            run_key="allow_calibration_materialize",
            eval_keys=("materialize_calibration", "allow_iterable_calibration"),
            env_key="INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE",
            default=False,
        ),
    }


def initialize_run_report(
    *,
    config: RunConfig,
    serialized_config: dict[str, Any],
    cuda_flags: dict[str, Any],
    auto_config: dict[str, Any] | None = None,
    report_factory: type[RunReport] = RunReport,
    start_time: float | None = None,
) -> RunReport:
    report = report_factory()
    context = config.context
    report.meta["cuda_flags"] = cuda_flags
    report.meta["start_time"] = (
        float(start_time) if start_time is not None else float(time.time())
    )
    report.meta["config"] = serialized_config

    if context:
        normalized_context = dict(context)
        try:
            report.context.update(normalized_context)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            report.context = normalized_context

    run_id = context.get("run_id") if context is not None else None
    if run_id:
        report.meta["run_id"] = run_id
    plugins_meta = context.get("plugins") if context is not None else None
    if plugins_meta:
        report.meta["plugins"] = plugins_meta

    if auto_config:
        report.meta["auto"] = auto_config
        existing_auto = context.get("auto") if context is not None else None
        if isinstance(context, dict) and isinstance(existing_auto, dict):
            merged_auto = dict(existing_auto)
            merged_auto.update(auto_config)
            context["auto"] = merged_auto
            report.context["auto"] = context["auto"]
        elif isinstance(context, dict):
            context["auto"] = dict(auto_config)
            report.context["auto"] = context["auto"]

    return report


def finalize_run_report(
    report: RunReport,
    *,
    final_status: str,
    end_time: float | None = None,
) -> None:
    end_ts = float(end_time) if end_time is not None else float(time.time())
    report.status = final_status
    report.meta["end_time"] = end_ts
    start_time = report.meta.get("start_time")
    if isinstance(start_time, int | float):
        report.meta["duration"] = end_ts - float(start_time)


def merge_execution_metrics(
    report: RunReport,
    *,
    timings: dict[str, float],
    guard_timings: dict[str, float],
    memory_snapshots: list[dict[str, Any]],
    memory_summary: dict[str, Any],
) -> None:
    metrics_obj: object = report.metrics
    if isinstance(metrics_obj, dict):
        metrics = metrics_obj
    else:
        report.metrics = {}
        metrics = report.metrics

    if timings:
        metrics.setdefault("timings", {}).update(timings)

    if guard_timings:
        metrics["guard_timings"] = guard_timings

    if not memory_snapshots:
        return

    metrics["memory_snapshots"] = memory_snapshots
    summary = dict(memory_summary)
    mem_peak = summary.get("memory_mb_peak")
    if isinstance(mem_peak, int | float):
        existing_peak = metrics.get("memory_mb_peak")
        if isinstance(existing_peak, int | float):
            summary["memory_mb_peak"] = max(float(existing_peak), float(mem_peak))
    metrics.update(summary)


def _profile_from_context(context: dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict):
        return None
    raw_profile = context.get("profile")
    if isinstance(raw_profile, str) and raw_profile.strip():
        return raw_profile.strip().lower()
    runtime_context = context.get("runtime")
    if isinstance(runtime_context, dict):
        raw_runtime_profile = runtime_context.get("profile")
        if isinstance(raw_runtime_profile, str) and raw_runtime_profile.strip():
            return raw_runtime_profile.strip().lower()
    return None


def initialize_services(
    runner: Any,
    config: Any,
    *,
    event_logger_factory: Any = EventLogger,
    checkpoint_factory: Any = CheckpointManager,
) -> None:
    """Initialize event logging and checkpoint services."""
    if config.event_path:
        run_id = None
        if isinstance(config.context, dict):
            run_id = config.context.get("run_id")
        runner.event_logger = event_logger_factory(config.event_path, run_id=run_id)

    if config.checkpoint_interval > 0:
        runner.checkpoint_manager = checkpoint_factory()


def cleanup_services(runner: Any) -> None:
    """Clean up event logging and checkpoint services."""
    if runner.event_logger:
        runner.event_logger.close()
        runner.event_logger = None

    if runner.checkpoint_manager:
        runner.checkpoint_manager.cleanup()
        runner.checkpoint_manager = None


def record_timing(
    timings: dict[str, float],
    key: str,
    start: float,
    *,
    perf_counter: Any = time.perf_counter,
) -> None:
    timings[key] = max(0.0, float(perf_counter() - start))


def capture_memory(
    memory_snapshots: list[dict[str, Any]],
    phase: str,
    *,
    capture_fn: Any = capture_memory_snapshot,
) -> None:
    snapshot = capture_fn(phase)
    if snapshot:
        memory_snapshots.append(snapshot)


class CoreRunner:
    """
    Core pipeline execution orchestrator.

    Coordinates the full InvarLock pipeline while maintaining torch-independence
    in the core coordination logic. Provides event logging, checkpointing,
    and rollback capabilities.
    """

    def __init__(self):
        self.event_logger = None
        self.checkpoint_manager = None
        self._active_model: Any | None = None
        self._active_adapter: ModelAdapter | None = None

    def execute(
        self,
        model: Any,
        adapter: ModelAdapter,
        edit: ModelEdit | EditLike,
        guards: list[Guard],
        config: RunConfig,
        calibration_data: Any = None,
        auto_config: dict[str, Any] | None = None,
        edit_config: dict[str, Any] | None = None,
        edit_runtime: EditRuntime | None = None,
        preview_n: int | None = None,
        final_n: int | None = None,
    ) -> RunReport:
        """Execute the full InvarLock pipeline."""
        from .runner_runtime.execution_plan import (
            RunnerExecutionRequest,
            execute_runner_execution_plan,
        )

        return execute_runner_execution_plan(
            self,
            RunnerExecutionRequest(
                model=model,
                adapter=adapter,
                edit=edit,
                guards=guards,
                config=config,
                calibration_data=calibration_data,
                auto_config=auto_config,
                edit_config=edit_config,
                edit_runtime=edit_runtime,
                preview_n=preview_n,
                final_n=final_n,
            ),
            initialize_run_report_fn=initialize_run_report,
            collect_cuda_flags_fn=collect_cuda_flags,
            profile_from_context_fn=_profile_from_context,
            record_timing_fn=record_timing,
            capture_memory_fn=capture_memory,
            finalize_run_report_fn=finalize_run_report,
            merge_execution_metrics_fn=merge_execution_metrics,
            runner_execution_errors=_RUNNER_EXECUTION_ERRORS,
        )

    def _initialize_services(self, config: RunConfig) -> None:
        initialize_services(
            self,
            config,
            event_logger_factory=EventLogger,
            checkpoint_factory=CheckpointManager,
        )

    def _cleanup_services(self) -> None:
        cleanup_services(self)

    def _prepare_phase(
        self, model: Any, adapter: ModelAdapter, report: RunReport
    ) -> dict[str, Any]:
        """Phase 1: Model preparation and analysis."""
        self._log_event("prepare", "start", LogLevel.INFO)
        model_desc = adapter.describe(model)
        report.meta["model"] = model_desc

        if self.checkpoint_manager:
            checkpoint_id = self.checkpoint_manager.create_checkpoint(model, adapter)
            report.meta["initial_checkpoint"] = checkpoint_id
            self._log_event(
                "prepare", "checkpoint_created", LogLevel.INFO, {"id": checkpoint_id}
            )

        self._log_event(
            "prepare",
            "complete",
            LogLevel.INFO,
            {"layers": model_desc.get("n_layer", 0)},
        )
        return model_desc

    def _edit_phase(
        self,
        model: Any,
        adapter: ModelAdapter,
        edit: ModelEdit | EditLike,
        model_desc: dict[str, Any],
        report: RunReport,
        edit_config: dict[str, Any] | None,
        edit_runtime: EditRuntime | None,
    ) -> dict[str, Any]:
        """Phase 2: Apply edit operation."""
        edit_label = "baseline" if edit.name == "baseline" else edit.name
        self._log_event("edit", "start", LogLevel.INFO, {"edit": edit_label})
        report.meta["edit_name"] = edit.name

        if not edit.can_edit(model_desc):
            raise ValueError(f"Edit '{edit.name}' cannot be applied to this model")

        edit_result = edit.apply(
            model,
            adapter,
            plan=dict(edit_config or {}) or None,
            runtime=edit_runtime,
        )
        report.edit = edit_result
        if not isinstance(report.context, dict):
            report.context = {}
        edit_context = report.context.setdefault("edit", {})
        if isinstance(edit_result, dict):
            edit_context.setdefault("name", edit_result.get("name", edit.name))
            deltas = edit_result.get("deltas") or {}
            if isinstance(deltas, dict):
                edit_context["params_changed"] = deltas.get("params_changed", 0)
                edit_context["layers_modified"] = deltas.get("layers_modified", 0)
            else:
                edit_context.setdefault("params_changed", 0)
        else:
            edit_context.setdefault("name", edit.name)
            edit_context.setdefault("params_changed", 0)

        self._log_event(
            "edit",
            "complete",
            LogLevel.INFO,
            {"edit": edit.name, "result": edit_result},
        )
        return edit_result

    def _prepare_guards_phase(
        self,
        model: Any,
        adapter: ModelAdapter,
        guards: list[Guard],
        calibration_data: Any,
        report: RunReport,
        auto_config: dict[str, Any] | None = None,
        config: RunConfig | None = None,
    ) -> None:
        prepare_guards_phase(
            self,
            model,
            adapter,
            guards,
            calibration_data,
            report,
            auto_config,
            config,
        )

    def _guard_phase(
        self,
        model: Any,
        adapter: ModelAdapter,
        guards: list[Guard],
        report: RunReport,
        *,
        guard_timings: dict[str, float] | None = None,
        result_keys: list[str] | None = None,
        result_stages: list[str | None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        return guard_phase(
            self,
            model,
            adapter,
            guards,
            report,
            guard_timings=guard_timings,
            result_keys=result_keys,
            result_stages=result_stages,
        )

    def _eval_phase(
        self,
        model: Any,
        adapter: ModelAdapter,
        calibration_data: Any,
        report: RunReport,
        preview_n: int | None = None,
        final_n: int | None = None,
        config: RunConfig | None = None,
    ) -> dict[str, Any]:
        return eval_phase(
            self,
            model,
            adapter,
            calibration_data,
            report,
            preview_n,
            final_n,
            config,
            evaluate_metric_tail_fn=evaluate_metric_tail,
        )

    def _compute_real_metrics(
        self,
        model: Any,
        calibration_data: Any,
        adapter: ModelAdapter,
        preview_n: int | None = None,
        final_n: int | None = None,
        config: RunConfig | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return compute_real_metrics(
            self,
            model,
            calibration_data,
            adapter,
            preview_n,
            final_n,
            config,
            compute_independent_delta_log_ci_fn=compute_independent_delta_log_ci,
            logspace_to_ratio_ci_fn=logspace_to_ratio_ci,
            coverage_requirements=BOOTSTRAP_COVERAGE_REQUIREMENTS,
        )

    def _measure_latency(self, model: Any, sample_data: Any, device: Any) -> float:
        return measure_latency(model, sample_data, device)

    def _samples_to_dataloader(self, samples: list[Any]) -> Any:
        return samples_to_dataloader(samples)

    def _finalize_phase(
        self,
        model: Any,
        adapter: ModelAdapter,
        guard_results: dict[str, dict[str, Any]],
        metrics: dict[str, Any],
        config: RunConfig,
        report: RunReport,
    ) -> str:
        return finalize_phase(
            self, model, adapter, guard_results, metrics, config, report
        )

    def _handle_error(
        self,
        error: Exception,
        report: RunReport,
        model: Any | None = None,
        adapter: ModelAdapter | None = None,
    ) -> None:
        handle_error(self, error, report, model=model, adapter=adapter)

    def _resolve_guard_policies(
        self, report: RunReport, auto_config: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        return resolve_guard_policies(
            self,
            report,
            auto_config,
            resolver=resolve_tier_policies,
        )

    def _apply_guard_policy(self, guard: Guard, policy: dict[str, Any]) -> None:
        apply_guard_policy(self, guard, policy)

    def _log_event(
        self,
        component: str,
        operation: str,
        level: LogLevel,
        data: dict[str, Any] | None = None,
    ) -> None:
        if self.event_logger:
            self.event_logger.log(component, operation, level, data)

    def _serialize_config(self, config: RunConfig) -> dict[str, Any]:
        return serialize_config(config)

    def _resolve_policy_flags(self, config: RunConfig | None) -> dict[str, bool]:
        return resolve_policy_flags(config)
