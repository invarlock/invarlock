"""
InvarLock Core Runner
=================

Main pipeline execution orchestrator: prepare → edit → guards → eval → finalize/rollback.
Torch-independent coordination with proper event logging and checkpoint management.
"""

from __future__ import annotations

import time
from typing import Any

from invarlock.eval.tail_stats import evaluate_metric_tail
from invarlock.observability.metrics import (
    capture_memory_snapshot,
    reset_peak_memory_stats,
    summarize_memory_snapshots,
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
from .bootstrap import compute_paired_delta_log_ci, logspace_to_ratio_ci
from .checkpoint import CheckpointManager
from .events import EventLogger
from .exceptions import InvarlockError
from .runner_context import (
    collect_cuda_flags as _collect_cuda_flags,
)
from .runner_context import (
    resolve_policy_flags,
    serialize_config,
)
from .runner_eval_metrics import compute_real_metrics
from .runner_eval_phase import eval_phase
from .runner_finalize import finalize_phase, handle_error
from .runner_guards import (
    apply_guard_policy,
    guard_phase,
    prepare_guards_phase,
    resolve_guard_policies,
)
from .runner_latency import measure_latency, samples_to_dataloader
from .runner_lifecycle import (
    finalize_run_report,
    initialize_run_report,
    merge_execution_metrics,
)
from .runner_pairing import BOOTSTRAP_COVERAGE_REQUIREMENTS
from .runner_services import (
    capture_memory,
    cleanup_services,
    initialize_services,
    record_timing,
)
from .types import LogLevel, RunStatus

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
        self._initialize_services(config)
        self._active_model = model
        self._active_adapter = adapter

        report = initialize_run_report(
            config=config,
            serialized_config=self._serialize_config(config),
            cuda_flags=_collect_cuda_flags(),
            auto_config=auto_config,
            report_factory=RunReport,
        )

        report.status = RunStatus.RUNNING.value
        timings: dict[str, float] = {}
        guard_timings: dict[str, float] = {}
        memory_snapshots: list[dict[str, Any]] = []
        total_start = time.perf_counter()

        try:
            self._log_event(
                "runner",
                "start",
                LogLevel.INFO,
                {
                    "edit": edit.name,
                    "guards": [guard.name for guard in guards],
                    "context": report.context,
                },
            )

            reset_peak_memory_stats()
            phase_start = time.perf_counter()
            try:
                model_desc = self._prepare_phase(model, adapter, report)
            finally:
                record_timing(timings, "prepare", phase_start)
                capture_memory(
                    memory_snapshots, "prepare", capture_fn=capture_memory_snapshot
                )

            reset_peak_memory_stats()
            phase_start = time.perf_counter()
            try:
                self._prepare_guards_phase(
                    model,
                    adapter,
                    guards,
                    calibration_data,
                    report,
                    auto_config,
                    config,
                )
            finally:
                record_timing(timings, "prepare_guards", phase_start)
                capture_memory(
                    memory_snapshots,
                    "prepare_guards",
                    capture_fn=capture_memory_snapshot,
                )

            reset_peak_memory_stats()
            phase_start = time.perf_counter()
            try:
                self._edit_phase(
                    model,
                    adapter,
                    edit,
                    model_desc,
                    report,
                    edit_config,
                    edit_runtime,
                )
            finally:
                record_timing(timings, "edit", phase_start)
                capture_memory(
                    memory_snapshots, "edit", capture_fn=capture_memory_snapshot
                )

            reset_peak_memory_stats()
            phase_start = time.perf_counter()
            try:
                guard_results = self._guard_phase(
                    model, adapter, guards, report, guard_timings=guard_timings
                )
            finally:
                record_timing(timings, "guards", phase_start)
                capture_memory(
                    memory_snapshots, "guards", capture_fn=capture_memory_snapshot
                )

            reset_peak_memory_stats()
            phase_start = time.perf_counter()
            try:
                metrics = self._eval_phase(
                    model,
                    adapter,
                    calibration_data,
                    report,
                    preview_n,
                    final_n,
                    config,
                )
            finally:
                record_timing(timings, "eval", phase_start)
                capture_memory(
                    memory_snapshots, "eval", capture_fn=capture_memory_snapshot
                )

            reset_peak_memory_stats()
            phase_start = time.perf_counter()
            try:
                final_status = self._finalize_phase(
                    model, adapter, guard_results, metrics, config, report
                )
            finally:
                record_timing(timings, "finalize", phase_start)
                capture_memory(
                    memory_snapshots, "finalize", capture_fn=capture_memory_snapshot
                )

            finalize_run_report(report, final_status=final_status)
            self._log_event(
                "runner",
                "complete",
                LogLevel.INFO,
                {"status": final_status, "duration": report.meta["duration"]},
            )
            return report
        except _RUNNER_EXECUTION_ERRORS as error:
            self._handle_error(error, report, model=model, adapter=adapter)
            return report
        finally:
            record_timing(timings, "total", total_start)
            merge_execution_metrics(
                report,
                timings=timings,
                guard_timings=guard_timings,
                memory_snapshots=memory_snapshots,
                memory_summary=summarize_memory_snapshots(memory_snapshots)
                if memory_snapshots
                else {},
            )
            self._active_model = None
            self._active_adapter = None
            self._cleanup_services()

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
    ) -> dict[str, dict[str, Any]]:
        return guard_phase(
            self,
            model,
            adapter,
            guards,
            report,
            guard_timings=guard_timings,
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
            compute_paired_delta_log_ci_fn=compute_paired_delta_log_ci,
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
