"""
InvarLock Core Runner
=================

Main pipeline execution orchestrator: prepare → edit → guards → eval → finalize/rollback.
Torch-independent coordination with proper event logging and checkpoint management.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from array import array
from collections.abc import Sequence
from typing import Any

import numpy as np

from .api import Guard, ModelAdapter, ModelEdit, RunConfig, RunReport
from .auto_tuning import resolve_tier_policies
from .bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)
from .checkpoint import CheckpointManager
from .events import EventLogger
from .types import LogLevel, RunStatus

BOOTSTRAP_COVERAGE_REQUIREMENTS = {
    # Minimum window counts and bootstrap replicates expected per policy tier.
    # Individual configs can request more aggressive settings, but these values
    # represent the guard-rail floor that CI profiles should maintain.
    "conservative": {"preview": 220, "final": 220, "replicates": 1500},
    "balanced": {"preview": 180, "final": 180, "replicates": 1200},
    "aggressive": {"preview": 140, "final": 140, "replicates": 800},
}

__all__ = ["CoreRunner"]


def _collect_cuda_flags() -> dict[str, Any]:
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
    except Exception:  # pragma: no cover - fallback when torch missing
        pass

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace:
        flags["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return flags


class CoreRunner:
    """
    Core pipeline execution orchestrator.

    Coordinates the full InvarLock pipeline while maintaining torch-independence
    in the core coordination logic. Provides event logging, checkpointing,
    and rollback capabilities.
    """

    def __init__(self):
        self.event_logger: EventLogger | None = None
        self.checkpoint_manager: CheckpointManager | None = None

    def execute(
        self,
        model: Any,
        adapter: ModelAdapter,
        edit: ModelEdit,
        guards: list[Guard],
        config: RunConfig,
        calibration_data: Any = None,
        auto_config: dict[str, Any] | None = None,
        edit_config: dict[str, Any] | None = None,
        preview_n: int | None = None,
        final_n: int | None = None,
    ) -> RunReport:
        """
        Execute the full InvarLock pipeline.

        Args:
            model: The model to process
            adapter: Model adapter for model-specific operations
            edit: Edit to apply
            guards: Safety guards to run
            config: Runtime configuration
            calibration_data: Optional calibration/validation data
            auto_config: Optional auto-tuning configuration

        Returns:
            RunReport with execution results
        """
        # Initialize services
        self._initialize_services(config)

        # Create report
        report = RunReport()
        report.meta["cuda_flags"] = _collect_cuda_flags()
        report.meta["start_time"] = time.time()
        report.meta["config"] = self._serialize_config(config)
        if config.context:
            try:
                report.context.update(config.context)
            except Exception:
                # Defensive: ensure context remains a dict even if update fails
                report.context = dict(config.context)

        if isinstance(config.context, dict):
            run_id = config.context.get("run_id")
            if run_id:
                report.meta["run_id"] = run_id
            plugins_meta = config.context.get("plugins")
            if plugins_meta:
                report.meta["plugins"] = plugins_meta

        # Store auto configuration for tier resolution
        if auto_config:
            report.meta["auto"] = auto_config

        report.status = RunStatus.RUNNING.value

        try:
            # Log start
            self._log_event(
                "runner",
                "start",
                LogLevel.INFO,
                {
                    "edit": edit.name,
                    "guards": [g.name for g in guards],
                    "context": report.context,
                },
            )

            # Phase 1: Prepare (describe model, create checkpoint)
            model_desc = self._prepare_phase(model, adapter, report)

            # Phase 2: Prepare guards (must happen before edit)
            self._prepare_guards_phase(
                model, adapter, guards, calibration_data, report, auto_config
            )

            # Phase 3: Apply edit
            self._edit_phase(model, adapter, edit, model_desc, report, edit_config)

            # Phase 4: Run guards
            guard_results = self._guard_phase(model, adapter, guards, report)

            # Phase 5: Evaluate final metrics
            metrics = self._eval_phase(
                model,
                adapter,
                calibration_data,
                report,
                preview_n,
                final_n,
                config,
            )

            # Phase 6: Finalize or rollback
            final_status = self._finalize_phase(
                model, adapter, guard_results, metrics, config, report
            )

            report.status = final_status
            report.meta["end_time"] = time.time()
            report.meta["duration"] = (
                report.meta["end_time"] - report.meta["start_time"]
            )

            self._log_event(
                "runner",
                "complete",
                LogLevel.INFO,
                {"status": final_status, "duration": report.meta["duration"]},
            )

            return report

        except Exception as e:
            self._handle_error(e, report)
            return report

        finally:
            self._cleanup_services()

    def _initialize_services(self, config: RunConfig) -> None:
        """Initialize event logging and checkpoint services."""
        if config.event_path:
            run_id = None
            if isinstance(config.context, dict):
                run_id = config.context.get("run_id")
            self.event_logger = EventLogger(config.event_path, run_id=run_id)

        if config.checkpoint_interval > 0:
            self.checkpoint_manager = CheckpointManager()

    def _cleanup_services(self) -> None:
        """Clean up services."""
        if self.event_logger:
            self.event_logger.close()

        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup()

    def _prepare_phase(
        self, model: Any, adapter: ModelAdapter, report: RunReport
    ) -> dict[str, Any]:
        """Phase 1: Model preparation and analysis."""
        self._log_event("prepare", "start", LogLevel.INFO)

        # Describe model structure
        model_desc = adapter.describe(model)
        report.meta["model"] = model_desc

        # Create initial checkpoint
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
        edit: ModelEdit,
        model_desc: dict[str, Any],
        report: RunReport,
        edit_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Phase 2: Apply edit operation."""
        edit_label = "baseline" if edit.name == "baseline" else edit.name
        self._log_event("edit", "start", LogLevel.INFO, {"edit": edit_label})

        # Store edit name for tier resolution
        report.meta["edit_name"] = edit.name

        # Check if edit can be applied
        if not edit.can_edit(model_desc):
            raise ValueError(f"Edit '{edit.name}' cannot be applied to this model")

        # Apply edit with configuration parameters
        if edit_config:
            edit_result = edit.apply(model, adapter, **edit_config)
        else:
            edit_result = edit.apply(model, adapter)
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
    ) -> None:
        """Phase 2: Prepare safety guards with tier-resolved policies."""
        self._log_event(
            "guards_prepare", "start", LogLevel.INFO, {"count": len(guards)}
        )

        # Resolve tier policies before guard preparation
        tier_policies = self._resolve_guard_policies(report, auto_config)

        for guard in guards:
            self._log_event(
                "guard_prepare", "start", LogLevel.INFO, {"guard": guard.name}
            )

            try:
                guard_policy: dict[str, Any] = tier_policies.get(guard.name, {})

                # Apply tier-resolved policy to guard
                if guard_policy:
                    self._apply_guard_policy(guard, guard_policy)
                    self._log_event(
                        "guard_prepare",
                        "policy_applied",
                        LogLevel.INFO,
                        {"guard": guard.name, "policy": guard_policy},
                    )

                if hasattr(guard, "set_run_context"):
                    try:
                        guard.set_run_context(report)
                    except Exception as exc:
                        self._log_event(
                            "guard_prepare",
                            "context_error",
                            LogLevel.WARNING,
                            {"guard": guard.name, "error": str(exc)},
                        )

                # Call prepare method if it exists (most guards need this)
                if hasattr(guard, "prepare"):
                    prepare_result = guard.prepare(
                        model, adapter, calibration_data, guard_policy
                    )
                    self._log_event(
                        "guard_prepare",
                        "complete",
                        LogLevel.INFO,
                        {
                            "guard": guard.name,
                            "ready": prepare_result.get("ready", False),
                        },
                    )
                else:
                    self._log_event(
                        "guard_prepare",
                        "skipped",
                        LogLevel.INFO,
                        {"guard": guard.name, "reason": "no_prepare_method"},
                    )

            except Exception as e:
                self._log_event(
                    "guard_prepare",
                    "error",
                    LogLevel.ERROR,
                    {"guard": guard.name, "error": str(e)},
                )

        # Store resolved policies in report for certificate
        report.meta["tier_policies"] = tier_policies

        self._log_event(
            "guards_prepare", "complete", LogLevel.INFO, {"count": len(guards)}
        )

    def _guard_phase(
        self, model: Any, adapter: ModelAdapter, guards: list[Guard], report: RunReport
    ) -> dict[str, dict[str, Any]]:
        """Phase 4: Run safety guards."""
        self._log_event("guards", "start", LogLevel.INFO, {"count": len(guards)})

        guard_results = {}

        for guard in guards:
            self._log_event("guard", "start", LogLevel.INFO, {"guard": guard.name})

            if hasattr(guard, "set_run_context"):
                try:
                    guard.set_run_context(report)
                except Exception as exc:  # pragma: no cover - defensive
                    self._log_event(
                        "guard",
                        "context_error",
                        LogLevel.WARNING,
                        {"guard": guard.name, "error": str(exc)},
                    )

            try:
                result = guard.validate(model, adapter, report.context)
                guard_results[guard.name] = result

                # Log guard result
                status = "passed" if result.get("passed", False) else "failed"
                self._log_event(
                    "guard",
                    "complete",
                    LogLevel.INFO,
                    {"guard": guard.name, "status": status},
                )

            except Exception as e:
                guard_results[guard.name] = {"passed": False, "error": str(e)}
                self._log_event(
                    "guard",
                    "error",
                    LogLevel.ERROR,
                    {"guard": guard.name, "error": str(e)},
                )

        report.guards = guard_results

        # Summary
        passed_guards = sum(1 for r in guard_results.values() if r.get("passed", False))
        self._log_event(
            "guards",
            "complete",
            LogLevel.INFO,
            {"total": len(guards), "passed": passed_guards},
        )

        return guard_results

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
        """Phase 4: Final evaluation metrics."""
        self._log_event("eval", "start", LogLevel.INFO)

        if calibration_data is not None:
            if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                length_hint = None
                try:
                    length_hint = len(calibration_data)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - defensive
                    length_hint = None
                first_batch = None
                indexable = hasattr(calibration_data, "__getitem__")
                if isinstance(calibration_data, list | tuple):
                    if calibration_data:
                        first_batch = calibration_data[0]
                elif indexable:
                    try:
                        first_batch = calibration_data[0]  # type: ignore[index]
                    except Exception:  # pragma: no cover - defensive
                        first_batch = None
                masked_preview = None
                first_keys = None
                if isinstance(first_batch, dict):
                    first_keys = list(first_batch.keys())
                    labels_preview = first_batch.get("labels")
                    if isinstance(labels_preview, list | tuple):
                        try:
                            masked_preview = sum(
                                1 for tok in labels_preview if tok != -100
                            )
                        except Exception:  # pragma: no cover - defensive
                            masked_preview = None
                self._log_event(
                    "eval",
                    "calibration_snapshot",
                    LogLevel.DEBUG,
                    {
                        "calibration_type": type(calibration_data).__name__,
                        "length_hint": length_hint,
                        "indexable": bool(indexable),
                        "first_batch_keys": first_keys,
                        "first_batch_masked": masked_preview,
                    },
                )
            # Compute real perplexity using calibration data
            metrics, eval_windows = self._compute_real_metrics(
                model,
                calibration_data,
                adapter,
                preview_n,
                final_n,
                config,
            )
        else:
            # Fallback to mock metrics if no calibration data
            self._log_event(
                "eval",
                "warning",
                LogLevel.WARNING,
                {"message": "No calibration data provided, using mock metrics"},
            )
            # Provide a minimal primary_metric snapshot and basic perf counters
            metrics = {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 25.0,
                    "final": 26.0,
                },
                "latency_ms_per_tok": 15.0,
                "memory_mb_peak": 1024.0,
            }
            eval_windows = {"preview": {}, "final": {}}

        # Store metrics in report
        if hasattr(report, "metrics"):
            report.metrics.update(metrics)
        else:
            report.metrics = metrics

        report.evaluation_windows = eval_windows

        self._log_event("eval", "complete", LogLevel.INFO, {"metrics": metrics})

        return metrics

    def _compute_real_metrics(
        self,
        model: Any,
        calibration_data: Any,
        adapter: ModelAdapter,
        preview_n: int | None = None,
        final_n: int | None = None,
        config: RunConfig | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute real evaluation metrics using calibration data."""
        import os

        import psutil
        import torch

        _ = adapter  # Adapter kept for API parity; direct HF forward used.

        model.eval()

        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            print(
                f"[debug] compute_real_metrics preview_n={preview_n} final_n={final_n} calibration_len={len(calibration_data) if hasattr(calibration_data, '__len__') else 'n/a'}"
            )
        device = next(model.parameters()).device

        eval_device_override = os.environ.get("INVARLOCK_EVAL_DEVICE")
        if eval_device_override:
            override_device = torch.device(eval_device_override)
            if override_device != device:
                model.to(override_device)
                device = override_device

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        total_available = (
            len(calibration_data) if hasattr(calibration_data, "__len__") else 0
        )
        if total_available == 0:
            raise ValueError("Calibration data is empty; cannot compute metrics.")

        if preview_n is None:
            preview_n = max(total_available // 2, 1)
        if final_n is None:
            final_n = preview_n

        preview_n = max(int(preview_n), 0)
        final_n = max(int(final_n), 0)
        max_needed = max(preview_n, final_n)
        if max_needed <= 0:
            raise ValueError("preview_n and final_n cannot both be zero.")

        if max_needed > total_available:
            self._log_event(
                "eval",
                "data_scaled",
                LogLevel.WARNING,
                {
                    "requested_preview": preview_n,
                    "requested_final": final_n,
                    "available": total_available,
                },
            )
            preview_n = min(preview_n, total_available)
            final_n = min(final_n, total_available)
            max_needed = max(preview_n, final_n)

        requested_preview = preview_n
        requested_final = final_n

        max_needed = preview_n + final_n
        if max_needed > total_available:
            self._log_event(
                "eval",
                "window_shortage",
                LogLevel.WARNING,
                {
                    "requested_preview": preview_n,
                    "requested_final": final_n,
                    "available": total_available,
                },
            )
            max_needed = min(total_available, max_needed)

        preview_n = min(preview_n, total_available)
        final_start = preview_n
        remaining = max(total_available - preview_n, 0)
        if final_n > remaining:
            self._log_event(
                "eval",
                "final_window_shortage",
                LogLevel.WARNING,
                {
                    "requested_final": final_n,
                    "available_after_preview": remaining,
                    "requested_preview": requested_preview,
                    "requested_final_original": requested_final,
                },
            )
            final_n = remaining

        preview_data = calibration_data[:preview_n]
        final_data = calibration_data[final_start : final_start + final_n]

        eval_context: dict[str, Any] = {}
        if config and isinstance(config.context, dict):
            eval_context = config.context.get("eval", {}) or {}

        loss_cfg = (
            eval_context.get("loss", {}) if isinstance(eval_context, dict) else {}
        )
        resolved_loss_mode = str(
            loss_cfg.get("resolved_type") or loss_cfg.get("type") or ""
        ).lower()
        bootstrap_cfg = eval_context.get("bootstrap", {}) or {}
        bootstrap_enabled = bool(bootstrap_cfg.get("enabled", True))
        bootstrap_method = str(
            bootstrap_cfg.get("method", "bca_paired_delta_log")
        ).lower()
        bootstrap_replicates = int(
            bootstrap_cfg.get("replicates", bootstrap_cfg.get("n", 1000) or 1000)
        )
        bootstrap_alpha = float(bootstrap_cfg.get("alpha", 0.05) or 0.05)
        bootstrap_seed_cfg = bootstrap_cfg.get("seed")
        ci_band = float(bootstrap_cfg.get("ci_band", 0.10) or 0.10)

        single_method = "bca"
        delta_method = "bca"
        if bootstrap_method == "percentile":
            single_method = "percentile"
            delta_method = "percentile"
        elif bootstrap_method == "bca_paired_delta_log":
            single_method = "bca"
            delta_method = "bca"
        else:
            single_method = bootstrap_method
            delta_method = bootstrap_method

        dataset_seed = None
        profile_label = ""
        pairing_context: dict[str, Any] = {}
        if config and isinstance(config.context, dict):
            dataset_cfg = config.context.get("dataset", {})
            if isinstance(dataset_cfg, dict):
                dataset_seed = dataset_cfg.get("seed")
            profile_label = str(config.context.get("profile", "")).lower()
            pairing_context = config.context.get("pairing_baseline", {}) or {}

        bootstrap_seed = (
            bootstrap_seed_cfg if bootstrap_seed_cfg is not None else dataset_seed
        )
        try:
            bootstrap_seed = int(bootstrap_seed) if bootstrap_seed is not None else 0
        except (TypeError, ValueError):
            bootstrap_seed = 0

        if bootstrap_replicates <= 0:
            bootstrap_enabled = False
        if not (0.0 < bootstrap_alpha < 1.0):
            bootstrap_alpha = 0.05

        pm_preview = 50.0
        pm_final = 50.0
        pm_ratio = 1.0
        ratio_ci: tuple[float, float] = (pm_ratio, pm_ratio)
        preview_log_ci: tuple[float, float] = (
            math.log(pm_preview),
            math.log(pm_preview),
        )
        final_log_ci: tuple[float, float] = (math.log(pm_final), math.log(pm_final))
        delta_log_ci: tuple[float, float] = (0.0, 0.0)
        preview_mean_log = math.log(pm_preview)
        final_mean_log = math.log(pm_final)
        delta_mean_log = 0.0
        preview_log_losses: list[float] = []
        final_log_losses: list[float] = []
        preview_tokens_ct = 0
        final_tokens_ct = 0
        preview_batches_ct = 0
        final_batches_ct = 0
        window_overlap_fraction = 0.0
        # Defaults for pairing metrics to avoid unbound locals on error paths
        window_match_fraction = 1.0
        pairing_reason = None
        preview_pair_stats = {"matched": 0, "expected": 0}
        final_pair_stats = {"matched": 0, "expected": 0}
        preview_window_ids: list[int] = []
        final_window_ids: list[int] = []
        preview_tokens: list[list[int]] = []
        final_tokens: list[list[int]] = []
        preview_limit = min(preview_n, len(preview_data)) if preview_data else 0
        final_limit = min(final_n, len(final_data)) if final_data else 0

        # Safe defaults in case of early exceptions inside compute block
        preview_actual_tokens_ct = int(preview_tokens_ct)
        final_actual_tokens_ct = int(final_tokens_ct)
        preview_masked_total = int(preview_tokens_ct)
        final_masked_total = int(final_tokens_ct)
        preview_token_counts = []
        final_token_counts = []
        preview_attention_masks: list[list[int]] = []
        final_attention_masks: list[list[int]] = []
        preview_mask_counts: list[int] = []
        final_mask_counts: list[int] = []
        preview_labels: list[list[int]] = []
        final_labels: list[list[int]] = []
        preview_actual_token_counts: list[int] = []
        final_actual_token_counts: list[int] = []

        # Defaults for degeneracy flags
        degenerate_delta = False
        degenerate_reason: str | None = None

        bootstrap_info = {
            "enabled": bool(bootstrap_enabled),
            "method": bootstrap_method,
            "alpha": float(bootstrap_alpha),
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "ci_band": float(ci_band),
        }

        alignment_logged = False

        # Initialize to safe defaults to ensure later metrics assembly succeeds
        # even if an exception occurs during the main compute block.
        delta_samples: list[float] = []
        delta_weights: list[float] = []

        try:

            def _resolve_limit(batches: Sequence[Any], requested: int) -> int:
                if not batches:
                    return 0
                if requested <= 0:
                    return len(batches)
                return min(len(batches), requested)

            def _compute_slice_summary(
                batches: Sequence[Any],
                max_batches: int,
                start_idx: int,
            ) -> dict[str, Any]:
                nonlocal alignment_logged

                total_tokens_local = 0
                actual_tokens_local = 0
                weighted_log_loss = 0.0
                log_losses: list[float] = []
                window_ids: list[int] = []
                collected_tokens: list[list[int]] = []
                collected_attn: list[list[int]] = []
                collected_labels: list[list[int]] = []
                token_counts: list[int] = []
                masked_token_counts: list[int] = []
                actual_token_counts: list[int] = []
                count = 0
                zero_mask_batches = 0
                any_labels_seen = False
                store_windows = os.environ.get(
                    "INVARLOCK_STORE_EVAL_WINDOWS", "1"
                ).lower() not in {"0", "false", "no"}

                if not batches:
                    return {
                        "ppl": float("nan"),
                        "total_tokens": 0,
                        "num_batches": 0,
                        "log_losses": [],
                        "window_ids": [],
                        "tokens": [],
                        "attention_masks": [],
                        "weighted_log_loss": 0.0,
                        "window_token_counts": [],
                    }

                limit = _resolve_limit(batches, max_batches)

                for batch in batches[:limit]:
                    if max_batches > 0 and count >= max_batches:
                        break

                    labels = None
                    if isinstance(batch, dict):
                        input_ids = batch.get("input_ids", batch.get("inputs"))
                        attention_mask = batch.get("attention_mask")
                        labels = batch.get("labels")
                    else:
                        input_ids = batch
                        attention_mask = None

                    if input_ids is None:
                        continue

                    if isinstance(input_ids, torch.Tensor):
                        input_ids_t = input_ids.to(device=device, dtype=torch.long)
                    else:
                        input_ids_t = torch.as_tensor(
                            input_ids, device=device, dtype=torch.long
                        )

                    if input_ids_t.dim() == 1:
                        input_ids_t = input_ids_t.unsqueeze(0)

                    attn_t = None
                    if attention_mask is not None:
                        if isinstance(attention_mask, torch.Tensor):
                            attn_t = attention_mask.to(device=device, dtype=torch.long)
                        else:
                            attn_t = torch.as_tensor(
                                attention_mask, device=device, dtype=torch.long
                            )
                        if attn_t.dim() == 1:
                            attn_t = attn_t.unsqueeze(0)

                    if labels is not None:
                        any_labels_seen = True
                        if isinstance(labels, torch.Tensor):
                            labels_t = labels.to(device=device, dtype=torch.long)
                        else:
                            labels_t = torch.as_tensor(
                                labels, device=device, dtype=torch.long
                            )
                        if labels_t.dim() == 1:
                            labels_t = labels_t.unsqueeze(0)
                    else:
                        labels_t = input_ids_t.clone()
                        if attn_t is not None:
                            labels_t = labels_t.masked_fill(attn_t == 0, -100)

                    snapshot = input_ids_t.detach().cpu()
                    attn_snapshot = (
                        attn_t.detach().cpu() if attn_t is not None else None
                    )

                    with torch.no_grad():
                        if attn_t is not None:
                            outputs = model(
                                input_ids_t, attention_mask=attn_t, labels=labels_t
                            )
                        else:
                            outputs = model(input_ids_t, labels=labels_t)

                    loss_val = (
                        outputs.loss.item()
                        if hasattr(outputs, "loss") and hasattr(outputs.loss, "item")
                        else None
                    )
                    if loss_val is None:
                        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                            self._log_event(
                                "eval",
                                "missing_loss",
                                LogLevel.DEBUG,
                                {
                                    "has_loss_attr": bool(hasattr(outputs, "loss")),
                                    "labels_provided": bool(labels is not None),
                                    "window_index": start_idx + count,
                                },
                            )
                        continue

                    if attn_snapshot is not None:
                        tokens_in_batch = int(attn_snapshot.sum().item())
                    else:
                        tokens_in_batch = int(input_ids_t.numel())

                    if tokens_in_batch <= 0:
                        continue

                    masked_tokens_batch = int((labels_t != -100).sum().item())
                    effective_masked = masked_tokens_batch
                    if labels is not None and masked_tokens_batch <= 0:
                        zero_mask_batches += 1
                        effective_masked = tokens_in_batch
                        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                            sample_labels = None
                            try:
                                sample_labels = labels_t[0].detach().cpu().tolist()[:8]
                            except Exception:  # pragma: no cover - defensive
                                sample_labels = None
                            self._log_event(
                                "eval",
                                "zero_mask_batch",
                                LogLevel.WARNING,
                                {
                                    "window_index": start_idx + count,
                                    "tokens_in_batch": tokens_in_batch,
                                    "masked_tokens": masked_tokens_batch,
                                    "labels_sample": sample_labels,
                                    "fallback_weight": effective_masked,
                                },
                            )
                    effective_weight = (
                        effective_masked if labels is not None else tokens_in_batch
                    )
                    if effective_weight <= 0:
                        continue

                    if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                        print(
                            f"[debug] eval batch loss={float(loss_val):.6f} masked_tokens={masked_tokens_batch} tokens_in_batch={tokens_in_batch}"
                        )

                    if store_windows:
                        for row in snapshot:
                            collected_tokens.append(row.tolist())

                        if attn_snapshot is not None:
                            for row in attn_snapshot:
                                collected_attn.append(row.tolist())
                        else:
                            for row in snapshot:
                                collected_attn.append([1] * len(row))
                        collected_labels.extend(labels_t.detach().cpu().tolist())

                    if not alignment_logged:
                        self._log_event(
                            "eval",
                            "label_alignment",
                            LogLevel.INFO,
                            {
                                "ignore_index": -100,
                                "used_attention_mask": bool(attn_snapshot is not None),
                                "tokens_in_batch": tokens_in_batch,
                                "masked_tokens": masked_tokens_batch,
                            },
                        )
                        alignment_logged = True

                    log_losses.append(float(loss_val))
                    actual_tokens_local += tokens_in_batch
                    total_tokens_local += effective_weight
                    weighted_log_loss += float(loss_val) * effective_weight
                    token_counts.append(effective_weight)
                    masked_token_counts.append(masked_tokens_batch)
                    if labels is not None and masked_tokens_batch <= 0:
                        masked_token_counts[-1] = effective_masked
                    actual_token_counts.append(tokens_in_batch)
                    window_ids.append(start_idx + count)
                    count += 1

                if count == 0:
                    if zero_mask_batches and os.environ.get("INVARLOCK_DEBUG_TRACE"):
                        self._log_event(
                            "eval",
                            "zero_mask_total",
                            LogLevel.ERROR,
                            {
                                "zero_mask_batches": zero_mask_batches,
                                "requested": limit,
                            },
                        )
                    if resolved_loss_mode == "mlm":
                        error_msg = (
                            "MLM evaluation produced zero usable batches; "
                            "ensure baseline pairing includes masked tokens."
                        )
                        if any_labels_seen:
                            error_msg = (
                                "MLM evaluation saw labels but zero masked tokens were accumulated; "
                                "check calibration data integrity."
                            )
                        self._log_event(
                            "eval",
                            "mlm_missing_masks",
                            LogLevel.ERROR,
                            {
                                "any_labels": bool(any_labels_seen),
                                "requested": limit,
                                "zero_mask_batches": zero_mask_batches,
                            },
                        )
                        raise ValueError(error_msg)
                    return {
                        "ppl": float("nan"),
                        "total_tokens": total_tokens_local,
                        "actual_total_tokens": actual_tokens_local,
                        "num_batches": 0,
                        "log_losses": [],
                        "window_ids": [],
                        "tokens": [],
                        "attention_masks": [],
                        "weighted_log_loss": 0.0,
                        "window_token_counts": [],
                        "masked_token_counts": [],
                        "actual_token_counts": [],
                        "labels": [],
                    }

                mean_loss = (
                    weighted_log_loss / total_tokens_local
                    if total_tokens_local > 0
                    else sum(log_losses) / max(count, 1)
                )
                return {
                    "ppl": float(math.exp(mean_loss)),
                    "total_tokens": total_tokens_local,
                    "num_batches": count,
                    "log_losses": log_losses,
                    "window_ids": window_ids,
                    "tokens": collected_tokens,
                    "attention_masks": collected_attn,
                    "weighted_log_loss": weighted_log_loss,
                    "window_token_counts": token_counts,
                    "masked_token_counts": masked_token_counts,
                    "actual_token_counts": actual_token_counts,
                    "labels": collected_labels,
                    "actual_total_tokens": actual_tokens_local,
                }

            preview_limit = _resolve_limit(preview_data, preview_n)
            final_limit = _resolve_limit(final_data, final_n)

            preview_summary = _compute_slice_summary(preview_data, preview_limit, 0)
            final_summary = _compute_slice_summary(
                final_data, final_limit, preview_summary["num_batches"]
            )

            preview_log_losses = preview_summary["log_losses"]
            final_log_losses = final_summary["log_losses"]
            preview_tokens_ct = preview_summary["total_tokens"]
            final_tokens_ct = final_summary["total_tokens"]
            preview_batches_ct = preview_summary["num_batches"]
            final_batches_ct = final_summary["num_batches"]
            preview_window_ids = list(preview_summary["window_ids"])
            final_window_ids = list(final_summary["window_ids"])
            preview_tokens = list(preview_summary["tokens"])
            final_tokens = list(final_summary["tokens"])
            preview_token_counts = list(preview_summary.get("window_token_counts", []))
            final_token_counts = list(final_summary.get("window_token_counts", []))
            preview_attention_masks = list(preview_summary.get("attention_masks", []))
            final_attention_masks = list(final_summary.get("attention_masks", []))
            preview_mask_counts = list(preview_summary.get("masked_token_counts", []))
            final_mask_counts = list(final_summary.get("masked_token_counts", []))
            preview_labels = list(preview_summary.get("labels", []))
            final_labels = list(final_summary.get("labels", []))
            preview_actual_token_counts = list(
                preview_summary.get("actual_token_counts", [])
            )
            final_actual_token_counts = list(
                final_summary.get("actual_token_counts", [])
            )
            preview_actual_tokens_ct = int(
                preview_summary.get("actual_total_tokens", preview_tokens_ct)
            )
            final_actual_tokens_ct = int(
                final_summary.get("actual_total_tokens", final_tokens_ct)
            )
            preview_masked_total = (
                sum(preview_mask_counts)
                if preview_mask_counts
                else int(preview_tokens_ct)
            )
            final_masked_total = (
                sum(final_mask_counts) if final_mask_counts else int(final_tokens_ct)
            )
            preview_weighted_loss = float(preview_summary.get("weighted_log_loss", 0.0))
            final_weighted_loss = float(final_summary.get("weighted_log_loss", 0.0))

            if preview_tokens_ct > 0:
                preview_mean_log = float(preview_weighted_loss / preview_tokens_ct)
                pm_preview = math.exp(preview_mean_log)
            elif preview_log_losses:
                preview_mean_log = float(np.mean(preview_log_losses))
                pm_preview = math.exp(preview_mean_log)
            else:
                pm_preview = preview_summary["ppl"]
                if not math.isfinite(pm_preview) or pm_preview <= 0:
                    pm_preview = 50.0
                preview_mean_log = math.log(pm_preview)

            if final_tokens_ct > 0:
                final_mean_log = float(final_weighted_loss / final_tokens_ct)
                pm_final = math.exp(final_mean_log)
            elif final_log_losses:
                final_mean_log = float(np.mean(final_log_losses))
                pm_final = math.exp(final_mean_log)
            else:
                pm_final = final_summary["ppl"]
                if not math.isfinite(pm_final) or pm_final <= 0:
                    pm_final = 50.0
                final_mean_log = math.log(pm_final)

            delta_mean_log = final_mean_log - preview_mean_log
            pm_ratio = math.exp(delta_mean_log)

            if not (math.isfinite(delta_mean_log) and math.isfinite(pm_ratio)):
                raise RuntimeError("Invalid perplexity ratio or delta")

            expected_ratio = math.exp(delta_mean_log)
            if abs(pm_ratio - expected_ratio) > 1e-6:
                raise RuntimeError(
                    "Primary-metric ratio mismatch with exp(mean ΔlogNLL)"
                )

            if bootstrap_enabled and preview_log_losses:
                preview_log_ci = compute_logloss_ci(
                    preview_log_losses,
                    method=single_method,
                    replicates=bootstrap_replicates,
                    alpha=bootstrap_alpha,
                    seed=bootstrap_seed + 7,
                )
            else:
                preview_log_ci = (preview_mean_log, preview_mean_log)

            # primary_metric consumers use log-space intervals; skip ppl-space tuple here

            if bootstrap_enabled and final_log_losses:
                final_log_ci = compute_logloss_ci(
                    final_log_losses,
                    method=single_method,
                    replicates=bootstrap_replicates,
                    alpha=bootstrap_alpha,
                    seed=bootstrap_seed + 13,
                )
            else:
                final_log_ci = (final_mean_log, final_mean_log)

            # primary_metric consumers use log-space intervals; skip ppl-space tuple here

            if (
                bootstrap_enabled
                and final_log_losses
                and preview_log_losses
                and len(final_log_losses)
                and len(preview_log_losses)
            ):
                delta_log_ci = compute_paired_delta_log_ci(
                    final_log_losses,
                    preview_log_losses,
                    method=delta_method,
                    replicates=bootstrap_replicates,
                    alpha=bootstrap_alpha,
                    seed=bootstrap_seed + 97,
                )
                ratio_ci = logspace_to_ratio_ci(delta_log_ci)
                expected_ratio_ci = tuple(math.exp(bound) for bound in delta_log_ci)
                if any(
                    abs(r - e) > 1e-6
                    for r, e in zip(ratio_ci, expected_ratio_ci, strict=False)
                ):
                    raise RuntimeError("Ratio CI inconsistent with Δlog CI")
            else:
                delta_log_ci = (delta_mean_log, delta_mean_log)
                ratio_ci = (pm_ratio, pm_ratio)

            delta_samples: list[float] = []
            delta_weights: list[float] = []
            if final_log_losses and preview_log_losses:
                limit = min(len(final_log_losses), len(preview_log_losses))
                if limit:
                    delta_samples = [
                        final_log_losses[i] - preview_log_losses[i]
                        for i in range(limit)
                    ]
                    if preview_token_counts and len(preview_token_counts) >= limit:
                        delta_weights = [
                            float(max(preview_token_counts[i], 1)) for i in range(limit)
                        ]

            degenerate_delta = False
            degenerate_reason: str | None = None
            if len(delta_samples) < 2:
                if len(delta_samples) == 0:
                    degenerate_delta = True
                    degenerate_reason = "no_pairs"
                else:
                    degenerate_delta = True
                    degenerate_reason = "single_pair"
            elif np.allclose(delta_samples, delta_samples[0]):
                degenerate_delta = True
                degenerate_reason = "no_variation"

            if degenerate_delta:
                self._log_event(
                    "eval",
                    "degenerate_delta_samples",
                    LogLevel.ERROR,
                    {
                        "reason": degenerate_reason,
                        "sample_count": len(delta_samples),
                    },
                )
                if profile_label in {"ci", "release"}:
                    raise RuntimeError(
                        f"Degenerate paired ΔlogNLL distribution ({degenerate_reason})"
                    )

            def _hash_tokens(tokens: list[int]) -> bytes:
                if not tokens:
                    return b""
                token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
                return hashlib.blake2b(token_array.tobytes(), digest_size=16).digest()

            def _duplicate_fraction(seqs: list[list[int]]) -> float:
                if not seqs:
                    return 0.0
                hashes = [_hash_tokens(seq) for seq in seqs]
                unique = len(set(hashes))
                if not hashes:
                    return 0.0
                return max(0.0, (len(hashes) - unique) / len(hashes))

            def _compare_with_baseline(
                run_ids: list[int],
                run_tokens: list[list[int]],
                baseline_section: dict[str, Any] | None,
                split_label: str,
            ) -> dict[str, Any]:
                stats = {
                    "matched": 0,
                    "expected": 0,
                    "missing_ids": [],
                    "mismatched_ids": [],
                    "unexpected_ids": [],
                    "reason": None,
                }

                if not baseline_section:
                    stats["matched"] = len(run_tokens)
                    stats["expected"] = len(run_tokens)
                    stats["reason"] = "no_baseline_reference"
                    return stats

                base_ids = baseline_section.get("window_ids") or []
                base_tokens = baseline_section.get("input_ids") or []
                if not isinstance(base_ids, list) or not isinstance(base_tokens, list):
                    stats["matched"] = len(run_tokens)
                    stats["expected"] = len(run_tokens)
                    stats["reason"] = "invalid_baseline_reference"
                    return stats

                base_map: dict[int, bytes] = {}
                for bid, seq in zip(base_ids, base_tokens, strict=False):
                    try:
                        bid_int = int(bid)
                    except Exception:
                        continue
                    seq_list = list(seq) if not isinstance(seq, list) else seq
                    base_map[bid_int] = _hash_tokens(seq_list)

                stats["expected"] = len(base_map)
                matched = 0
                seen_ids: set[int] = set()
                mismatched: list[int] = []
                unexpected: list[int] = []

                for rid, seq in zip(run_ids, run_tokens, strict=False):
                    try:
                        rid_int = int(rid)
                    except Exception:
                        unexpected.append(rid)
                        continue

                    hashed = _hash_tokens(seq)
                    if rid_int not in base_map:
                        unexpected.append(rid_int)
                        continue

                    seen_ids.add(rid_int)
                    if hashed == base_map[rid_int]:
                        matched += 1
                    else:
                        mismatched.append(rid_int)

                missing = [bid for bid in base_map if bid not in seen_ids]
                stats.update(
                    {
                        "matched": matched,
                        "missing_ids": missing,
                        "mismatched_ids": mismatched,
                        "unexpected_ids": unexpected,
                    }
                )

                if missing:
                    stats["reason"] = f"{split_label}_missing_ids:{missing[:3]}"
                elif mismatched:
                    stats["reason"] = f"{split_label}_token_mismatch:{mismatched[:3]}"
                elif unexpected:
                    stats["reason"] = f"{split_label}_unexpected_ids:{unexpected[:3]}"
                else:
                    stats["reason"] = None

                return stats

            baseline_preview = (
                pairing_context.get("preview")
                if isinstance(pairing_context, dict)
                else {}
            )
            baseline_final = (
                pairing_context.get("final")
                if isinstance(pairing_context, dict)
                else {}
            )

            preview_pair_stats = _compare_with_baseline(
                preview_window_ids, preview_tokens, baseline_preview, "preview"
            )
            final_pair_stats = _compare_with_baseline(
                final_window_ids, final_tokens, baseline_final, "final"
            )

            total_expected = (
                preview_pair_stats["expected"] + final_pair_stats["expected"]
            )
            total_matched = preview_pair_stats["matched"] + final_pair_stats["matched"]
            window_match_fraction = (
                float(total_matched / total_expected) if total_expected > 0 else 1.0
            )
            window_overlap_fraction = _duplicate_fraction(preview_tokens + final_tokens)

            pairing_reason = None
            if total_expected > 0:
                for stats_dict, label in (
                    (preview_pair_stats, "preview"),
                    (final_pair_stats, "final"),
                ):
                    if (
                        stats_dict["expected"]
                        and stats_dict["matched"] < stats_dict["expected"]
                    ):
                        pairing_reason = stats_dict.get("reason") or f"{label}_mismatch"
                        break
            if pairing_reason is None:
                if window_overlap_fraction > 0.0:
                    pairing_reason = "duplicate_windows"
                elif not pairing_context:
                    pairing_reason = preview_pair_stats.get(
                        "reason"
                    ) or final_pair_stats.get("reason")

            if pairing_context and window_match_fraction < 0.999999:
                self._log_event(
                    "eval",
                    "window_pairing_mismatch",
                    LogLevel.ERROR,
                    {
                        "match_fraction": window_match_fraction,
                        "overlap_fraction": window_overlap_fraction,
                        "reason": pairing_reason,
                        "preview": preview_pair_stats,
                        "final": final_pair_stats,
                    },
                )

            if window_overlap_fraction > 0.0 and pairing_context:
                self._log_event(
                    "eval",
                    "window_overlap_warning",
                    LogLevel.WARNING,
                    {
                        "duplicate_fraction": window_overlap_fraction,
                        "match_fraction": window_match_fraction,
                        "preview": preview_pair_stats,
                        "final": final_pair_stats,
                    },
                )

            if pairing_context and profile_label in {"ci", "release"}:
                if window_match_fraction < 0.999999:
                    raise RuntimeError(
                        f"Window pairing mismatch detected (fraction={window_match_fraction:.3f}, reason={pairing_reason})"
                    )
                if window_overlap_fraction > 0.0:
                    raise RuntimeError(
                        f"Window duplication detected (overlap_fraction={window_overlap_fraction:.3f})"
                    )

            tier = "balanced"
            if config and isinstance(config.context, dict):
                auto_section = config.context.get("auto", {})
                if isinstance(auto_section, dict):
                    tier = str(auto_section.get("tier", tier)).lower()

            coverage_requirements = BOOTSTRAP_COVERAGE_REQUIREMENTS.get(
                tier, BOOTSTRAP_COVERAGE_REQUIREMENTS["balanced"]
            )

            def _meets_requirement(actual: int, required: int) -> bool:
                if required <= 0:
                    return True
                slack = max(1, int(required * 0.95))
                return actual >= slack

            preview_required = int(coverage_requirements.get("preview", 0))
            final_required = int(coverage_requirements.get("final", 0))
            replicates_required = int(coverage_requirements.get("replicates", 0))

            preview_ok = _meets_requirement(preview_batches_ct, preview_required)
            final_ok = _meets_requirement(final_batches_ct, final_required)
            replicates_ok = (
                _meets_requirement(bootstrap_replicates, replicates_required)
                if bootstrap_enabled
                else True
            )

            if not (preview_ok and final_ok and replicates_ok):
                self._log_event(
                    "eval",
                    "bootstrap_coverage_warning",
                    LogLevel.WARNING,
                    {
                        "tier": tier,
                        "preview_used": preview_batches_ct,
                        "preview_required": preview_required,
                        "final_used": final_batches_ct,
                        "final_required": final_required,
                        "replicates_used": bootstrap_replicates,
                        "replicates_required": replicates_required,
                    },
                )
                # In CI/Release profiles, treat insufficient coverage as a hard error
                if pairing_context and profile_label in {"ci", "release"}:
                    from invarlock.cli.errors import InvarlockError

                    raise InvarlockError(
                        code="E005",
                        message=(
                            "INSUFFICIENT-SAMPLE: bootstrap coverage below policy floors in CI/Release"
                        ),
                    )

            bootstrap_info.update(
                {
                    "enabled": bool(bootstrap_enabled),
                    "method": bootstrap_method,
                    "alpha": float(bootstrap_alpha),
                    "replicates": int(bootstrap_replicates),
                    "seed": int(bootstrap_seed),
                    "ci_band": float(ci_band),
                    "window_duplicate_fraction": float(window_overlap_fraction),
                    "window_match_fraction": float(window_match_fraction),
                    "coverage": {
                        "tier": tier,
                        "preview": {
                            "used": int(preview_batches_ct),
                            "required": preview_required,
                            "ok": bool(preview_ok),
                        },
                        "final": {
                            "used": int(final_batches_ct),
                            "required": final_required,
                            "ok": bool(final_ok),
                        },
                        "replicates": {
                            "used": int(bootstrap_replicates),
                            "required": replicates_required,
                            "ok": bool(replicates_ok),
                        },
                    },
                }
            )

        except Exception as exc:  # pragma: no cover - defensive fallback
            self._log_event(
                "eval",
                "error",
                LogLevel.ERROR,
                {"message": f"Primary-metric computation failed: {exc}"},
            )

        pm_ratio = pm_final / pm_preview if pm_preview > 0 else 1.0

        latency_ms_per_tok = self._measure_latency(
            model, preview_data[:1] if preview_data else final_data[:1], device
        )

        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(initial_memory, current_memory)

        eval_samples = 0
        total_tokens = 0
        masked_total_tokens = 0
        try:
            eval_samples = int(preview_batches_ct) + int(final_batches_ct)
            total_tokens = int(preview_actual_tokens_ct) + int(final_actual_tokens_ct)
            masked_total_tokens = int(preview_masked_total) + int(final_masked_total)
        except Exception:
            pass

        paired_windows_count = len(delta_samples)
        unweighted_delta_mean = (
            float(np.mean(delta_samples)) if delta_samples else float(delta_mean_log)
        )
        preview_weighted_delta_mean: float | None = None
        if delta_weights:
            total_weight = float(sum(delta_weights))
            if total_weight > 0.0:
                preview_weighted_delta_mean = float(
                    np.dot(delta_samples, delta_weights) / total_weight
                )
        paired_delta_mean = float(delta_mean_log)
        paired_delta_std = (
            float(np.std(delta_samples, ddof=1)) if len(delta_samples) > 1 else 0.0
        )
        paired_delta_min = float(min(delta_samples)) if delta_samples else None
        paired_delta_max = float(max(delta_samples)) if delta_samples else None

        # Resolve primary metric kind from resolved loss mode
        pm_kind = "ppl_causal"
        if resolved_loss_mode == "mlm":
            pm_kind = "ppl_mlm"
        elif resolved_loss_mode in {"seq2seq", "s2s", "t5"}:
            pm_kind = "ppl_seq2seq"

        metrics = {
            "primary_metric": {
                "kind": pm_kind,
                "preview": float(pm_preview),
                "final": float(pm_final),
            },
            "logloss_preview": float(preview_mean_log),
            "logloss_final": float(final_mean_log),
            "logloss_delta": float(delta_mean_log),
            "logloss_preview_ci": tuple(map(float, preview_log_ci)),
            "logloss_final_ci": tuple(map(float, final_log_ci)),
            "logloss_delta_ci": tuple(map(float, delta_log_ci)),
            "latency_ms_per_tok": latency_ms_per_tok,
            "memory_mb_peak": peak_memory,
            "eval_samples": eval_samples,
            "total_tokens": total_tokens,
            "preview_total_tokens": int(preview_actual_tokens_ct),
            "final_total_tokens": int(final_actual_tokens_ct),
            "masked_tokens_total": masked_total_tokens,
            "masked_tokens_preview": int(preview_masked_total),
            "masked_tokens_final": int(final_masked_total),
            "reduction": {
                "mode": "token_mean",
                "implementation": "huggingface_cross_entropy",
            },
            "window_overlap_fraction": float(window_overlap_fraction),
            "window_match_fraction": float(window_match_fraction),
            "window_pairing_reason": pairing_reason,
            "window_pairing_preview": {
                "matched": preview_pair_stats["matched"],
                "expected": preview_pair_stats["expected"],
                "reason": preview_pair_stats.get("reason"),
            },
            "window_pairing_final": {
                "matched": final_pair_stats["matched"],
                "expected": final_pair_stats["expected"],
                "reason": final_pair_stats.get("reason"),
            },
            "bootstrap": bootstrap_info,
            "paired_windows": paired_windows_count,
            "paired_delta_summary": {
                "mean": paired_delta_mean,
                "mean_unweighted": unweighted_delta_mean,
                "mean_preview_weighted": (
                    preview_weighted_delta_mean
                    if preview_weighted_delta_mean is not None
                    else unweighted_delta_mean
                ),
                "std": paired_delta_std,
                "min": paired_delta_min,
                "max": paired_delta_max,
                "degenerate": degenerate_delta,
                "degenerate_reason": degenerate_reason,
            },
        }

        eval_windows = {
            "preview": {
                "window_ids": preview_window_ids[:preview_limit],
                "logloss": list(preview_log_losses),
                "input_ids": preview_tokens,
                "attention_masks": preview_attention_masks,
                "token_counts": preview_token_counts,
                "masked_token_counts": preview_mask_counts,
                "actual_token_counts": preview_actual_token_counts,
                "labels": preview_labels,
            },
            "final": {
                "window_ids": final_window_ids[:final_limit],
                "logloss": list(final_log_losses),
                "input_ids": final_tokens,
                "attention_masks": final_attention_masks,
                "token_counts": final_token_counts,
                "masked_token_counts": final_mask_counts,
                "actual_token_counts": final_actual_token_counts,
                "labels": final_labels,
            },
        }

        return metrics, eval_windows

    def _measure_latency(self, model: Any, sample_data: Any, device: Any) -> float:
        """Simple latency measurement for a sample."""
        import time

        import torch

        if not sample_data:
            return 0.0

        # Model eval is managed by caller to avoid duplicate invocations in tests

        # Get a sample for timing
        sample = sample_data[0] if sample_data else None
        if sample is None:
            return 0.0

        if isinstance(sample, dict):
            input_ids = sample.get("input_ids", sample.get("inputs"))
        else:
            input_ids = sample

        if input_ids is None:
            return 0.0

        # Convert to tensor if needed
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        # Some tests patch torch.tensor with dim side effects; guard against exceptions
        try:
            dim_val = input_ids.dim()
        except Exception:
            dim_val = 2  # assume already batched
        if dim_val == 1:
            try:
                input_ids = input_ids.unsqueeze(0)
            except Exception:
                pass

        # to(device) may be a Mock; guard call
        try:
            input_ids = input_ids.to(device)
        except Exception:
            pass

        # Simple timing measurement
        with torch.no_grad():
            try:
                # Prepare labels and attention mask if available
                labels_t = input_ids
                attn_t = None
                token_type_t = None
                if isinstance(sample, dict) and "attention_mask" in sample:
                    try:
                        attn_t = torch.tensor(sample["attention_mask"])
                        try:
                            if attn_t.dim() == 1:
                                attn_t = attn_t.unsqueeze(0)
                        except Exception:
                            pass
                        try:
                            attn_t = attn_t.to(device)
                        except Exception:
                            pass
                    except Exception:
                        attn_t = None
                if isinstance(sample, dict) and "token_type_ids" in sample:
                    try:
                        token_type_t = torch.tensor(sample["token_type_ids"])
                        if token_type_t.dim() == 1:
                            token_type_t = token_type_t.unsqueeze(0)
                        token_type_t = token_type_t.to(device)
                    except Exception:
                        token_type_t = None

                def _call_model():
                    if attn_t is not None:
                        return model(
                            input_ids,
                            attention_mask=attn_t,
                            labels=labels_t,
                            token_type_ids=token_type_t,
                        )
                    else:
                        return model(
                            input_ids,
                            labels=labels_t,
                            token_type_ids=token_type_t,
                        )

                # Warmup
                for _ in range(3):
                    _ = _call_model()

                # Measure
                start_time = time.time()
                for _ in range(10):
                    _ = _call_model()
                end_time = time.time()

                total_time = (end_time - start_time) * 1000  # Convert to ms
                try:
                    total_tokens = input_ids.numel() * 10  # 10 iterations
                except Exception:
                    total_tokens = 0

                return total_time / total_tokens if total_tokens > 0 else 0.0

            except Exception:
                return 0.0

    def _samples_to_dataloader(self, samples: list) -> Any:
        """
        Convert list of samples to DataLoader-compatible format.

        Args:
            samples: List of sample dictionaries with 'input_ids' and 'attention_mask'

        Returns:
            Simple iterable that yields batches compatible with compute_perplexity()
        """

        class SampleDataLoader:
            def __init__(self, samples):
                self.samples = samples

            def __iter__(self):
                for sample in self.samples:
                    # Each sample is already a dict with 'input_ids' and 'attention_mask'
                    # Convert to tensor format that compute_perplexity expects
                    import torch

                    input_ids = sample.get("input_ids", sample.get("inputs"))
                    attention_mask = sample.get("attention_mask")

                    if input_ids is None:
                        continue

                    # Convert to tensors if needed and add batch dimension
                    if not isinstance(input_ids, torch.Tensor):
                        input_ids = torch.tensor(input_ids, dtype=torch.long)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)

                    if attention_mask is not None:
                        if not isinstance(attention_mask, torch.Tensor):
                            attention_mask = torch.tensor(
                                attention_mask, dtype=torch.long
                            )
                        if attention_mask.dim() == 1:
                            attention_mask = attention_mask.unsqueeze(0)

                    batch = {"input_ids": input_ids}
                    if attention_mask is not None:
                        batch["attention_mask"] = attention_mask

                    token_type = sample.get("token_type_ids")
                    if token_type is not None:
                        if not isinstance(token_type, torch.Tensor):
                            token_type = torch.tensor(token_type, dtype=torch.long)
                        if token_type.dim() == 1:
                            token_type = token_type.unsqueeze(0)
                        batch["token_type_ids"] = token_type

                    labels = sample.get("labels")
                    if labels is None:
                        labels = input_ids.clone()
                        if attention_mask is not None:
                            labels = labels.masked_fill(attention_mask == 0, -100)
                    else:
                        if not isinstance(labels, torch.Tensor):
                            labels = torch.tensor(labels, dtype=torch.long)
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(0)
                    batch["labels"] = labels

                    yield batch

            def __len__(self):
                return len(self.samples)

        return SampleDataLoader(samples)

    def _finalize_phase(
        self,
        model: Any,
        adapter: ModelAdapter,
        guard_results: dict[str, dict[str, Any]],
        metrics: dict[str, Any],
        config: RunConfig,
        report: RunReport,
    ) -> str:
        """Phase 5: Finalize or rollback based on results."""
        self._log_event("finalize", "start", LogLevel.INFO)

        # Check if guards passed
        all_guards_passed = all(r.get("passed", False) for r in guard_results.values())

        # Check for catastrophic drift spike using primary metric preview/final
        pm = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
        pm_prev = pm.get("preview") if isinstance(pm, dict) else None
        pm_fin = pm.get("final") if isinstance(pm, dict) else None
        try:
            drift_ratio = (
                float(pm_fin) / float(pm_prev)
                if isinstance(pm_fin, (int | float))
                and isinstance(pm_prev, (int | float))
                and float(pm_prev) > 0.0
                else float("inf")
            )
        except Exception:
            drift_ratio = float("inf")
        spike_threshold = getattr(config, "spike_threshold", 2.0)
        is_catastrophic_spike = drift_ratio > spike_threshold

        # Check if standard metrics are acceptable against configured max ratio
        metrics_acceptable = drift_ratio <= getattr(config, "max_pm_ratio", 2.0)

        # Determine rollback reason and status
        rollback_reason = None
        if is_catastrophic_spike:
            rollback_reason = (
                f"catastrophic_ppl_spike (ratio: {drift_ratio:.3f} > {spike_threshold})"
            )
            status = RunStatus.ROLLBACK.value

            self._log_event(
                "finalize",
                "catastrophic_spike_detected",
                LogLevel.ERROR,
                {
                    "primary_metric_drift_ratio": drift_ratio,
                    "spike_threshold": spike_threshold,
                    "immediate_rollback": True,
                },
            )
        elif (not all_guards_passed) or (not metrics_acceptable):
            # Match historical/test expectation string exactly
            rollback_reason = "guards_failed or metrics_unacceptable"
            status = RunStatus.ROLLBACK.value
        else:
            status = RunStatus.SUCCESS.value

        # Execute the determined action
        if status == RunStatus.SUCCESS.value:
            self._log_event(
                "finalize",
                "success",
                LogLevel.INFO,
                {
                    "guards_passed": all_guards_passed,
                    "metrics_ok": metrics_acceptable,
                },
            )
        else:
            # Perform rollback if checkpoint available
            if self.checkpoint_manager and "initial_checkpoint" in report.meta:
                checkpoint_id = report.meta["initial_checkpoint"]
                self.checkpoint_manager.restore_checkpoint(
                    model, adapter, checkpoint_id
                )
                # Match test expectation: only include checkpoint and reason
                self._log_event(
                    "finalize",
                    "rollback",
                    LogLevel.WARNING,
                    {
                        "checkpoint": checkpoint_id,
                        "reason": rollback_reason,
                    },
                )

                # Store rollback metadata in report
                report.meta["rollback_reason"] = rollback_reason
                report.meta["rollback_checkpoint"] = checkpoint_id
                report.meta["guard_recovered"] = True

            else:
                # Match test expectation: log without additional data payload
                self._log_event("finalize", "rollback_unavailable", LogLevel.ERROR)

        return status

    def _handle_error(self, error: Exception, report: RunReport) -> None:
        """Handle pipeline errors."""
        report.status = RunStatus.FAILED.value
        report.error = str(error)
        report.meta["end_time"] = time.time()

        if "start_time" in report.meta:
            report.meta["duration"] = (
                report.meta["end_time"] - report.meta["start_time"]
            )

        self._log_event("runner", "error", LogLevel.ERROR, {"error": str(error)})

        # Attempt rollback on error
        if self.checkpoint_manager and "initial_checkpoint" in report.meta:
            try:
                checkpoint_id = report.meta["initial_checkpoint"]
                # Would need model and adapter here for actual rollback
                self._log_event(
                    "runner",
                    "emergency_rollback",
                    LogLevel.WARNING,
                    {"checkpoint": checkpoint_id},
                )
            except Exception as rollback_error:
                self._log_event(
                    "runner",
                    "rollback_failed",
                    LogLevel.CRITICAL,
                    {"error": str(rollback_error)},
                )

    def _resolve_guard_policies(
        self, report: RunReport, auto_config: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Resolve tier-based guard policies from configuration."""
        # Use passed auto_config if available, otherwise extract from report meta
        if auto_config is None:
            config_meta = report.meta.get("config", {})

            # Try to get auto config from various possible locations
            if hasattr(report, "auto_config"):
                auto_config = report.auto_config
            elif "auto" in config_meta:
                auto_config = config_meta["auto"]
            else:
                # Fallback to default balanced tier
                auto_config = {"tier": "balanced", "enabled": True}

        # Extract tier and edit name
        tier = auto_config.get("tier", "balanced")
        edit_name = None
        if hasattr(report, "edit") and report.edit:
            edit_name = report.edit.get("name")

        # Also try to get edit name from stored edit result in meta
        if not edit_name and "edit_name" in report.meta:
            edit_name = report.meta["edit_name"]

        # Get explicit guard overrides from config
        config_meta = report.meta.get("config", {})
        explicit_overrides = config_meta.get("guards", {})

        try:
            # Resolve tier policies
            policies = resolve_tier_policies(tier, edit_name, explicit_overrides)

            self._log_event(
                "auto_tuning",
                "tier_resolved",
                LogLevel.INFO,
                {"tier": tier, "edit": edit_name, "policies_count": len(policies)},
            )

            return policies

        except Exception as e:
            self._log_event(
                "auto_tuning",
                "tier_resolution_failed",
                LogLevel.ERROR,
                {"tier": tier, "error": str(e)},
            )
            # Return empty policies dict on failure
            return {}

    def _apply_guard_policy(self, guard: Guard, policy: dict[str, Any]) -> None:
        """Apply resolved policy parameters to a guard instance."""
        try:
            # Apply policy parameters to guard
            for param_name, param_value in policy.items():
                if hasattr(guard, param_name):
                    setattr(guard, param_name, param_value)
                elif hasattr(guard, "config") and isinstance(guard.config, dict):
                    # Try to set in guard's config dict
                    guard.config[param_name] = param_value
                elif hasattr(guard, "policy") and isinstance(guard.policy, dict):
                    # Try to set in guard's policy dict
                    guard.policy[param_name] = param_value
                else:
                    # Last resort: add to guard as attribute
                    setattr(guard, param_name, param_value)

        except Exception as e:
            self._log_event(
                "auto_tuning",
                "policy_application_failed",
                LogLevel.WARNING,
                {"guard": guard.name, "policy": policy, "error": str(e)},
            )

    def _log_event(
        self,
        component: str,
        operation: str,
        level: LogLevel,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Log an event if event logger is available."""
        if self.event_logger:
            self.event_logger.log(component, operation, level, data)

    def _serialize_config(self, config: RunConfig) -> dict[str, Any]:
        """Serialize RunConfig for storage in report."""
        return {
            "device": config.device,
            "max_pm_ratio": config.max_pm_ratio,
            "checkpoint_interval": config.checkpoint_interval,
            "dry_run": config.dry_run,
            "verbose": config.verbose,
            "guards": config.context.get("guards", {}) if config.context else {},
        }
