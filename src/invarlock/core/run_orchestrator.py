"""Typed run orchestration owner for config-driven run commands."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from invarlock.core.exceptions import ConfigError, InvarlockError
from invarlock.core.run_execution_context_policy import (
    build_run_context_payload as _build_run_context_payload_impl,
)
from invarlock.core.run_execution_context_policy import (
    build_run_execution_config_payloads as _build_run_execution_config_payloads_impl,
)
from invarlock.core.run_policy import (
    resolve_guard_overhead_threshold as _resolve_guard_overhead_threshold_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_acceptance_range as _resolve_pm_acceptance_range_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_drift_band as _resolve_pm_drift_band_impl,
)
from invarlock.core.run_policy import (
    should_measure_overhead as _should_measure_overhead_impl,
)
from invarlock.core.run_retry_policy import (
    build_restore_failure_attempt_summary as _build_restore_failure_attempt_summary_impl,
)
from invarlock.core.run_retry_policy import (
    decide_failed_retry_transition as _decide_failed_retry_transition_impl,
)
from invarlock.core.run_retry_policy import (
    record_retry_attempt as _record_retry_attempt_impl,
)
from invarlock.core.run_retry_policy import (
    resolve_retry_validation_transition as _resolve_retry_validation_transition_impl,
)
from invarlock.core.run_timing_policy import (
    build_timing_summary_payload as _build_timing_summary_payload_impl,
)
from invarlock.core.run_orchestrator_types import (
    RunAdapterSelectedEvent,
    RunAttemptStartedEvent,
    RunAutoTuneAdjustmentEvent,
    RunBaselineScheduleLoadedEvent,
    RunCalibrationBatchSizesDebugEvent,
    RunCleanupStatusEvent,
    RunConfigLoadedEvent,
    RunConfigLoadingEvent,
    RunContextEvent,
    RunDatasetLoadingEvent,
    RunDeterministicSeedsEvent,
    RunDeviceResolvedEvent,
    RunDiagnosticEvent,
    RunEditSelectedEvent,
    RunExecutePipelineEvent,
    RunEvaluationReportFailedEvent,
    RunEvaluationReportPassedEvent,
    RunEvaluationReportStartedEvent,
    RunExecutionEvent,
    RunExecutionFailure,
    RunExecutionObserver,
    RunExecutionOutcome,
    RunExecutionRequest,
    RunExecutionResult,
    RunExecutionServices,
    RunFailureEvent,
    RunGuardChainResolvedEvent,
    RunGuardOverheadSummaryEvent,
    RunLoadModelOnceEvent,
    RunMaskedTokensDebugEvent,
    RunOutputDirectoryReadyEvent,
    RunPipelineStartedEvent,
    RunPreviewLabelsDebugEvent,
    RunPrimaryMetricSummaryEvent,
    RunRetryAttemptStartedEvent,
    RunRetryExhaustedEvent,
    RunRetrySummaryEvent,
    RunRetryValidationErrorEvent,
    RunSnapshotModeEvent,
    RunTelemetryFailedEvent,
    RunTelemetrySavedEvent,
    TimingSummaryPayload,
    _RunExecutionHalt,
)
from invarlock.model_utils import set_seed

# class RunExecutionEvent
# class RunLifecycleEvent
# class RunDiagnosticEvent
# class RunContextEvent
# class RunAggregateEvent
# class RunFailureEvent
# class RunPrimaryMetricSummaryEvent
# Typed contracts live in `invarlock.core.run_orchestrator_types` and are
# re-exported here so the owner boundary remains stable.


def _coerce_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return float(default)
    return coerced if math.isfinite(coerced) else float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def execute_run_request(
    request: RunExecutionRequest,
    *,
    services: RunExecutionServices,
    observer: RunExecutionObserver | None = None,
) -> RunExecutionOutcome:
    """Execute a config-driven run and return a typed outcome/event stream."""

    config = request.config
    device = request.device
    profile = request.profile
    out = request.out
    edit = request.edit
    edit_label = request.edit_label
    tier = request.tier
    metric_kind = request.metric_kind
    probes = request.probes
    until_pass = request.until_pass
    max_attempts = request.max_attempts
    timeout = request.timeout
    baseline = request.baseline
    no_cleanup = request.no_cleanup
    capture_timings = request.capture_timings
    telemetry = request.telemetry
    prefer_local_files_only = request.prefer_local_files_only
    eval_device_override = request.eval_device_override
    determinism_mode = request.determinism_mode
    determinism_warn_only = request.determinism_warn_only
    tiny_relax_enabled = request.tiny_relax_enabled
    export_model_requested = request.export_model_requested
    export_dir_override = request.export_dir

    CONFIG_VALUE_EXCEPTIONS = (AttributeError, TypeError, ValueError, KeyError)
    NUMERIC_EXCEPTIONS = (TypeError, ValueError, OverflowError)
    OPTIONAL_RUNTIME_EXCEPTIONS = (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    )

    _SnapshotRestoreFailed = services.SnapshotRestoreFailed
    _adjust_edit_params = services.adjust_edit_params
    _assemble_run_report = services.assemble_run_report
    _build_snapshot_execution_plan = services.build_snapshot_execution_plan
    _build_provider_dataset_plan = services.build_provider_dataset_plan
    _execute_guarded_run = services.execute_guarded_run
    _load_baseline_pairing_evidence = services.load_baseline_pairing_evidence
    _materialize_run_dataset = services.materialize_run_dataset
    _free_model_memory = services.free_model_memory
    _init_retry_controller = services.init_retry_controller
    _load_model_with_cfg = services.load_model_with_cfg
    _persist_run_report_outputs = services.persist_run_report_outputs
    _prepare_config_for_run = services.prepare_config_for_run
    _resolve_device_and_output = services.resolve_device_and_output
    _resolve_snapshot_config = services.resolve_snapshot_config
    _resolve_snapshot_retry_transition = services.resolve_snapshot_retry_transition
    _run_bare_control = services.run_bare_control
    _safe_int = services.safe_int
    _to_serialisable_dict = services.to_serialisable_dict
    _validate_retry_evaluation_report = services.validate_retry_evaluation_report
    _validate_and_harvest_baseline_schedule = (
        services.validate_and_harvest_baseline_schedule
    )
    _materialize_baseline_pairing_schedule = (
        services.materialize_baseline_pairing_schedule
    )
    _resolve_tokenizer = services.resolve_tokenizer
    detect_model_profile = services.detect_model_profile
    get_psutil = services.get_psutil
    get_torch = services.get_torch

    """
    Run InvarLock pipeline with the given configuration.

    The command assembles non-overlapping preview/final windows, executes the
    GuardChain (invariants → spectral → RMT → variance), checks pairing/overlap
    invariants, enforces the configured guard-overhead budget (default ≤1 %),
    and emits a run report plus JSONL
    events suitable for evaluation report generation.
    """

    profile_normalized = (str(profile or "")).strip().lower()
    until_pass = bool(until_pass)
    max_attempts = int(max_attempts)
    no_cleanup = bool(no_cleanup)
    telemetry = bool(telemetry)
    timings: dict[str, float] = {}
    timing_summary: TimingSummaryPayload | None = None
    collect_timings = bool(capture_timings or telemetry)
    total_start: float | None = perf_counter() if collect_timings else None

    # Use shared CLI coercers from invarlock.cli.utils
    report_path_out: str | None = None
    snapshot_tmpdir: str | None = None
    outcome_result: RunExecutionResult | None = None
    outcome_failure: RunExecutionFailure | None = None
    emitted_events: list[RunExecutionEvent] = []

    def _emit(event: RunExecutionEvent) -> None:
        emitted_events.append(event)
        if observer is not None:
            observer(event)

    def _emit_diagnostic(
        *,
        origin: str | None = None,
        code: str | None = None,
        summary: str | None = None,
        level: str | None = None,
        **context: Any,
    ) -> None:
        _emit(
            RunDiagnosticEvent(
                source=origin,
                code=code,
                summary=summary,
                level=level,
                context=dict(context),
            )
        )

    def _emit_guard_overhead_summary(
        guard_overhead_info: dict[str, Any],
        *,
        default_threshold: float,
    ) -> None:
        _emit(
            RunGuardOverheadSummaryEvent(
                guard_overhead_info=guard_overhead_info,
                default_threshold=default_threshold,
            )
        )

    def _emit_retry_summary(retry_controller: Any | None) -> None:
        if not retry_controller or not getattr(
            retry_controller, "attempt_history", None
        ):
            return
        try:
            summary = retry_controller.get_attempt_summary()
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        if not isinstance(summary, dict):
            return
        _emit(RunRetrySummaryEvent(summary=summary))

    def _halt(
        code: str,
        *,
        summary: str | None = None,
        error: Exception | None = None,
        **context: Any,
    ) -> None:
        failure = RunExecutionFailure(
            code=code,
            summary=summary,
            error=error,
            context=dict(context),
        )
        _emit(RunFailureEvent(failure=failure))
        raise _RunExecutionHalt(failure)

    @contextmanager
    def _record_timed_step(key: str):
        start = perf_counter()
        yield
        elapsed = max(0.0, float(perf_counter() - start))
        if collect_timings:
            timings[key] = elapsed

    def _fail_run(message: str, *, error: Exception | None = None) -> None:
        _halt("pipeline_failed", summary=message, error=error)

    def _emit_transition_diagnostic(source: str, diagnostic: Any) -> None:
        code = getattr(diagnostic, "code", None)
        if isinstance(code, str) and code:
            details = getattr(diagnostic, "details", None)
            context = getattr(diagnostic, "context", None)
            payload = {}
            if isinstance(details, dict):
                payload.update(details)
            if isinstance(context, dict):
                payload.update(context)
            payload.setdefault("diagnostic_source", source)
            summary = getattr(diagnostic, "summary", None)
            if not isinstance(summary, str) or not summary:
                message = getattr(diagnostic, "message", None)
                summary = message if isinstance(message, str) and message else None
            _emit_diagnostic(
                origin=source,
                code=code,
                summary=summary,
                **payload,
            )
            return
        kind = getattr(diagnostic, "kind", None)
        if isinstance(kind, str) and kind:
            payload = {}
            metadata = getattr(diagnostic, "metadata", None)
            if isinstance(metadata, dict):
                payload.update(metadata)
            context = getattr(diagnostic, "context", None)
            if isinstance(context, dict):
                payload.update(context)
            payload.setdefault("diagnostic_source", source)
            level = getattr(diagnostic, "level", None)
            if not isinstance(level, str) or not level:
                severity = getattr(diagnostic, "severity", None)
                level = severity if isinstance(severity, str) and severity else None
            summary = getattr(diagnostic, "summary", None)
            if not isinstance(summary, str) or not summary:
                message = getattr(diagnostic, "message", None)
                summary = message if isinstance(message, str) and message else None
            _emit_diagnostic(
                origin=source,
                code=kind,
                summary=summary,
                level=level,
                **payload,
            )
            return
        payload = {"diagnostic_source": source}
        metadata = getattr(diagnostic, "metadata", None)
        if isinstance(metadata, dict):
            payload.update(metadata)
        details = getattr(diagnostic, "details", None)
        if isinstance(details, dict):
            payload.update(details)
        context = getattr(diagnostic, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        level = getattr(diagnostic, "level", None)
        if not isinstance(level, str) or not level:
            severity = getattr(diagnostic, "severity", None)
            level = severity if isinstance(severity, str) and severity else None
        summary = getattr(diagnostic, "summary", None)
        if not isinstance(summary, str) or not summary:
            message = getattr(diagnostic, "message", None)
            summary = message if isinstance(message, str) and message else None
        if len(payload) > 1 or (isinstance(summary, str) and summary):
            _emit_diagnostic(
                origin=source,
                code="transition_diagnostic",
                summary=summary,
                level=level,
                **payload,
            )

    def _cfg_section_value(cfg_obj: Any, name: str) -> Any:
        section_fn = getattr(cfg_obj, "section", None)
        if callable(section_fn):
            try:
                section = section_fn(name)
            except CONFIG_VALUE_EXCEPTIONS:
                section = None
            if section is not None:
                return section
        try:
            return getattr(cfg_obj, name)
        except CONFIG_VALUE_EXCEPTIONS:
            return None

    _optional_dep_unset = object()
    _optional_torch_cache = _optional_dep_unset
    _optional_psutil_cache = _optional_dep_unset

    def _optional_torch() -> Any | None:
        nonlocal _optional_torch_cache
        if _optional_torch_cache is _optional_dep_unset:
            loaded = get_torch()
            _optional_torch_cache = loaded if loaded else None
        return _optional_torch_cache

    def _optional_psutil() -> Any | None:
        nonlocal _optional_psutil_cache
        if _optional_psutil_cache is _optional_dep_unset:
            loaded = get_psutil()
            _optional_psutil_cache = loaded if loaded else None
        return _optional_psutil_cache

    def _require_torch() -> Any:
        loaded = _optional_torch()
        if loaded is not None:
            return loaded
        _halt("torch_missing")

    # use module-level _derive_mlm_seed

    try:
        _require_torch()

        # Import InvarLock components
        from invarlock.core.api import RunConfig
        from invarlock.core.registry import get_registry
        from invarlock.core.runner import CoreRunner

        _emit(RunConfigLoadingEvent(config_path=config))
        cfg = _prepare_config_for_run(
            config_path=config,
            profile=profile,
            edit=edit,
            tier=tier,
            probes=probes,
        )
        _emit(RunConfigLoadedEvent())

        # cfg prepared by helper above
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
        except CONFIG_VALUE_EXCEPTIONS:
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

        adapter_name = str(getattr(cfg.model, "adapter", "")).lower()
        model_id_raw = str(getattr(cfg.model, "id", ""))
        model_profile = detect_model_profile(
            model_id=model_id_raw, adapter=adapter_name
        )
        tokenizer_hash: str | None = None
        tokenizer: Any | None = None

        eval_section = _cfg_section_value(cfg, "eval") or {}
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
        mask_prob = _coerce_float(
            loss_cfg.get("mask_prob") if isinstance(loss_cfg, dict) else None,
            0.15,
        )
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

        # Set deterministic seeds for Python/NumPy/Torch and record provenance
        raw_seed_value = 42
        if hasattr(cfg, "dataset"):
            try:
                raw_seed_value = getattr(cfg.dataset, "seed", 42)
            except CONFIG_VALUE_EXCEPTIONS:
                raw_seed_value = 42
        try:
            seed_value = int(raw_seed_value)
        except NUMERIC_EXCEPTIONS:
            seed_value = 42
        set_seed(seed_value)
        # Enforce deterministic algorithms in CI/Release profiles when torch is available
        profile_label = profile_normalized or None
        torch_mod = _optional_torch()
        if torch_mod is not None and profile_label in {"ci", "release"}:
            try:  # pragma: no cover - behavior depends on torch availability
                resolved_determinism_mode = determinism_mode or "throughput"
                warn_only = False
                if resolved_determinism_mode.lower() != "strict":
                    warn_only = True
                if bool(determinism_warn_only):
                    warn_only = True
                if hasattr(torch_mod, "use_deterministic_algorithms"):
                    torch_mod.use_deterministic_algorithms(True, warn_only=warn_only)
                if hasattr(torch_mod.backends, "cudnn"):
                    torch_mod.backends.cudnn.benchmark = False
                    try:
                        torch_mod.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    except (AttributeError, TypeError, RuntimeError):
                        pass
            except OPTIONAL_RUNTIME_EXCEPTIONS:
                # If we cannot enforce determinism here, we will rely on core checks
                pass
        try:
            numpy_seed = int(np.random.get_state()[1][0])
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            numpy_seed = seed_value
        torch_seed = None
        if torch_mod is not None:
            try:
                torch_seed = int(torch_mod.initial_seed())
            except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
                torch_seed = seed_value
        seed_bundle = {
            "python": int(seed_value),
            "numpy": int(numpy_seed),
            "torch": int(torch_seed) if torch_seed is not None else None,
        }
        _emit(
            RunDeterministicSeedsEvent(
                python_seed=seed_bundle["python"],
                numpy_seed=seed_bundle["numpy"],
                torch_seed=seed_bundle["torch"],
            )
        )

        # Resolve device and output directory
        resolved_device, output_dir = _resolve_device_and_output(
            cfg, device=device, out=out
        )
        _emit(
            RunDeviceResolvedEvent(
                requested_device=str(device or "auto"),
                resolved_device=str(resolved_device),
            )
        )

        determinism_meta: dict[str, Any] | None = None
        try:
            from invarlock.core.determinism_policy import apply_determinism_preset

            preset = apply_determinism_preset(
                profile=profile_label,
                device=resolved_device,
                seed=int(seed_bundle.get("python") or seed_value),
                threads=int(os.environ.get("INVARLOCK_OMP_THREADS", 1) or 1),
            )
            if isinstance(preset, dict) and preset:
                determinism_meta = preset
                preset_seeds = preset.get("seeds")
                if isinstance(preset_seeds, dict) and preset_seeds:
                    for key in ("python", "numpy", "torch"):
                        if key in preset_seeds:
                            seed_bundle[key] = preset_seeds.get(key)
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ):
            determinism_meta = None

        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{output_dir.name}-{timestamp}" if output_dir.name else timestamp

        _emit(RunOutputDirectoryReadyEvent(run_dir=str(run_dir), run_id=run_id))

        # Initialize retry controller if --until-pass mode enabled
        retry_controller = _init_retry_controller(
            until_pass=until_pass,
            max_attempts=max_attempts,
            timeout=timeout,
            baseline=baseline,
        )
        (
            measure_guard_overhead,
            skip_overhead,
            skip_overhead_source,
        ) = _should_measure_overhead_impl(profile_normalized, cfg)
        direct_reuse_loaded_model = (
            skip_overhead
            and profile_normalized in {"ci", "release"}
            and retry_controller is None
        )
        emitted_skip_overhead_warning = False

        baseline_report_data: dict[str, Any] | None = None
        pairing_schedule: dict[str, Any] | None = None
        if baseline:
            baseline_path = Path(baseline)
            strict_baseline = profile_normalized in {"ci", "release"}
            baseline_evidence = _load_baseline_pairing_evidence(
                baseline_path=baseline_path,
                tokenizer_hash=tokenizer_hash,
            )
            baseline_report_data = baseline_evidence.report_data
            pairing_schedule = baseline_evidence.pairing_schedule
            tokenizer_hash = baseline_evidence.tokenizer_hash
            if baseline_evidence.status == "loaded":
                _emit(RunBaselineScheduleLoadedEvent())
            elif baseline_evidence.message:
                if strict_baseline:
                    raise InvarlockError(code="E001", message=baseline_evidence.message)
                _emit_diagnostic(
                    code="baseline_schedule_fallback",
                    summary=baseline_evidence.message,
                    level="warning",
                )

        requested_preview = _safe_int(getattr(cfg.dataset, "preview_n", 0), 0)
        requested_final = _safe_int(getattr(cfg.dataset, "final_n", 0), 0)
        effective_preview = requested_preview
        effective_final = requested_final
        preview_count = effective_preview
        final_count = effective_final
        # Default split prior to provider resolution; updated if provider exposes splits
        try:
            resolved_split = getattr(cfg.dataset, "split", None) or "validation"
        except (AttributeError, TypeError):
            resolved_split = "validation"
        used_fallback_split: bool = False

        # Execute the pipeline using CoreRunner
        _emit(RunPipelineStartedEvent())

        # Get registry and create components
        registry = get_registry()
        adapter = registry.get_adapter(cfg.model.adapter)
        edit_name = getattr(getattr(cfg, "edit", None), "name", None)
        if not isinstance(edit_name, str) or not edit_name.strip():
            _halt("edit_name_missing")
        try:
            edit_op = registry.get_edit(edit_name.strip())
        except (AttributeError, KeyError) as exc:
            _halt("unknown_edit", error=exc, edit_name=edit_name.strip())

        adapter_meta = registry.get_plugin_metadata(cfg.model.adapter, "adapters")
        try:
            from invarlock.core.adapter_provenance import (
                extract_adapter_provenance,
            )

            prov = extract_adapter_provenance(cfg.model.adapter)
            # Attach a small, stable provenance dict under adapter plugin metadata
            adapter_meta["provenance"] = prov.to_dict()
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            # Best-effort only; absence should not break runs
            pass
        try:
            edit_meta = registry.get_plugin_metadata(edit_name.strip(), "edits")
        except KeyError:
            edit_meta = {
                "name": edit_name.strip(),
                "module": "edits.unknown",
                "version": "unknown",
            }

        guards = []
        guard_metadata: list[dict[str, Any]] = []
        guards_section = _cfg_section_value(cfg, "guards") or {}
        guard_order = (
            guards_section.get("order", [])
            if isinstance(guards_section, dict)
            else getattr(guards_section, "order", [])
        )
        for guard_name in guard_order:
            if guard_name != "noop":
                try:
                    guard = registry.get_guard(guard_name)
                    guards.append(guard)
                    guard_metadata.append(
                        registry.get_plugin_metadata(guard_name, "guards")
                    )
                except KeyError:
                    _emit_diagnostic(code="guard_missing", guard_name=guard_name)
        plugin_provenance = {
            "adapter": adapter_meta,
            "edit": edit_meta,
            "guards": guard_metadata,
        }
        pm_acceptance_range = _resolve_pm_acceptance_range_impl(cfg)
        pm_drift_band = _resolve_pm_drift_band_impl(cfg)
        guard_overhead_threshold = _resolve_guard_overhead_threshold_impl(cfg)

        _emit(RunAdapterSelectedEvent(adapter_name=str(adapter.name)))

        run_context = _build_run_context_payload_impl(
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
            to_serialisable_dict_fn=_to_serialisable_dict,
        )
        eval_context = run_context.get("eval")
        if isinstance(eval_context, dict) and isinstance(eval_device_override, str):
            eval_context["device_override"] = eval_device_override
        run_config = RunConfig(
            device=resolved_device,
            max_pm_ratio=_coerce_float(
                eval_section.get("max_pm_ratio")
                if isinstance(eval_section, dict)
                else getattr(eval_section, "max_pm_ratio", None),
                1.5,
            ),
            event_path=run_dir / "events.jsonl",
            context=run_context,
        )
        skip_model_load = False

        # Load model using adapter
        # Load calibration data if dataset is configured
        calibration_data: list[dict[str, Any]] = []
        dataset_meta: dict[str, Any] = {}
        window_plan: dict[str, Any] | None = None
        preview_records: list[dict[str, Any]] = []
        final_records: list[dict[str, Any]] = []
        preview_mask_counts: list[int] = []
        final_mask_counts: list[int] = []
        dataset_timing_start: float | None = perf_counter() if collect_timings else None
        if pairing_schedule or cfg.dataset.provider:
            if not pairing_schedule:
                _emit(RunDatasetLoadingEvent(provider=str(cfg.dataset.provider)))
            try:
                dataset_result = _materialize_run_dataset(
                    pairing_schedule=pairing_schedule,
                    cfg=cfg,
                    model_profile=model_profile,
                    resolved_device=resolved_device,
                    profile=profile,
                    profile_normalized=profile_normalized,
                    requested_preview=requested_preview,
                    requested_final=requested_final,
                    effective_preview=effective_preview,
                    effective_final=effective_final,
                    use_mlm=use_mlm,
                    mask_prob=mask_prob,
                    mask_seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    resolved_loss_type=resolved_loss_type,
                    tier=tier,
                    baseline_report_data=baseline_report_data,
                    tokenizer=tokenizer,
                    tokenizer_hash=tokenizer_hash,
                    resolved_split=resolved_split,
                    validate_and_harvest_baseline_schedule_fn=(
                        _validate_and_harvest_baseline_schedule
                    ),
                    materialize_baseline_pairing_schedule_fn=(
                        _materialize_baseline_pairing_schedule
                    ),
                    resolve_tokenizer_fn=_resolve_tokenizer,
                    build_provider_dataset_plan_fn=_build_provider_dataset_plan,
                )
            except ValueError as exc:
                _fail_run(str(exc), error=exc)
            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                RuntimeError,
                TypeError,
            ) as exc:
                _fail_run(str(exc), error=exc)

            for diagnostic in dataset_result.diagnostics:
                _emit_transition_diagnostic("dataset", diagnostic)

            resolved_split = dataset_result.resolved_split
            used_fallback_split = dataset_result.used_fallback_split
            tokenizer = dataset_result.tokenizer
            tokenizer_hash = dataset_result.tokenizer_hash
            calibration_data = dataset_result.calibration_data
            dataset_meta = dataset_result.dataset_meta
            window_plan = dataset_result.window_plan
            preview_count = dataset_result.preview_count
            final_count = dataset_result.final_count
            effective_preview = dataset_result.effective_preview
            effective_final = dataset_result.effective_final
            preview_mask_counts = dataset_result.preview_mask_counts
            final_mask_counts = dataset_result.final_mask_counts
            preview_records = dataset_result.preview_records
            final_records = dataset_result.final_records

        try:
            run_context["dataset"]["preview_n"] = preview_count
            run_context["dataset"]["final_n"] = final_count
        except (KeyError, TypeError):
            pass
        run_context["dataset_meta"] = dataset_meta
        if window_plan:
            run_context["window_plan"] = window_plan
        if dataset_timing_start is not None:
            timings["load_dataset"] = max(
                0.0, float(perf_counter() - dataset_timing_start)
            )

        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            _emit(
                RunCalibrationBatchSizesDebugEvent(
                    preview_count=int(preview_count),
                    final_count=int(final_count),
                    total_count=len(calibration_data),
                )
            )
            if use_mlm and calibration_data:
                masked_preview = sum(
                    entry.get("mlm_masked", 0)
                    for entry in calibration_data[:preview_count]
                )
                masked_final = sum(
                    entry.get("mlm_masked", 0)
                    for entry in calibration_data[preview_count:]
                )
                _emit(
                    RunMaskedTokensDebugEvent(
                        preview_masked=int(masked_preview),
                        final_masked=int(masked_final),
                    )
                )
                _emit(
                    RunPreviewLabelsDebugEvent(
                        labels=tuple(calibration_data[0]["labels"][:10])
                    )
                )

        # Execute the real pipeline using CoreRunner
        _emit(RunExecutePipelineEvent(guard_count=len(guards)))
        runner = CoreRunner()

        execution_payloads = _build_run_execution_config_payloads_impl(
            cfg=cfg,
            model_profile=model_profile,
        )
        auto_config = execution_payloads.auto_config
        edit_config = execution_payloads.edit_config

        _emit(RunEditSelectedEvent(edit_name=str(edit_op.name)))
        _emit(
            RunGuardChainResolvedEvent(
                guard_names=tuple(
                    str(getattr(guard, "name", "unknown")) for guard in guards
                )
            )
        )

        # Model load/snapshot strategy
        model = None
        restore_fn = None
        snapshot_tmpdir: str | None = None
        snapshot_provenance: dict[str, bool] = {
            "restore_failed": False,
            "reload_path_used": False,
        }

        # Try single-load with snapshot/restore if adapter supports it; fallback to reload per attempt
        try:
            # Load once
            _emit(RunLoadModelOnceEvent(model_id=str(cfg.model.id)))
            with _record_timed_step("load_model"):
                model = _load_model_with_cfg(
                    adapter,
                    cfg,
                    resolved_device,
                    profile=profile_normalized,
                    event_path=run_dir / "events.jsonl",
                    warning_context={"phase": "load_model", "run_id": run_id},
                    prefer_local_files_only=prefer_local_files_only,
                )

            if direct_reuse_loaded_model:
                snapshot_plan = _build_snapshot_execution_plan(
                    adapter=adapter,
                    model=model,
                    cfg_snapshot=None,
                    direct_reuse_loaded_model=True,
                    skip_overhead_source=skip_overhead_source,
                )
            else:
                try:
                    cfg_snapshot = _resolve_snapshot_config(
                        _cfg_section_value(cfg, "context") or {}
                    )
                except OPTIONAL_RUNTIME_EXCEPTIONS:
                    cfg_snapshot = {}
                snapshot_plan = _build_snapshot_execution_plan(
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
                _emit(
                    RunSnapshotModeEvent(enabled=bool(snapshot_plan.snapshot_enabled))
                )
            for diagnostic in snapshot_plan.diagnostics:
                _emit_transition_diagnostic("snapshot_plan", diagnostic)
        except OPTIONAL_RUNTIME_EXCEPTIONS:
            # On any failure, fall back to reload-per-attempt path
            _free_model_memory(model)
            model = None
            restore_fn = None

        # RETRY LOOP - All report processing inside loop
        attempt = 1
        snapshot_retry_transition = _resolve_snapshot_retry_transition(
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
            _emit_transition_diagnostic("snapshot_retry", diagnostic)

        while True:
            # Reset RNG streams each attempt to guarantee determinism across retries
            set_seed(seed_bundle["python"])

            if retry_controller:
                _emit(
                    RunAttemptStartedEvent(
                        attempt=int(attempt),
                        max_attempts=int(max_attempts),
                    )
                )
                if attempt > 1:
                    _emit(
                        RunRetryAttemptStartedEvent(
                            attempt=int(attempt),
                            max_attempts=int(max_attempts),
                        )
                    )
            else:
                if attempt > 1:
                    _emit(RunAttemptStartedEvent(attempt=int(attempt)))

            # Adjust parameters for retry attempts
            if retry_controller and attempt > 1:
                adjustment = _adjust_edit_params(
                    edit_op.name, edit_config, attempt, None
                )
                edit_config = adjustment.params
                for diagnostic in adjustment.diagnostics:
                    _emit_transition_diagnostic("retry_adjustment", diagnostic)

            guard_overhead_payload: dict[str, Any] | None = None
            try:
                if skip_overhead and profile_normalized in {"ci", "release"}:
                    skip_reason = (
                        "context.run.skip_overhead_check"
                        if skip_overhead_source
                        == "config:context.run.skip_overhead_check"
                        else "context.eval.skip_overhead_check"
                    )
                    guard_overhead_payload = {
                        "overhead_threshold": guard_overhead_threshold,
                        "evaluated": False,
                        "passed": True,
                        "skipped": True,
                        "skip_reason": skip_reason,
                        "mode": "skipped",
                        "source": skip_overhead_source
                        or "config:context.run.skip_overhead_check",
                        "diagnostics": [
                            {
                                "kind": "guard_overhead_info",
                                "severity": "info",
                                "message": "Overhead check skipped via config policy",
                                "details": {},
                            }
                        ],
                        "checks": {},
                    }
                elif measure_guard_overhead:
                    guard_overhead_payload = _run_bare_control(
                        adapter=adapter,
                        edit_op=edit_op,
                        cfg=cfg,
                        model=model,
                        run_config=run_config,
                        calibration_data=calibration_data,
                        auto_config=auto_config,
                        edit_config=edit_config,
                        preview_count=preview_count,
                        final_count=final_count,
                        seed_bundle=seed_bundle,
                        resolved_device=resolved_device,
                        restore_fn=restore_fn,
                        resolved_loss_type=resolved_loss_type,
                        overhead_threshold=guard_overhead_threshold,
                        profile_normalized=profile_normalized,
                        snapshot_provenance=snapshot_provenance,
                        skip_model_load=skip_model_load,
                        prefer_local_files_only=prefer_local_files_only,
                    )

                # Ensure clean state for guarded run
                with _record_timed_step("execute"):
                    core_report, model = _execute_guarded_run(
                        runner=runner,
                        adapter=adapter,
                        model=model,
                        cfg=cfg,
                        edit_op=edit_op,
                        run_config=run_config,
                        guards=guards,
                        calibration_data=calibration_data,
                        auto_config=auto_config,
                        edit_config=edit_config,
                        preview_count=preview_count,
                        final_count=final_count,
                        restore_fn=restore_fn,
                        resolved_device=resolved_device,
                        profile_normalized=profile_normalized,
                        snapshot_provenance=snapshot_provenance,
                        skip_model_load=skip_model_load,
                        prefer_local_files_only=prefer_local_files_only,
                    )
            except _SnapshotRestoreFailed as exc:
                snapshot_provenance["restore_failed"] = True
                _free_model_memory(model)
                model = None
                restore_fn = None
                _emit_diagnostic(code="snapshot_restore_fallback", error=str(exc))
                retry_transition = _decide_failed_retry_transition_impl(
                    retry_controller,
                    attempt=attempt,
                    attempt_summary=_build_restore_failure_attempt_summary_impl(),
                    edit_config=edit_config,
                    passed=False,
                )
                for diagnostic in retry_transition.diagnostics:
                    _emit_transition_diagnostic("retry_failure", diagnostic)
                if retry_transition.should_retry:
                    attempt = retry_transition.next_attempt
                    continue
                _halt("snapshot_restore_failed", error=exc)

            debug_metric_diffs_enabled = str(
                os.environ.get("DEBUG_METRIC_DIFFS", "")
            ).strip().lower() in {"1", "true", "yes", "on"}

            assembly_result = _assemble_run_report(
                core_report=core_report,
                cfg=cfg,
                run_context=run_context,
                profile_normalized=profile_normalized,
                auto_config=auto_config,
                resolved_device=resolved_device,
                seed_bundle=seed_bundle,
                guard_overhead_threshold=guard_overhead_threshold,
                model_profile=model_profile,
                determinism_meta=determinism_meta,
                pm_acceptance_range=pm_acceptance_range,
                pm_drift_band=pm_drift_band,
                tokenizer_hash=tokenizer_hash,
                resolved_split=resolved_split,
                preview_count=preview_count,
                final_count=final_count,
                snapshot_provenance=snapshot_provenance,
                edit_op=edit_op,
                edit_label=edit_label,
                run_dir=run_dir,
                run_config=run_config,
                resolved_loss_type=resolved_loss_type,
                timings=timings,
                guard_overhead_payload=guard_overhead_payload,
                baseline=baseline,
                preview_records=preview_records,
                final_records=final_records,
                use_mlm=use_mlm,
                preview_mask_counts=preview_mask_counts,
                final_mask_counts=final_mask_counts,
                profile=profile,
                used_fallback_split=used_fallback_split,
                baseline_report_data=baseline_report_data,
                effective_preview=effective_preview,
                effective_final=effective_final,
                metric_kind=metric_kind,
                window_plan=window_plan,
                debug_metric_diffs_enabled=debug_metric_diffs_enabled,
            )
            report = assembly_result.report
            timings = assembly_result.timings
            provenance_result = assembly_result.provenance_result
            metrics_enrichment = assembly_result.metrics_enrichment

            try:
                if provenance_result.missing_evaluation_windows_for_baseline:
                    _halt(
                        "baseline_windows_missing",
                        message=(
                            provenance_result.missing_evaluation_windows_message
                            or "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but evaluation windows were not produced. Check capacity/pairing config."
                        ),
                    )
            except InvarlockError as ce:
                _halt("invarlock_error", summary=str(ce), error=ce)
            except RuntimeError as exc:
                _fail_run(str(exc), error=exc)

            # Optional: export HF-loadable model snapshot when requested
            save_model_cfg = False
            try:
                save_model_cfg = bool(
                    getattr(getattr(cfg, "output", {}), "save_model", False)
                )
            except (AttributeError, TypeError):
                save_model_cfg = False
            if bool(export_model_requested) or save_model_cfg:
                try:
                    # Resolve destination with precedence:
                    # 1) cfg.output.model_dir (absolute or relative to run_dir)
                    # 2) env INVARLOCK_EXPORT_DIR (absolute or relative)
                    # 3) cfg.output.model_subdir (under run_dir)
                    # 4) default: run_dir / "model"
                    export_dir: Path | None = None
                    # (1) explicit model_dir in config
                    try:
                        out_cfg = getattr(cfg, "output", None)
                        model_dir_cfg = None
                        if out_cfg is not None:
                            model_dir_cfg = getattr(
                                out_cfg, "model_dir", None
                            ) or getattr(out_cfg, "model_path", None)
                        if model_dir_cfg:
                            p = Path(str(model_dir_cfg))
                            export_dir = p if p.is_absolute() else (run_dir / p)
                    except OPTIONAL_RUNTIME_EXCEPTIONS:
                        export_dir = None
                    # (2) env override
                    if export_dir is None and isinstance(export_dir_override, str):
                        if export_dir_override.strip():
                            p = Path(export_dir_override.strip())
                            export_dir = p if p.is_absolute() else (run_dir / p)
                    # (3) config subdir
                    if export_dir is None:
                        try:
                            resolved_export_subdir = str(
                                getattr(
                                    getattr(cfg, "output", {}), "model_subdir", "model"
                                )
                            )
                        except OPTIONAL_RUNTIME_EXCEPTIONS:
                            resolved_export_subdir = "model"
                        export_dir = run_dir / resolved_export_subdir

                    # Ensure directory exists
                    ok = False
                    if hasattr(adapter, "save_pretrained") and model is not None:
                        ok = bool(adapter.save_pretrained(model, export_dir))  # type: ignore[attr-defined]
                    if ok:
                        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
                        if callable(save_tokenizer):
                            try:
                                save_tokenizer(str(export_dir))
                            except OPTIONAL_RUNTIME_EXCEPTIONS:
                                _emit_diagnostic(code="export_tokenizer_missing")
                        report["artifacts"]["checkpoint_path"] = str(export_dir)
                    else:
                        _emit_diagnostic(code="export_adapter_directory_missing")
                except OPTIONAL_RUNTIME_EXCEPTIONS:
                    _emit_diagnostic(code="export_failed")

            pairing_violations = metrics_enrichment.pairing_violations
            if pairing_violations:
                violation = pairing_violations[0]
                err = InvarlockError(
                    code=violation.code,
                    message=violation.message,
                    details=violation.details,
                )
                _halt("invarlock_error", summary=str(err), error=err)
            if metrics_enrichment.debug_diffs_line:
                _emit_diagnostic(
                    code="metric_diffs_debug",
                    summary=metrics_enrichment.debug_diffs_line,
                )

            persistence_result = _persist_run_report_outputs(
                report=report,
                run_dir=run_dir,
                run_config=run_config,
                telemetry=telemetry,
            )
            report_path_out = persistence_result.report_path_out or report_path_out
            if persistence_result.telemetry_saved_path:
                _emit(
                    RunTelemetrySavedEvent(
                        path=str(persistence_result.telemetry_saved_path)
                    )
                )
            elif persistence_result.telemetry_error:
                _emit(
                    RunTelemetryFailedEvent(
                        error=str(persistence_result.telemetry_error)
                    )
                )

            # Metrics display
            pm_obj = None
            try:
                pm_obj = report.get("metrics", {}).get("primary_metric")
            except (AttributeError, TypeError, KeyError):
                pm_obj = None
            if isinstance(pm_obj, dict) and pm_obj:
                try:
                    pm_kind = str(pm_obj.get("kind", "primary")).lower()
                    pm_prev = pm_obj.get("preview")
                    pm_fin = pm_obj.get("final")
                    ratio_vs_base = pm_obj.get("ratio_vs_baseline")
                    if isinstance(pm_prev, (int | float)) and isinstance(
                        pm_fin, (int | float)
                    ):
                        _emit(
                            RunPrimaryMetricSummaryEvent(
                                metric_kind=pm_kind,
                                preview=float(pm_prev),
                                final=float(pm_fin),
                                ratio_vs_baseline=(
                                    float(ratio_vs_base)
                                    if isinstance(ratio_vs_base, (int | float))
                                    and math.isfinite(ratio_vs_base)
                                    else None
                                ),
                            )
                        )
                except (TypeError, ValueError):
                    pass
            # Legacy ppl_* console block removed in favor of primary_metric summary

            guard_overhead_info = report.get("guard_overhead")
            if guard_overhead_info:
                _emit_guard_overhead_summary(
                    guard_overhead_info,
                    default_threshold=guard_overhead_threshold,
                )
                threshold_fraction = float(
                    guard_overhead_info.get(
                        "overhead_threshold", guard_overhead_threshold
                    )
                    or guard_overhead_threshold
                )
                if not guard_overhead_info.get("passed", True):
                    # Only fail hard when the overhead check was actually evaluated
                    # (e.g., for causal LMs with available bare/guarded PM). For
                    # masked LM flows where ppl-like PM is undefined, record as not evaluated
                    # and continue without aborting the run.
                    loss_type_ctx = None
                    try:
                        loss_type_ctx = (
                            run_config.context.get("eval", {})
                            .get("loss", {})
                            .get("resolved_type")
                        )
                    except (AttributeError, KeyError, TypeError):
                        loss_type_ctx = None
                    if (
                        measure_guard_overhead
                        and guard_overhead_info.get("evaluated", False)
                        and str(loss_type_ctx).lower() != "mlm"
                    ):
                        _halt(
                            "guard_overhead_budget_exceeded",
                            threshold_fraction=float(threshold_fraction),
                        )

            # Drift gate status is no longer surfaced in console; rely on evaluation report gates

            # Evaluation report validation for --until-pass mode
            if retry_controller and baseline:
                _emit(RunEvaluationReportStartedEvent())
                retry_validation = _validate_retry_evaluation_report(
                    report=report,
                    baseline_report_data=baseline_report_data,
                    baseline_path=Path(baseline) if baseline else None,
                )
                if retry_validation.telemetry_summary:
                    _emit_diagnostic(
                        code="retry_validation_telemetry_summary",
                        summary=retry_validation.telemetry_summary,
                    )

                retry_decision = _resolve_retry_validation_transition_impl(
                    retry_controller,
                    attempt=attempt,
                    validation_result=retry_validation,
                    edit_config=edit_config,
                )
                retry_disposition = str(
                    getattr(retry_decision, "status", "error") or "error"
                )
                retry_gate_codes = tuple(
                    str(item)
                    for item in (getattr(retry_decision, "validation_gates", ()) or ())
                )
                retry_error = getattr(retry_decision, "error", None)
                retry_summary = str(
                    getattr(retry_error, "message", None) or "Retry validation failed"
                )

                if retry_disposition == "passed":
                    _emit(RunEvaluationReportPassedEvent())
                    break

                if retry_disposition in {"retry", "exhausted"}:
                    _emit(RunEvaluationReportFailedEvent(gate_codes=retry_gate_codes))

                    edit_config = retry_decision.updated_edit_config
                    head_adjustment = retry_decision.head_adjustment
                    if head_adjustment is not None:
                        _emit(
                            RunAutoTuneAdjustmentEvent(
                                global_k=int(head_adjustment["global_k"]),
                                keep_low=int(head_adjustment["keep_low"]),
                                keep_high=int(head_adjustment["keep_high"]),
                            )
                        )

                    for diagnostic in retry_decision.diagnostics:
                        _emit_transition_diagnostic("retry_validation", diagnostic)
                    if retry_disposition == "retry":
                        attempt = retry_decision.next_attempt or (attempt + 1)
                        continue
                    _emit(RunRetryExhaustedEvent(attempt=int(attempt)))
                    break

                if retry_disposition == "error":
                    _emit(RunRetryValidationErrorEvent(summary=retry_summary))
                    break

                _emit(RunRetryValidationErrorEvent(summary=retry_summary))
                break
            else:
                if retry_controller:
                    _record_retry_attempt_impl(
                        retry_controller,
                        attempt=attempt,
                        attempt_summary={
                            "passed": True,
                            "failures": [],
                            "validation": {},
                        },
                        edit_config=edit_config,
                    )
                # No retry mode - single run
                break

        _emit_retry_summary(retry_controller)

        if capture_timings:
            total_duration = (
                max(0.0, float(perf_counter() - total_start))
                if total_start is not None
                else None
            )
            summary_payload = _build_timing_summary_payload_impl(
                timings=timings,
                total_duration=total_duration,
                report=report if isinstance(report, dict) else None,
            )
            if summary_payload is not None:
                timings = dict(summary_payload.timings)
                timing_summary = summary_payload

        outcome_result = RunExecutionResult(
            report_path=report_path_out,
            timings=dict(timings),
            timing_summary=timing_summary,
        )

    except FileNotFoundError as e:
        outcome_failure = RunExecutionFailure(
            code="config_file_missing",
            summary=str(e),
            error=e,
            context={"path": str(e)},
        )
        _emit(
            RunFailureEvent(
                failure=RunExecutionFailure(
                    code="config_file_missing",
                    summary=str(e),
                    error=e,
                    context={"path": str(e)},
                )
            )
        )
    except InvarlockError as ce:
        outcome_failure = RunExecutionFailure(
            code="invarlock_error",
            summary=str(ce),
            error=ce,
        )
        _emit(RunFailureEvent(failure=outcome_failure))
    except _RunExecutionHalt as halt:
        outcome_failure = halt.failure
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
        MemoryError,
        ImportError,
        ModuleNotFoundError,
    ) as e:
        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            import traceback

            traceback.print_exc()
        if isinstance(e, ValueError) and "Invalid RunReport" in str(e):
            # Emit a clearer message for schema failures (exit 2)
            outcome_failure = RunExecutionFailure(
                code="schema_invalid_run_report",
                summary=str(e),
                error=e,
            )
            _emit(RunFailureEvent(failure=outcome_failure))
        elif isinstance(e, ModuleNotFoundError | ImportError) and "torch" in str(e):
            outcome_failure = RunExecutionFailure(
                code="torch_missing",
                summary=str(e),
                error=e,
            )
            _emit(RunFailureEvent(failure=outcome_failure))
        else:
            outcome_failure = RunExecutionFailure(
                code="pipeline_failed",
                summary=str(e),
                error=e,
            )
            _emit(RunFailureEvent(failure=outcome_failure))
    finally:
        # Cleanup snapshot directory if used (always print once per run)
        try:
            if snapshot_tmpdir and not no_cleanup:
                try:
                    import shutil as _sh

                    _sh.rmtree(snapshot_tmpdir, ignore_errors=True)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                finally:
                    _emit(RunCleanupStatusEvent(removed=True))
            else:
                _emit(RunCleanupStatusEvent(removed=False))
        except (AttributeError, NameError, TypeError, OSError):
            # Best-effort cleanup printing; never raise from finally
            pass
    return RunExecutionOutcome(
        ok=outcome_failure is None and outcome_result is not None,
        result=outcome_result,
        failure=outcome_failure,
        events=tuple(emitted_events),
    )
