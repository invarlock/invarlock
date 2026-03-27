# mypy: ignore-errors
"""Typed run execution owner for config-driven run commands."""

from __future__ import annotations

from typing import Any

from invarlock.core.config_execution import ConfigExecutionRequest


def execute_run_request(
    request: ConfigExecutionRequest,
    *,
    deps: Any,
) -> str | None:
    """Execute a config-driven run using an explicit typed request."""

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
    style = request.style
    progress = request.progress
    timing = request.timing
    telemetry = request.telemetry
    no_color = request.no_color
    prefer_local_files_only = request.prefer_local_files_only

    COERCE_EXCEPTIONS = (AttributeError, TypeError, ValueError, Exception)
    NUMERIC_EXCEPTIONS = (TypeError, ValueError, OverflowError)
    NON_FATAL_RUNTIME_EXCEPTIONS = (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
        Exception,
    )

    InvarlockError = deps.InvarlockError
    ConfigError = deps.ConfigError
    Path = deps.Path
    _SnapshotRestoreFailed = deps._SnapshotRestoreFailed
    _apply_mlm_masks = deps._apply_mlm_masks
    _adjust_edit_params = deps._adjust_edit_params
    _apply_warning_filters = deps._apply_warning_filters
    _assemble_run_report = deps._assemble_run_report
    _build_snapshot_execution_plan = deps._build_snapshot_execution_plan
    _build_provider_dataset_plan = deps._build_provider_dataset_plan
    _build_run_context_payload = deps._build_run_context_payload
    _build_run_execution_config_payloads = deps._build_run_execution_config_payloads
    _build_timing_summary_payload = deps._build_timing_summary_payload
    _build_restore_failure_attempt_summary = deps._build_restore_failure_attempt_summary
    _decide_failed_retry_transition = deps._decide_failed_retry_transition
    _coerce_float = deps._coerce_float
    _coerce_int = deps._coerce_int
    _event = deps._event
    _execute_guarded_run = deps._execute_guarded_run
    _extract_pairing_schedule = deps._extract_pairing_schedule
    _load_baseline_pairing_evidence = deps._load_baseline_pairing_evidence
    _materialize_run_dataset = deps._materialize_run_dataset
    _format_guard_chain = deps._format_guard_chain
    _format_kv_line = deps._format_kv_line
    _free_model_memory = deps._free_model_memory
    _hash_sequences = deps._hash_sequences
    _init_retry_controller = deps._init_retry_controller
    _load_model_with_cfg = deps._load_model_with_cfg
    _normalize_overhead_result = deps._normalize_overhead_result
    _persist_run_report_outputs = deps._persist_run_report_outputs
    _prepare_config_for_run = deps._prepare_config_for_run
    _print_guard_overhead_summary = deps._print_guard_overhead_summary
    _print_pipeline_start = deps._print_pipeline_start
    _print_retry_summary = deps._print_retry_summary
    _record_retry_attempt = deps._record_retry_attempt
    _resolve_device_and_output = deps._resolve_device_and_output
    _resolve_exit_code = deps._resolve_exit_code
    _resolve_guard_overhead_threshold = deps._resolve_guard_overhead_threshold
    _resolve_pm_min_tokens_target = deps._resolve_pm_min_tokens_target
    _resolve_pm_acceptance_range = deps._resolve_pm_acceptance_range
    _resolve_pm_drift_band = deps._resolve_pm_drift_band
    _resolve_retry_validation_transition = deps._resolve_retry_validation_transition
    _resolve_snapshot_config = deps._resolve_snapshot_config
    _resolve_snapshot_retry_transition = deps._resolve_snapshot_retry_transition
    _run_bare_control = deps._run_bare_control
    _safe_int = deps._safe_int
    _should_measure_overhead = deps._should_measure_overhead
    _style_from_console = deps._style_from_console
    _tensor_or_list_to_ints = deps._tensor_or_list_to_ints
    _to_serialisable_dict = deps._to_serialisable_dict
    _tokenizer_digest = deps._tokenizer_digest
    _validate_retry_evaluation_report = deps._validate_retry_evaluation_report
    _validate_and_harvest_baseline_schedule = deps._validate_and_harvest_baseline_schedule
    click = deps.click
    console = deps.console
    datetime = deps.datetime
    detect_model_profile = deps.detect_model_profile
    math = deps.math
    np = deps.np
    os = deps.os
    perf_counter = deps.perf_counter
    get_psutil = deps.get_psutil
    print_timing_summary = deps.print_timing_summary
    resolve_output_style = deps.resolve_output_style
    set_seed = deps.set_seed
    timed_step = deps.timed_step
    get_torch = deps.get_torch
    typer = deps.typer

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
    progress = bool(progress)
    timing = bool(timing)
    telemetry = bool(telemetry)
    no_color = bool(no_color)

    output_style = resolve_output_style(
        style=str(style) if style is not None else None,
        profile=profile_normalized,
        progress=progress,
        timing=timing,
        no_color=no_color,
    )
    console._invarlock_output_style = output_style
    if not output_style.color:
        console.no_color = True
    timings: dict[str, float] = {}
    collect_timings = bool(output_style.timing or telemetry)
    total_start: float | None = perf_counter() if collect_timings else None

    _apply_warning_filters(profile_normalized)

    # Use shared CLI coercers from invarlock.cli.utils
    report_path_out: str | None = None
    snapshot_tmpdir: str | None = None

    def _fail_run(message: str) -> None:
        _event(console, "FAIL", message, emoji="❌", profile=profile_normalized)
        # Generic failure path → exit 1 (InvarlockError paths handle code 3 separately)
        raise typer.Exit(1)

    def _provider_event(tag: str, message: str, emoji: str | None = None) -> None:
        _event(
            console,
            tag,
            message,
            emoji=emoji,
            profile=profile_normalized,
        )

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
        _event(
            console,
            "FAIL",
            "Torch is required for this command. "
            'Install extras with: pip install "invarlock[hf]" '
            'or "invarlock[adapters]".',
            emoji="❌",
            profile=profile_normalized,
        )
        raise typer.Exit(1)

    # use module-level _extract_pairing_schedule

    # use module-level _to_int_list, _tensor_or_list_to_ints, _safe_int

    # Use the module-level _hash_sequences to avoid duplication

    # use module-level _derive_mlm_seed

    # use module-level _apply_mlm_masks

    # use module-level _tokenizer_digest

    _require_torch()

    try:
        # Import InvarLock components
        from invarlock.core.api import RunConfig
        from invarlock.core.registry import get_registry
        from invarlock.core.runner import CoreRunner

        # Load and validate configuration via helper (preserves console prints)
        cfg = _prepare_config_for_run(
            config_path=config,
            profile=profile,
            edit=edit,
            tier=tier,
            probes=probes,
            console=console,
        )

        # cfg prepared by helper above
        edit_payload: dict[str, Any] = {}
        try:
            cfg_dump = cfg.model_dump()
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            cfg_dump = None
        if isinstance(cfg_dump, dict):
            edit_section = cfg_dump.get("edit")
            if isinstance(edit_section, dict):
                edit_payload.update(edit_section)
        try:
            edit_obj = getattr(cfg, "edit", None)
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            edit_obj = None
        edit_dict = getattr(edit_obj, "__dict__", None)
        if isinstance(edit_dict, dict):
            edit_payload.update(edit_dict)

        try:
            removed_edit_kind = (
                edit_payload.get("kind") if isinstance(edit_payload, dict) else None
            )
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            removed_edit_kind = None
        if removed_edit_kind is not None:
            raise ConfigError(
                code="E007",
                message=(
                    "CONFIG-KEY-REMOVED: edit.kind. Use edit.name with a canonical "
                    "edit plugin name."
                ),
                details={"removed_keys": ["edit.kind"]},
            )

        try:
            removed_edit_parameters = (
                edit_payload.get("parameters")
                if isinstance(edit_payload, dict)
                else None
            )
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            removed_edit_parameters = None
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

        loss_cfg = getattr(cfg.eval, "loss", None)
        resolved_loss_type = (
            str(getattr(loss_cfg, "type", "auto")).lower() if loss_cfg else "auto"
        )
        if resolved_loss_type == "auto":
            resolved_loss_type = model_profile.default_loss
        use_mlm = resolved_loss_type == "mlm"
        mask_prob = _coerce_float(getattr(loss_cfg, "mask_prob", None), 0.15)
        mask_seed = _coerce_int(getattr(loss_cfg, "seed", None), 42)
        random_token_prob = _coerce_float(
            getattr(loss_cfg, "random_token_prob", None), 0.1
        )
        original_token_prob = _coerce_float(
            getattr(loss_cfg, "original_token_prob", None), 0.1
        )
        if loss_cfg is not None and getattr(loss_cfg, "type", None) == "auto":
            try:
                loss_cfg.type = resolved_loss_type  # type: ignore[assignment]
            except COERCE_EXCEPTIONS:
                pass

        # Set deterministic seeds for Python/NumPy/Torch and record provenance
        raw_seed_value = 42
        if hasattr(cfg, "dataset"):
            try:
                raw_seed_value = getattr(cfg.dataset, "seed", 42)
            except COERCE_EXCEPTIONS:
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
                determinism_mode = (
                    os.environ.get("PACK_DETERMINISM")
                    or os.environ.get("INVARLOCK_DETERMINISM")
                    or "throughput"
                )
                warn_only = False
                if determinism_mode and determinism_mode.lower() != "strict":
                    warn_only = True
                warn_only_env = os.environ.get("INVARLOCK_DETERMINISM_WARN_ONLY", "")
                if warn_only_env.strip().lower() in {"1", "true", "yes", "y", "on"}:
                    warn_only = True
                if hasattr(torch_mod, "use_deterministic_algorithms"):
                    torch_mod.use_deterministic_algorithms(True, warn_only=warn_only)
                if hasattr(torch_mod.backends, "cudnn"):
                    torch_mod.backends.cudnn.benchmark = False
                    try:
                        torch_mod.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    except (AttributeError, TypeError, RuntimeError):
                        pass
            except NON_FATAL_RUNTIME_EXCEPTIONS:
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
        _event(
            console,
            "INIT",
            "Deterministic seeds → "
            f"python={seed_bundle['python']}, numpy={seed_bundle['numpy']}, "
            f"torch={seed_bundle['torch'] if seed_bundle['torch'] is not None else 'N/A'}",
            emoji="🎲",
            profile=profile_normalized,
        )

        # Resolve device and output directory
        resolved_device, output_dir = _resolve_device_and_output(
            cfg, device=device, out=out, console=console
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

        console.print(_format_kv_line("Output", str(run_dir)))
        console.print(_format_kv_line("Run ID", run_id))

        # Initialize retry controller if --until-pass mode enabled
        retry_controller = _init_retry_controller(
            until_pass=until_pass,
            max_attempts=max_attempts,
            timeout=timeout,
            baseline=baseline,
            console=console,
        )
        (
            measure_guard_overhead,
            skip_overhead,
            skip_overhead_source,
        ) = _should_measure_overhead(profile_normalized, cfg)
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
                _event(
                    console,
                    "DATA",
                    "Loaded baseline evaluation schedule for pairing",
                    emoji="🧬",
                    profile=profile_normalized,
                )
            elif baseline_evidence.message:
                if strict_baseline:
                    raise InvarlockError(code="E001", message=baseline_evidence.message)
                _event(
                    console,
                    "WARN",
                    f"{baseline_evidence.message}. Falling back to dataset schedule.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )

        requested_preview = int(getattr(cfg.dataset, "preview_n", 0))
        requested_final = int(getattr(cfg.dataset, "final_n", 0))
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
        _print_pipeline_start(console)

        # Get registry and create components
        registry = get_registry()
        adapter = registry.get_adapter(cfg.model.adapter)
        edit_name = getattr(getattr(cfg, "edit", None), "name", None)
        if not isinstance(edit_name, str) or not edit_name.strip():
            _event(
                console,
                "FAIL",
                "Edit configuration must specify a non-empty `edit.name`.",
                emoji="❌",
                profile=profile_normalized,
            )
            raise typer.Exit(1)
        try:
            edit_op = registry.get_edit(edit_name.strip())
        except (AttributeError, KeyError) as exc:
            _event(
                console,
                "FAIL",
                f"Unknown edit '{edit_name.strip()}'.",
                emoji="❌",
                profile=profile_normalized,
            )
            raise typer.Exit(2) from exc

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
        for guard_name in cfg.guards.order:
            if guard_name != "noop":
                try:
                    guard = registry.get_guard(guard_name)
                    guards.append(guard)
                    guard_metadata.append(
                        registry.get_plugin_metadata(guard_name, "guards")
                    )
                except KeyError:
                    _event(
                        console,
                        "WARN",
                        f"Guard '{guard_name}' not found, skipping",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
        plugin_provenance = {
            "adapter": adapter_meta,
            "edit": edit_meta,
            "guards": guard_metadata,
        }
        pm_acceptance_range = _resolve_pm_acceptance_range(cfg)
        pm_drift_band = _resolve_pm_drift_band(cfg)
        guard_overhead_threshold = _resolve_guard_overhead_threshold(cfg)

        _event(
            console,
            "DATA",
            f"Adapter: {adapter.name}",
            emoji="🔌",
            profile=profile_normalized,
        )

        tiny_relax_env = str(os.environ.get("INVARLOCK_TINY_RELAX", "")).strip().lower()
        run_context = _build_run_context_payload(
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
            tiny_relax_enabled=tiny_relax_env in {"1", "true", "yes", "on"},
        )
        run_config = RunConfig(
            device=resolved_device,
            max_pm_ratio=getattr(cfg.eval, "max_pm_ratio", 1.5),
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
                _event(
                    console,
                    "DATA",
                    f"Loading dataset: {cfg.dataset.provider}",
                    emoji="📊",
                    profile=profile_normalized,
                )
            try:
                dataset_result = _materialize_run_dataset(
                    pairing_schedule=pairing_schedule,
                    cfg=cfg,
                    model_profile=model_profile,
                    console=console,
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
                )
            except ValueError as exc:
                _fail_run(str(exc))
            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                RuntimeError,
                TypeError,
            ) as exc:
                _event(console, "FAIL", str(exc), emoji="❌", profile=profile)
                raise typer.Exit(1) from exc

            for notice in dataset_result.notices:
                _event(
                    console,
                    notice.tag,
                    notice.message,
                    emoji=notice.emoji,
                    profile=profile_normalized,
                )

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
            console.print(
                "[debug] calibration batch size => preview="
                f"{preview_count} final={final_count} total={len(calibration_data)}"
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
                console.print(
                    f"[debug] masked tokens (preview/final) = {masked_preview}/{masked_final}"
                )
                console.print(
                    f"[debug] sample labels first preview entry (first 10) = {calibration_data[0]['labels'][:10]}"
                )

        # Execute the real pipeline using CoreRunner
        _event(
            console,
            "EXEC",
            f"Executing pipeline with {len(guards)} guards...",
            emoji="⚙️",
            profile=profile_normalized,
        )
        runner = CoreRunner()

        execution_payloads = _build_run_execution_config_payloads(
            cfg=cfg,
            model_profile=model_profile,
        )
        auto_config = execution_payloads.auto_config
        edit_config = execution_payloads.edit_config

        console.print(_format_kv_line("Edit", str(edit_op.name)))
        console.print(_format_kv_line("Guards", _format_guard_chain(guards)))

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
            _event(
                console,
                "INIT",
                f"Loading model once: {cfg.model.id}",
                emoji="🔧",
                profile=profile_normalized,
            )
            with timed_step(
                console=console,
                style=_style_from_console(console, profile=profile_normalized),
                timings=timings,
                key="load_model",
                tag="INIT",
                message="Load model",
                emoji="🔧",
            ):
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
                    cfg_snapshot = _resolve_snapshot_config(getattr(cfg, "context", {}))
                except NON_FATAL_RUNTIME_EXCEPTIONS:
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
                _event(
                    console,
                    "INIT",
                    f"Snapshot mode: {'enabled' if snapshot_plan.snapshot_enabled else 'disabled'}",
                    emoji="💾",
                    profile=profile_normalized,
                )
            for notice in snapshot_plan.warning_notices:
                _event(
                    console,
                    "WARN",
                    notice,
                    emoji="⚠️",
                    profile=profile_normalized,
                )
        except NON_FATAL_RUNTIME_EXCEPTIONS:
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
        for notice in snapshot_retry_transition.warning_notices:
            _event(
                console,
                "WARN",
                notice,
                emoji="⚠️",
                profile=profile_normalized,
            )

        while True:
            # Reset RNG streams each attempt to guarantee determinism across retries
            set_seed(seed_bundle["python"])

            if retry_controller:
                console.print("\n")
                _event(
                    console,
                    "EXEC",
                    f"Attempt {attempt}/{max_attempts}",
                    emoji="🚀",
                    profile=profile_normalized,
                )
                if attempt > 1:
                    _event(
                        console,
                        "EXEC",
                        f"Retry attempt {attempt}/{max_attempts}",
                        emoji="🔄",
                        profile=profile_normalized,
                    )
            else:
                if attempt > 1:
                    console.print("\n")
                    _event(
                        console,
                        "EXEC",
                        f"Attempt {attempt}",
                        emoji="🚀",
                        profile=profile_normalized,
                    )

            # Adjust parameters for retry attempts
            if retry_controller and attempt > 1:
                adjustment = _adjust_edit_params(
                    edit_op.name, edit_config, attempt, None
                )
                edit_config = adjustment.params
                for notice in adjustment.notices:
                    _event(
                        console,
                        "INIT",
                        notice,
                        emoji="🔧",
                        profile=profile_normalized,
                    )

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
                        "messages": ["Overhead check skipped via config policy"],
                        "warnings": [],
                        "errors": [],
                        "checks": {},
                    }
                elif measure_guard_overhead:
                    bare_edit_config = dict(edit_config or {})
                    bare_edit_config["emit"] = False
                    guard_overhead_payload = _run_bare_control(
                        adapter=adapter,
                        edit_op=edit_op,
                        cfg=cfg,
                        model=model,
                        run_config=run_config,
                        calibration_data=calibration_data,
                        auto_config=auto_config,
                        edit_config=bare_edit_config,
                        preview_count=preview_count,
                        final_count=final_count,
                        seed_bundle=seed_bundle,
                        resolved_device=resolved_device,
                        restore_fn=restore_fn,
                        console=console,
                        resolved_loss_type=resolved_loss_type,
                        overhead_threshold=guard_overhead_threshold,
                        profile_normalized=profile_normalized,
                        snapshot_provenance=snapshot_provenance,
                        skip_model_load=skip_model_load,
                        prefer_local_files_only=prefer_local_files_only,
                    )

                # Ensure clean state for guarded run
                with timed_step(
                    console=console,
                    style=_style_from_console(console, profile=profile_normalized),
                    timings=timings,
                    key="execute",
                    tag="EXEC",
                    message="Execute pipeline",
                    emoji="⚙️",
                ):
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
                        console=console,
                        snapshot_provenance=snapshot_provenance,
                        skip_model_load=skip_model_load,
                        prefer_local_files_only=prefer_local_files_only,
                    )
            except _SnapshotRestoreFailed as exc:
                snapshot_provenance["restore_failed"] = True
                _free_model_memory(model)
                model = None
                restore_fn = None
                _event(
                    console,
                    "WARN",
                    "Snapshot restore failed; switching to reload-per-attempt.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
                _event(
                    console,
                    "WARN",
                    f"↳ {exc}",
                    profile=profile_normalized,
                )
                retry_transition = _decide_failed_retry_transition(
                    retry_controller,
                    attempt=attempt,
                    attempt_summary=_build_restore_failure_attempt_summary(),
                    edit_config=edit_config,
                    passed=False,
                )
                for notice in retry_transition.notices:
                    _event(
                        console,
                        "WARN",
                        notice,
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                if retry_transition.should_retry:
                    attempt = retry_transition.next_attempt
                    continue
                raise typer.Exit(1) from exc

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
                    _event(
                        console,
                        "FAIL",
                        provenance_result.missing_evaluation_windows_message
                        or "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but evaluation windows were not produced. Check capacity/pairing config.",
                        emoji="❌",
                        profile=profile_normalized,
                    )
                    raise typer.Exit(3)
            except InvarlockError as ce:
                console.print(str(ce))
                raise typer.Exit(_resolve_exit_code(ce, profile=profile)) from None
            except RuntimeError as _e:
                _fail_run(str(_e))
            except (typer.Exit, SystemExit, click.exceptions.Exit):
                raise

            # Optional: export HF-loadable model snapshot when requested
            export_env = str(
                os.environ.get("INVARLOCK_EXPORT_MODEL", "")
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            save_model_cfg = False
            try:
                save_model_cfg = bool(
                    getattr(getattr(cfg, "output", {}), "save_model", False)
                )
            except (AttributeError, TypeError):
                save_model_cfg = False
            if export_env or save_model_cfg:
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
                    except NON_FATAL_RUNTIME_EXCEPTIONS:
                        export_dir = None
                    # (2) env override
                    if export_dir is None:
                        env_dir_raw = os.environ.get("INVARLOCK_EXPORT_DIR", "")
                        if isinstance(env_dir_raw, str) and env_dir_raw.strip():
                            p = Path(env_dir_raw.strip())
                            export_dir = p if p.is_absolute() else (run_dir / p)
                    # (3) config subdir
                    if export_dir is None:
                        try:
                            resolved_export_subdir = str(
                                getattr(
                                    getattr(cfg, "output", {}), "model_subdir", "model"
                                )
                            )
                        except NON_FATAL_RUNTIME_EXCEPTIONS:
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
                            except NON_FATAL_RUNTIME_EXCEPTIONS:
                                _event(
                                    console,
                                    "WARN",
                                    "Exported model checkpoint without tokenizer artifacts; local tokenizer reload may fail.",
                                    emoji="⚠️",
                                    profile=profile_normalized,
                                )
                        report["artifacts"]["checkpoint_path"] = str(export_dir)
                    else:
                        _event(
                            console,
                            "WARN",
                            "Model export requested but adapter did not save a HF directory.",
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                except NON_FATAL_RUNTIME_EXCEPTIONS:
                    _event(
                        console,
                        "WARN",
                        "Model export requested but failed due to an unexpected error.",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

            pairing_violations = metrics_enrichment.pairing_violations
            if pairing_violations:
                violation = pairing_violations[0]
                err = InvarlockError(
                    code=violation.code,
                    message=violation.message,
                    details=violation.details,
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)
            if metrics_enrichment.debug_diffs_line:
                console.print(
                    "[dim]DEBUG_METRIC_DIFFS: "
                    + metrics_enrichment.debug_diffs_line
                    + "[/dim]"
                )

            persistence_result = _persist_run_report_outputs(
                report=report,
                run_dir=run_dir,
                run_config=run_config,
                console=console,
                telemetry=telemetry,
            )
            report_path_out = persistence_result.report_path_out or report_path_out
            if persistence_result.telemetry_saved_path:
                _event(
                    console,
                    "DATA",
                    f"Telemetry: {persistence_result.telemetry_saved_path}",
                    emoji="📈",
                    profile=profile_normalized,
                )
            elif persistence_result.telemetry_error:
                _event(
                    console,
                    "WARN",
                    f"Telemetry export failed: {persistence_result.telemetry_error}",
                    emoji="⚠️",
                    profile=profile_normalized,
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
                    if isinstance(pm_prev, (int | float)) and isinstance(
                        pm_fin, (int | float)
                    ):
                        _event(
                            console,
                            "METRIC",
                            f"Primary Metric [{pm_kind}] — preview: {pm_prev:.3f}, final: {pm_fin:.3f}",
                            emoji="📌",
                            profile=profile_normalized,
                        )
                    ratio_vs_base = pm_obj.get("ratio_vs_baseline")
                    if isinstance(ratio_vs_base, (int | float)) and math.isfinite(
                        ratio_vs_base
                    ):
                        _event(
                            console,
                            "METRIC",
                            f"Ratio vs baseline [{pm_kind}]: {ratio_vs_base:.3f}",
                            emoji="🔗",
                            profile=profile_normalized,
                        )
                except (TypeError, ValueError):
                    pass
            # Legacy ppl_* console block removed in favor of primary_metric summary

            guard_overhead_info = report.get("guard_overhead")
            if guard_overhead_info:
                threshold_fraction = _print_guard_overhead_summary(
                    console,
                    guard_overhead_info,
                    default_threshold=guard_overhead_threshold,
                )
                if not guard_overhead_info.get("passed", True):
                    _event(
                        console,
                        "FAIL",
                        "Guard overhead gate FAILED: Guards add more than the permitted budget",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
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
                        _fail_run(
                            "Guard overhead gate exceeded the configured budget "
                            f"(>{threshold_fraction * 100:.1f}% increase)"
                        )

            # Drift gate status is no longer surfaced in console; rely on evaluation report gates

            # Evaluation report validation for --until-pass mode
            if retry_controller and baseline:
                _event(
                    console,
                    "EXEC",
                    "Generating evaluation report...",
                    emoji="📜",
                    profile=profile_normalized,
                )
                retry_validation = _validate_retry_evaluation_report(
                    report=report,
                    baseline_report_data=baseline_report_data,
                    baseline_path=Path(baseline) if baseline else None,
                )
                if retry_validation.telemetry_summary:
                    console.print(retry_validation.telemetry_summary, markup=False)

                retry_decision = _resolve_retry_validation_transition(
                    retry_controller,
                    attempt=attempt,
                    validation_result=retry_validation,
                    edit_config=edit_config,
                )

                if retry_decision.action == "passed":
                    _event(
                        console,
                        "PASS",
                        "Evaluation report PASSED all gates!",
                        emoji="✅",
                        profile=profile_normalized,
                    )
                    break

                if retry_decision.action in {"retry", "exhausted"}:
                    _event(
                        console,
                        "FAIL",
                        "Evaluation report FAILED gates: "
                        f"{', '.join(retry_decision.failed_gates)}",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

                    edit_config = retry_decision.updated_edit_config
                    head_adjustment = retry_decision.head_adjustment
                    if head_adjustment is not None:
                        _event(
                            console,
                            "INIT",
                            "Auto-tune adjust: global_k → "
                            f"{head_adjustment['global_k']} "
                            f"(bounds {head_adjustment['keep_low']}-{head_adjustment['keep_high']})",
                            emoji="🔧",
                            profile=profile_normalized,
                        )

                    for notice in retry_decision.notices:
                        _event(
                            console,
                            "WARN",
                            notice,
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                    if retry_decision.action == "retry":
                        attempt = retry_decision.next_attempt or (attempt + 1)
                        continue
                    _event(
                        console,
                        "FAIL",
                        f"Exhausted retry budget after {attempt} attempts",
                        emoji="❌",
                        profile=profile_normalized,
                    )
                    break

                if retry_decision.action == "error":
                    _event(
                        console,
                        "WARN",
                        "Evaluation report validation failed: "
                        f"{retry_decision.error_message}",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                    break

                _event(
                    console,
                    "WARN",
                    "Evaluation report validation failed: "
                    f"{retry_decision.error_message}",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
                break
            else:
                if retry_controller:
                    _record_retry_attempt(
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

        _print_retry_summary(console, retry_controller)

        if output_style.timing:
            total_duration = (
                max(0.0, float(perf_counter() - total_start))
                if total_start is not None
                else None
            )
            summary_payload = _build_timing_summary_payload(
                timings=timings,
                total_duration=total_duration,
                report=report if isinstance(report, dict) else None,
            )
            if summary_payload is not None:
                print_timing_summary(
                    console,
                    summary_payload.timings,
                    style=output_style,
                    order=list(summary_payload.order),
                    extra_lines=list(summary_payload.extra_lines),
                )

        # Normal path falls through; cleanup handled below in finally
        return report_path_out

    except FileNotFoundError as e:
        _event(
            console,
            "FAIL",
            f"Configuration file not found: {e}",
            emoji="❌",
            profile=profile_normalized,
        )
        raise typer.Exit(1) from e
    except InvarlockError as ce:
        # InvarlockError → code 3 only in CI/Release; dev → 1
        console.print(str(ce))
        raise typer.Exit(_resolve_exit_code(ce, profile=profile)) from ce
    except (typer.Exit, SystemExit, click.exceptions.Exit):
        # Preserve explicit exit codes (e.g., parity checks, user-triggered exits)
        raise
    except Exception as e:
        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            import traceback

            traceback.print_exc()
        # Emit a clearer message for schema failures (exit 2)
        if isinstance(e, ValueError) and "Invalid RunReport" in str(e):
            _event(
                console,
                "FAIL",
                "Schema invalid: run report structure failed validation",
                emoji="❌",
                profile=profile_normalized,
            )
            code = 2
        else:
            _event(
                console,
                "FAIL",
                f"Pipeline execution failed: {e}",
                emoji="❌",
                profile=profile_normalized,
            )
            code = _resolve_exit_code(e, profile=profile)
        raise typer.Exit(code) from e
    finally:
        # Cleanup snapshot directory if used (always print once per run)
        try:
            if snapshot_tmpdir and not no_cleanup:
                try:
                    import shutil as _sh

                    _sh.rmtree(snapshot_tmpdir, ignore_errors=True)
                except Exception:
                    pass
                finally:
                    _event(
                        console,
                        "INFO",
                        "Cleanup: removed",
                        emoji="🧹",
                        profile=profile_normalized,
                    )
            else:
                _event(
                    console,
                    "INFO",
                    "Cleanup: skipped",
                    emoji="🧹",
                    profile=profile_normalized,
                )
        except (AttributeError, NameError, TypeError, OSError):
            # Best-effort cleanup printing; never raise from finally
            pass
