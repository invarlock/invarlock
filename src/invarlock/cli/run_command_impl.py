# mypy: ignore-errors
"""Implementation body for cli.commands.run.run_command."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from typing import Any


def run_command_impl(
    *,
    config: Any,
    device: Any = None,
    profile: Any = None,
    out: Any = None,
    edit: Any = None,
    edit_label: Any = None,
    tier: Any = None,
    metric_kind: Any = None,
    probes: Any = None,
    until_pass: Any = False,
    max_attempts: Any = 3,
    timeout: Any = None,
    baseline: Any = None,
    no_cleanup: Any = False,
    style: Any = None,
    progress: Any = False,
    timing: Any = False,
    telemetry: Any = False,
    no_color: Any = False,
    deps: Mapping[str, Any],
) -> str | None:
    """Run implementation moved out of run.py to keep command surface stable."""

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

    def _dep(name: str) -> Any:
        try:
            return deps[name]
        except KeyError as exc:
            raise RuntimeError(f"run_command_impl missing dependency: {name}") from exc

    InvarlockError = _dep("InvarlockError")
    ConfigError = _dep("ConfigError")
    Path = _dep("Path")
    _SnapshotRestoreFailed = _dep("_SnapshotRestoreFailed")
    _apply_mlm_masks = _dep("_apply_mlm_masks")
    _apply_mask_only_head_autotune = _dep("_apply_mask_only_head_autotune")
    _apply_warning_filters = _dep("_apply_warning_filters")
    _build_artifacts_payload = _dep("_build_artifacts_payload")
    _build_provider_dataset_plan = _dep("_build_provider_dataset_plan")
    _build_run_context_payload = _dep("_build_run_context_payload")
    _build_run_execution_config_payloads = _dep("_build_run_execution_config_payloads")
    _enrich_run_report_metrics = _dep("_enrich_run_report_metrics")
    _build_edit_payload = _dep("_build_edit_payload")
    _build_timing_summary_payload = _dep("_build_timing_summary_payload")
    _build_retry_result_summary = _dep("_build_retry_result_summary")
    _build_flags_payload = _dep("_build_flags_payload")
    _build_guard_entries = _dep("_build_guard_entries")
    _build_metrics_payload = _dep("_build_metrics_payload")
    _build_run_report_context = _dep("_build_run_report_context")
    _build_run_report_data = _dep("_build_run_report_data")
    _build_run_report_meta = _dep("_build_run_report_meta")
    _canonical_dataset_id = _dep("_canonical_dataset_id")
    _choose_snapshot_mode = _dep("_choose_snapshot_mode")
    _coerce_float = _dep("_coerce_float")
    _coerce_int = _dep("_coerce_int")
    _coerce_option = _dep("_coerce_option")
    _event = _dep("_event")
    _execute_guarded_run = _dep("_execute_guarded_run")
    _extract_pairing_schedule = _dep("_extract_pairing_schedule")
    _load_baseline_pairing_evidence = _dep("_load_baseline_pairing_evidence")
    _materialize_baseline_pairing_schedule = _dep(
        "_materialize_baseline_pairing_schedule"
    )
    _format_guard_chain = _dep("_format_guard_chain")
    _format_kv_line = _dep("_format_kv_line")
    _free_model_memory = _dep("_free_model_memory")
    _finalize_run_provenance = _dep("_finalize_run_provenance")
    _hash_sequences = _dep("_hash_sequences")
    _init_retry_controller = _dep("_init_retry_controller")
    _load_model_with_cfg = _dep("_load_model_with_cfg")
    _merge_core_timing_metrics = _dep("_merge_core_timing_metrics")
    _normalize_overhead_result = _dep("_normalize_overhead_result")
    _estimate_model_bytes = _dep("_estimate_model_bytes")
    _persist_ref_masks = _dep("_persist_ref_masks")
    _postprocess_and_summarize = _dep("_postprocess_and_summarize")
    _prepare_guard_overhead_report = _dep("_prepare_guard_overhead_report")
    _prepare_config_for_run = _dep("_prepare_config_for_run")
    _print_guard_overhead_summary = _dep("_print_guard_overhead_summary")
    _print_pipeline_start = _dep("_print_pipeline_start")
    _print_retry_summary = _dep("_print_retry_summary")
    _resolve_device_and_output = _dep("_resolve_device_and_output")
    _resolve_exit_code = _dep("_resolve_exit_code")
    _resolve_guard_overhead_threshold = _dep("_resolve_guard_overhead_threshold")
    _resolve_pm_min_tokens_target = _dep("_resolve_pm_min_tokens_target")
    _resolve_pm_acceptance_range = _dep("_resolve_pm_acceptance_range")
    _resolve_pm_drift_band = _dep("_resolve_pm_drift_band")
    _resolve_snapshot_config = _dep("_resolve_snapshot_config")
    _run_bare_control = _dep("_run_bare_control")
    _safe_int = _dep("_safe_int")
    _should_measure_overhead = _dep("_should_measure_overhead")
    _style_from_console = _dep("_style_from_console")
    _tensor_or_list_to_ints = _dep("_tensor_or_list_to_ints")
    _to_serialisable_dict = _dep("_to_serialisable_dict")
    _tokenizer_digest = _dep("_tokenizer_digest")
    _validate_retry_evaluation_report = _dep("_validate_retry_evaluation_report")
    _build_snapshot_provenance = _dep("_build_snapshot_provenance")
    _validate_and_harvest_baseline_schedule = _dep(
        "_validate_and_harvest_baseline_schedule"
    )
    click = _dep("click")
    console = _dep("console")
    datetime = _dep("datetime")
    detect_model_profile = _dep("detect_model_profile")
    math = _dep("math")
    np = _dep("np")
    os = _dep("os")
    perf_counter = _dep("perf_counter")
    get_psutil = _dep("get_psutil")
    print_timing_summary = _dep("print_timing_summary")
    resolve_output_style = _dep("resolve_output_style")
    resolve_tokenizer = _dep("resolve_tokenizer")
    set_seed = _dep("set_seed")
    shutil = _dep("shutil")
    timed_step = _dep("timed_step")
    get_torch = _dep("get_torch")
    typer = _dep("typer")

    """
    Run InvarLock pipeline with the given configuration.

    The command assembles non-overlapping preview/final windows, executes the
    GuardChain (invariants → spectral → RMT → variance), checks pairing/overlap
    invariants, enforces the configured guard-overhead budget (default ≤1 %),
    and emits a run report plus JSONL
    events suitable for evaluation report generation.
    """

    try:
        from typer.models import OptionInfo as _TyperOptionInfo  # noqa: F401
    except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
        _TyperOptionInfo = ()  # type: ignore[assignment]

    config = _coerce_option(config)
    device = _coerce_option(device)
    profile = _coerce_option(profile)
    profile_normalized = (str(profile or "")).strip().lower()
    out = _coerce_option(out)
    edit = _coerce_option(edit)
    edit_label = _coerce_option(edit_label)
    tier = _coerce_option(tier)
    metric_kind = _coerce_option(metric_kind)
    probes = _coerce_option(probes)
    until_pass = bool(_coerce_option(until_pass, False))
    max_attempts = int(_coerce_option(max_attempts, 3))
    timeout = _coerce_option(timeout)
    baseline = _coerce_option(baseline)
    no_cleanup = bool(_coerce_option(no_cleanup, False))
    style = _coerce_option(style)
    progress = bool(_coerce_option(progress, False))
    timing = bool(_coerce_option(timing, False))
    telemetry = bool(_coerce_option(telemetry, False))
    no_color = bool(_coerce_option(no_color, False))

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
        from invarlock.eval.data import get_provider
        from invarlock.reporting.report_types import create_empty_report

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
            legacy_edit_kind = (
                edit_payload.get("kind") if isinstance(edit_payload, dict) else None
            )
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            legacy_edit_kind = None
        if legacy_edit_kind is not None:
            raise ConfigError(
                code="E007",
                message=(
                    "CONFIG-KEY-REMOVED: edit.kind. Use edit.name with a canonical "
                    "edit plugin name."
                ),
                details={"removed_keys": ["edit.kind"]},
            )

        try:
            legacy_edit_parameters = (
                edit_payload.get("parameters")
                if isinstance(edit_payload, dict)
                else None
            )
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            legacy_edit_parameters = None
        if legacy_edit_parameters is not None:
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
            from invarlock.cli.determinism import apply_determinism_preset

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
            from invarlock.cli.provenance import (
                extract_adapter_provenance,
            )  # local import to avoid CLI import cycles

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
        calibration_data = None
        dataset_meta: dict[str, Any] = {}
        window_plan: dict[str, Any] | None = None
        preview_records: list[dict[str, Any]] = []
        final_records: list[dict[str, Any]] = []
        preview_mask_counts: list[int] = []
        final_mask_counts: list[int] = []
        dataset_timing_start: float | None = perf_counter() if collect_timings else None
        if pairing_schedule:
            harvested = _validate_and_harvest_baseline_schedule(
                cfg,
                pairing_schedule,
                baseline_report_data,
                tokenizer_hash=tokenizer_hash,
                resolved_loss_type=resolved_loss_type,
                profile=profile,
                baseline_path_str=str(baseline) if baseline else None,
                console=console,
            )
            dataset_meta = harvested["dataset_meta"]
            window_plan = harvested["window_plan"]
            calibration_data = harvested["calibration_data"]
            if use_mlm and tokenizer is None:
                try:
                    tokenizer, tokenizer_hash = resolve_tokenizer(model_profile)
                except (
                    ImportError,
                    ModuleNotFoundError,
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    _event(console, "FAIL", str(exc), emoji="❌", profile=profile)
                    raise typer.Exit(1) from exc
            try:
                materialized_baseline = _materialize_baseline_pairing_schedule(
                    pairing_schedule=pairing_schedule,
                    calibration_data=calibration_data,
                    dataset_meta=dataset_meta,
                    window_plan=window_plan,
                    tokenizer=tokenizer,
                    use_mlm=use_mlm,
                    mask_prob=mask_prob,
                    mask_seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    resolved_tier=tier
                    or getattr(getattr(cfg, "auto", None), "tier", None),
                    profile=profile,
                )
            except ValueError as exc:
                _fail_run(str(exc))

            calibration_data = materialized_baseline.calibration_data
            dataset_meta = materialized_baseline.dataset_meta
            window_plan = materialized_baseline.window_plan
            preview_count = materialized_baseline.preview_count
            final_count = materialized_baseline.final_count
            effective_preview = materialized_baseline.effective_preview
            effective_final = materialized_baseline.effective_final
            preview_mask_counts = materialized_baseline.preview_mask_counts
            final_mask_counts = materialized_baseline.final_mask_counts
            if use_mlm and os.environ.get("INVARLOCK_DEBUG_TRACE"):
                console.print(
                    "[debug] MLM pairing masks → preview="
                    f"{materialized_baseline.preview_mask_total}, "
                    f"final={materialized_baseline.final_mask_total}"
                )
        elif cfg.dataset.provider:
            _event(
                console,
                "DATA",
                f"Loading dataset: {cfg.dataset.provider}",
                emoji="📊",
                profile=profile_normalized,
            )
            try:
                dataset_plan = _build_provider_dataset_plan(
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
                    pairing_schedule_present=bool(pairing_schedule),
                    use_mlm=use_mlm,
                    mask_prob=mask_prob,
                    mask_seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    resolved_loss_type=resolved_loss_type,
                    tier=tier,
                    get_provider_fn=get_provider,
                )
            except RuntimeError as err:
                _fail_run(str(err))
            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                _event(console, "FAIL", str(exc), emoji="❌", profile=profile)
                raise typer.Exit(1) from exc

            for notice in dataset_plan.notices:
                _event(
                    console,
                    notice.tag,
                    notice.message,
                    emoji=notice.emoji,
                    profile=profile_normalized,
                )

            resolved_split = dataset_plan.resolved_split
            used_fallback_split = dataset_plan.used_fallback_split
            tokenizer = dataset_plan.tokenizer
            tokenizer_hash = dataset_plan.tokenizer_hash
            calibration_data = dataset_plan.calibration_data
            dataset_meta = dataset_plan.dataset_meta
            window_plan = dataset_plan.window_plan
            preview_count = dataset_plan.preview_count
            final_count = dataset_plan.final_count
            effective_preview = dataset_plan.effective_preview
            effective_final = dataset_plan.effective_final
            preview_mask_counts = dataset_plan.preview_mask_counts
            final_mask_counts = dataset_plan.final_mask_counts
            preview_records = dataset_plan.preview_records
            final_records = dataset_plan.final_records

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
                )

            if direct_reuse_loaded_model:
                skip_model_load = True
                source_note = (
                    f" ({skip_overhead_source})" if skip_overhead_source else ""
                )
                _event(
                    console,
                    "WARN",
                    f"Overhead check skipped via config policy{source_note}",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
                _event(
                    console,
                    "WARN",
                    "Reusing initially loaded model for guarded execution.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
                emitted_skip_overhead_warning = True
            else:
                # No edit-specific bootstrap logic

                # Load snapshot config from config.context.snapshot (highest precedence)
                try:
                    cfg_snapshot = _resolve_snapshot_config(getattr(cfg, "context", {}))
                except NON_FATAL_RUNTIME_EXCEPTIONS:
                    cfg_snapshot = {}

                supports_chunked = hasattr(adapter, "snapshot_chunked") and hasattr(
                    adapter, "restore_chunked"
                )
                supports_bytes = hasattr(adapter, "snapshot") and hasattr(
                    adapter, "restore"
                )
                est_mb = _estimate_model_bytes(model) / (1024.0 * 1024.0)
                try:
                    psutil_mod = _optional_psutil()
                    if psutil_mod is None:
                        raise AttributeError("psutil unavailable")
                    ram = psutil_mod.virtual_memory()
                    avail_mb = float(getattr(ram, "available", 0)) / (1024.0 * 1024.0)
                except (
                    AttributeError,
                    RuntimeError,
                    OSError,
                    TypeError,
                    ValueError,
                ):
                    avail_mb = 0.0
                try:
                    tmpdir = None
                    if isinstance(cfg_snapshot, dict):
                        tmpdir = cfg_snapshot.get("temp_dir") or None
                    if not tmpdir:
                        tmpdir = (
                            os.environ.get("TMPDIR")
                            or os.environ.get("TMP")
                            or tempfile.gettempdir()
                        )
                    du = shutil.disk_usage(tmpdir)
                    free_mb = float(du.free) / (1024.0 * 1024.0)
                except (OSError, TypeError, ValueError):
                    free_mb = 0.0

                mode = _choose_snapshot_mode(
                    snapshot_config=cfg_snapshot,
                    env_mode=os.environ.get("INVARLOCK_SNAPSHOT_MODE", "auto"),
                    supports_bytes=supports_bytes,
                    supports_chunked=supports_chunked,
                    estimated_model_mb=est_mb,
                    available_ram_mb=avail_mb,
                    disk_free_mb=free_mb,
                    env_ram_fraction=os.environ.get(
                        "INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION"
                    ),
                    env_threshold_mb=os.environ.get("INVARLOCK_SNAPSHOT_THRESHOLD_MB"),
                )
                enabled = mode in {"bytes", "chunked"}
                _event(
                    console,
                    "INIT",
                    f"Snapshot mode: {'enabled' if enabled else 'disabled'}",
                    emoji="💾",
                    profile=profile_normalized,
                )
                if mode == "chunked":
                    snapshot_tmpdir = adapter.snapshot_chunked(model)  # type: ignore[attr-defined]

                    def _restore():
                        adapter.restore_chunked(model, snapshot_tmpdir)  # type: ignore[attr-defined]

                    restore_fn = _restore
                elif mode == "bytes":
                    supports_chunked = hasattr(adapter, "snapshot_chunked") and hasattr(
                        adapter, "restore_chunked"
                    )
                    try:
                        base_blob = adapter.snapshot(model)  # type: ignore[attr-defined]
                    except NON_FATAL_RUNTIME_EXCEPTIONS:
                        if not supports_chunked:
                            raise
                        snapshot_tmpdir = adapter.snapshot_chunked(model)  # type: ignore[attr-defined]

                        def _restore_fallback_chunked():
                            adapter.restore_chunked(model, snapshot_tmpdir)  # type: ignore[attr-defined]

                        restore_fn = _restore_fallback_chunked
                    else:

                        def _restore2():
                            adapter.restore(model, base_blob)  # type: ignore[attr-defined]

                        restore_fn = _restore2
                else:
                    # reload path - properly free GPU memory before setting to None
                    _free_model_memory(model)
                    model = None
                    restore_fn = None
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            # On any failure, fall back to reload-per-attempt path
            _free_model_memory(model)
            model = None
            restore_fn = None

        # RETRY LOOP - All report processing inside loop
        attempt = 1
        if skip_overhead and profile_normalized in {"ci", "release"}:
            if not emitted_skip_overhead_warning:
                source_note = (
                    f" ({skip_overhead_source})" if skip_overhead_source else ""
                )
                _event(
                    console,
                    "WARN",
                    f"Overhead check skipped via config policy{source_note}",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
            if (
                retry_controller is None
                and model is not None
                and restore_fn is None
                and not skip_model_load
            ):
                skip_model_load = True
                _event(
                    console,
                    "WARN",
                    "Snapshot restore unavailable; reusing initially loaded model for guarded execution.",
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
                from invarlock.core.retry import adjust_edit_params

                adjustment = adjust_edit_params(
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
                if retry_controller:
                    retry_controller.record_attempt(
                        attempt,
                        {
                            "passed": False,
                            "failures": ["restore_failed"],
                            "validation": {},
                        },
                        edit_config,
                    )
                    should_retry = retry_controller.should_retry(False)
                    drain_notices = getattr(retry_controller, "drain_notices", None)
                    notices = drain_notices() if callable(drain_notices) else ()
                    for notice in notices:
                        _event(
                            console,
                            "WARN",
                            notice,
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                    if should_retry:
                        attempt += 1
                        continue
                raise typer.Exit(1) from exc

            if not hasattr(core_report, "context") or core_report.context is None:
                core_report.context = {}

            # Convert CoreRunner report to evaluation report
            report = create_empty_report()

            # Persist minimal run context for evaluation report provenance.
            try:
                report["context"] = _build_run_report_context(
                    profile_normalized=profile_normalized,
                    auto_config=auto_config,
                    run_context=run_context,
                )
            except (TypeError, ValueError, KeyError):
                pass

            # Code provenance: commit hash and InvarLock version
            commit_value = (
                getattr(cfg.meta, "commit", "") if hasattr(cfg, "meta") else ""
            )
            if not commit_value:
                try:
                    import subprocess

                    git_path = shutil.which("git")
                    if git_path:
                        commit_value = (
                            subprocess.check_output(
                                [git_path, "rev-parse", "HEAD"],
                                stderr=subprocess.DEVNULL,
                            )
                            .decode("utf-8", "ignore")
                            .strip()
                        )
                except (OSError, subprocess.SubprocessError):
                    commit_value = ""
            invarlock_version = None
            try:
                from invarlock import __version__ as _invarlock_version

                invarlock_version = _invarlock_version
            except ImportError:
                invarlock_version = None

            # Collect determinism/env flags
            env_flags: dict[str, object] = {}
            try:
                import os as _os

                torch_mod = _optional_torch()
                if torch_mod is not None:
                    try:
                        det_enabled = getattr(
                            torch_mod, "are_deterministic_algorithms_enabled", None
                        )
                        if callable(det_enabled):
                            env_flags["torch_deterministic_algorithms"] = bool(
                                det_enabled()
                            )
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                    try:
                        tf32_matmul = getattr(
                            getattr(torch_mod.backends, "cuda", object()),
                            "matmul",
                            None,
                        )
                        if tf32_matmul is not None and hasattr(
                            tf32_matmul, "allow_tf32"
                        ):
                            env_flags["cuda_matmul_allow_tf32"] = bool(
                                tf32_matmul.allow_tf32
                            )
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                    try:
                        cudnn_mod = getattr(torch_mod.backends, "cudnn", None)
                        if cudnn_mod is not None:
                            env_flags["cudnn_allow_tf32"] = bool(
                                getattr(cudnn_mod, "allow_tf32", None)
                            )
                            env_flags["cudnn_deterministic"] = bool(
                                getattr(cudnn_mod, "deterministic", None)
                            )
                            env_flags["cudnn_benchmark"] = bool(
                                getattr(cudnn_mod, "benchmark", None)
                            )
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                    try:
                        env_flags["mps_available"] = bool(
                            getattr(torch_mod.backends, "mps", None)
                            and torch_mod.backends.mps.is_available()
                        )
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                # Common environment variables for determinism
                env_flags["CUBLAS_WORKSPACE_CONFIG"] = _os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
                env_flags = {}

            report["meta"].update(
                _build_run_report_meta(
                    model_id=cfg.model.id,
                    adapter=cfg.model.adapter,
                    resolved_device=resolved_device,
                    commit_value=commit_value,
                    seed_bundle=seed_bundle,
                    auto_config=auto_config,
                    guard_overhead_threshold=guard_overhead_threshold,
                    model_profile=model_profile,
                    timestamp=datetime.now().isoformat(),
                    invarlock_version=invarlock_version,
                    env_flags=env_flags,
                    determinism_meta=determinism_meta,
                    pm_acceptance_range=pm_acceptance_range,
                    pm_drift_band=pm_drift_band,
                )
            )

            dataset_provider = getattr(cfg.dataset, "provider", None)
            if dataset_provider is None:
                dataset_provider = getattr(cfg.dataset, "dataset", None)
            dataset_meta_context = core_report.context.get("dataset_meta", {})
            data_payload, tokenizer_hash = _build_run_report_data(
                canonical_dataset_id=_canonical_dataset_id(dataset_provider),
                resolved_split=resolved_split,
                seq_len=cfg.dataset.seq_len,
                stride=getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
                preview_count=_safe_int(preview_count),
                final_count=_safe_int(final_count),
                dataset_meta_context=dataset_meta_context,
                tokenizer_hash=tokenizer_hash,
            )
            report["data"].update(data_payload)

            if tokenizer_hash:
                report["meta"]["tokenizer_hash"] = tokenizer_hash

            # Snapshot/restore provenance (survives retries).
            try:
                prov = report.setdefault("provenance", {})
                prov.update(_build_snapshot_provenance(snapshot_provenance))
            except (TypeError, KeyError):
                pass

            # Transfer edit information
            edit_payload, context_edit = _build_edit_payload(
                core_edit=(
                    core_report.edit
                    if hasattr(core_report, "edit")
                    and isinstance(core_report.edit, dict)
                    else None
                ),
                edit_name=edit_op.name,
                edit_label=edit_label,
            )
            if edit_payload:
                report["edit"].update(edit_payload)
            if context_edit and isinstance(core_report.context, dict):
                core_report.context.setdefault("edit", {})
                core_report.context["edit"].update(context_edit)

            mask_artifact_path = _persist_ref_masks(core_report, run_dir)
            report["artifacts"].update(
                _build_artifacts_payload(
                    event_path=run_config.event_path,
                    mask_artifact_path=mask_artifact_path,
                )
            )

            # Transfer metrics (PM-only: do not write ppl_* fields)
            if hasattr(core_report, "metrics") and core_report.metrics:
                timings = _merge_core_timing_metrics(timings, core_report.metrics)
                metrics_payload = _build_metrics_payload(
                    core_metrics=core_report.metrics,
                    window_plan_context=core_report.context.get("window_plan"),
                    dataset_meta_context=dataset_meta_context,
                    resolved_loss_type=resolved_loss_type,
                )
                report["metrics"].update(metrics_payload)

            if guard_overhead_payload is not None:
                report["guard_overhead"] = _prepare_guard_overhead_report(
                    guard_overhead_payload,
                    resolved_loss_type=resolved_loss_type,
                    core_report=core_report,
                    report=report,
                    default_threshold=guard_overhead_threshold,
                )

            had_baseline = bool(baseline and Path(baseline).exists())
            try:
                provenance_result = _finalize_run_provenance(
                    report=report,
                    core_report=core_report,
                    preview_records=preview_records,
                    final_records=final_records,
                    use_mlm=use_mlm,
                    preview_mask_counts=preview_mask_counts,
                    final_mask_counts=final_mask_counts,
                    had_baseline=had_baseline,
                    profile=profile,
                    resolved_split=resolved_split,
                    used_fallback_split=used_fallback_split,
                    baseline_report_data=baseline_report_data,
                )
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

            report["guards"].extend(
                _build_guard_entries(
                    core_report.guards
                    if hasattr(core_report, "guards")
                    and isinstance(core_report.guards, dict)
                    else None
                )
            )

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

            report["flags"].update(
                _build_flags_payload(
                    core_report.guards
                    if hasattr(core_report, "guards")
                    and isinstance(core_report.guards, dict)
                    else None
                )
            )

            debug_metric_diffs_enabled = str(
                os.environ.get("DEBUG_METRIC_DIFFS", "")
            ).strip().lower() in {"1", "true", "yes", "on"}
            metrics_enrichment = _enrich_run_report_metrics(
                report=report,
                core_report=core_report,
                run_config=run_config,
                cfg=cfg,
                model_profile=model_profile,
                baseline_requested=bool(baseline),
                baseline_report_data=baseline_report_data,
                metric_kind=metric_kind,
                resolved_loss_type=resolved_loss_type,
                effective_preview=effective_preview,
                effective_final=effective_final,
                profile_normalized=profile_normalized,
                window_plan=window_plan,
                debug_metric_diffs_enabled=debug_metric_diffs_enabled,
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

            telemetry_path: Path | None = None
            if telemetry:
                telemetry_path = run_dir / "telemetry.json"
                report.setdefault("artifacts", {})["telemetry_path"] = str(
                    telemetry_path
                )

            saved_files = _postprocess_and_summarize(
                report=report,
                run_dir=run_dir,
                run_config=run_config,
                console=console,
            )
            try:
                if isinstance(saved_files, dict) and saved_files.get("json"):
                    report_path_out = str(saved_files["json"])
            except (TypeError, KeyError):
                pass

            if telemetry and telemetry_path is not None:
                try:
                    from invarlock.reporting.telemetry import save_telemetry_report

                    saved_path = save_telemetry_report(
                        report, run_dir, filename=telemetry_path.name
                    )
                    if isinstance(saved_files, dict):
                        saved_files["telemetry"] = str(saved_path)
                    _event(
                        console,
                        "DATA",
                        f"Telemetry: {saved_path}",
                        emoji="📈",
                        profile=profile_normalized,
                    )
                except Exception as exc:  # pragma: no cover - best-effort
                    _event(
                        console,
                        "WARN",
                        f"Telemetry export failed: {exc}",
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

                retry_controller.record_attempt(
                    attempt, retry_validation.attempt_summary, edit_config
                )

                if retry_validation.status == "passed":
                    _event(
                        console,
                        "PASS",
                        "Evaluation report PASSED all gates!",
                        emoji="✅",
                        profile=profile_normalized,
                    )
                    break

                if retry_validation.status == "failed":
                    _event(
                        console,
                        "FAIL",
                        "Evaluation report FAILED gates: "
                        f"{', '.join(retry_validation.failed_gates)}",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

                    edit_config, head_adjustment = _apply_mask_only_head_autotune(
                        edit_config, retry_validation.validation
                    )
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

                    should_retry = retry_controller.should_retry(
                        retry_validation.passed
                    )
                    drain_notices = getattr(retry_controller, "drain_notices", None)
                    notices = drain_notices() if callable(drain_notices) else ()
                    for notice in notices:
                        _event(
                            console,
                            "WARN",
                            notice,
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                    if should_retry:
                        attempt += 1
                        continue
                    _event(
                        console,
                        "FAIL",
                        f"Exhausted retry budget after {attempt} attempts",
                        emoji="❌",
                        profile=profile_normalized,
                    )
                    break

                _event(
                    console,
                    "WARN",
                    "Evaluation report validation failed: "
                    f"{retry_validation.error_message}",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
                break
            else:
                if retry_controller:
                    retry_controller.record_attempt(
                        attempt,
                        {"passed": True, "failures": [], "validation": {}},
                        edit_config,
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
