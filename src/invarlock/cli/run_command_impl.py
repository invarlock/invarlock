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
    _build_timing_summary_payload = _dep("_build_timing_summary_payload")
    _build_retry_result_summary = _dep("_build_retry_result_summary")
    _build_fallback_evaluation_windows = _dep("_build_fallback_evaluation_windows")
    _canonical_dataset_id = _dep("_canonical_dataset_id")
    _choose_snapshot_mode = _dep("_choose_snapshot_mode")
    _coerce_float = _dep("_coerce_float")
    _coerce_int = _dep("_coerce_int")
    _coerce_option = _dep("_coerce_option")
    _compute_provider_digest = _dep("_compute_provider_digest")
    _enforce_provider_parity = _dep("_enforce_provider_parity")
    _event = _dep("_event")
    _execute_guarded_run = _dep("_execute_guarded_run")
    _extract_pairing_schedule = _dep("_extract_pairing_schedule")
    _extract_pm_snapshot_for_overhead = _dep("_extract_pm_snapshot_for_overhead")
    _format_debug_metric_diffs = _dep("_format_debug_metric_diffs")
    _format_guard_chain = _dep("_format_guard_chain")
    _format_kv_line = _dep("_format_kv_line")
    _free_model_memory = _dep("_free_model_memory")
    _hash_sequences = _dep("_hash_sequences")
    _init_retry_controller = _dep("_init_retry_controller")
    _load_model_with_cfg = _dep("_load_model_with_cfg")
    _maybe_plan_release_windows = _dep("_maybe_plan_release_windows")
    _merge_primary_metric_health = _dep("_merge_primary_metric_health")
    _normalize_overhead_result = _dep("_normalize_overhead_result")
    _estimate_model_bytes = _dep("_estimate_model_bytes")
    _finalize_guard_overhead_payload = _dep("_finalize_guard_overhead_payload")
    _persist_ref_masks = _dep("_persist_ref_masks")
    _postprocess_and_summarize = _dep("_postprocess_and_summarize")
    _prepare_config_for_run = _dep("_prepare_config_for_run")
    _print_guard_overhead_summary = _dep("_print_guard_overhead_summary")
    _print_pipeline_start = _dep("_print_pipeline_start")
    _print_retry_summary = _dep("_print_retry_summary")
    _resolve_device_and_output = _dep("_resolve_device_and_output")
    _resolve_effective_windows = _dep("_resolve_effective_windows")
    _resolve_exit_code = _dep("_resolve_exit_code")
    _resolve_guard_overhead_threshold = _dep("_resolve_guard_overhead_threshold")
    _resolve_metric_and_provider = _dep("_resolve_metric_and_provider")
    _resolve_pm_min_tokens_target = _dep("_resolve_pm_min_tokens_target")
    _resolve_pm_acceptance_range = _dep("_resolve_pm_acceptance_range")
    _resolve_pm_drift_band = _dep("_resolve_pm_drift_band")
    _resolve_provider_and_split = _dep("_resolve_provider_and_split")
    _resolve_snapshot_config = _dep("_resolve_snapshot_config")
    _run_bare_control = _dep("_run_bare_control")
    _safe_int = _dep("_safe_int")
    _serialize_evaluation_windows = _dep("_serialize_evaluation_windows")
    _should_measure_overhead = _dep("_should_measure_overhead")
    _style_from_console = _dep("_style_from_console")
    _tensor_or_list_to_ints = _dep("_tensor_or_list_to_ints")
    _to_serialisable_dict = _dep("_to_serialisable_dict")
    _tokenizer_digest = _dep("_tokenizer_digest")
    _validate_and_harvest_baseline_schedule = _dep(
        "_validate_and_harvest_baseline_schedule"
    )
    click = _dep("click")
    console = _dep("console")
    copy = _dep("copy")
    datetime = _dep("datetime")
    detect_model_profile = _dep("detect_model_profile")
    hashlib = _dep("hashlib")
    json = _dep("json")
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
    validate_guard_overhead = _dep("validate_guard_overhead")

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
            if not baseline_path.exists():
                msg = (
                    "PAIRING-EVIDENCE-MISSING: baseline report path does not exist "
                    f"({baseline})"
                )
                if strict_baseline:
                    raise InvarlockError(code="E001", message=msg)
                _event(
                    console,
                    "WARN",
                    f"{msg}. Falling back to dataset schedule.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
            else:
                try:
                    with baseline_path.open(encoding="utf-8") as f:
                        baseline_report_data = json.load(f)
                except (OSError, TypeError, ValueError) as exc:
                    msg = f"PAIRING-EVIDENCE-MISSING: baseline report JSON parse failed ({exc})"
                    if strict_baseline:
                        raise InvarlockError(code="E001", message=msg) from exc
                    _event(
                        console,
                        "WARN",
                        f"{msg}. Falling back to dataset schedule.",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                    baseline_report_data = None
                if isinstance(baseline_report_data, dict):
                    pairing_schedule = _extract_pairing_schedule(baseline_report_data)
                    if pairing_schedule:
                        # Normalize baseline report in-memory so downstream digest/parity
                        # computations see a consistent window_id + mask shape even for
                        # baselines missing some fields.
                        try:
                            ew = baseline_report_data.get("evaluation_windows")
                            if not isinstance(ew, dict):
                                ew = {}
                                baseline_report_data["evaluation_windows"] = ew
                            # Merge the sanitized pairing schedule into existing
                            # evaluation_windows without discarding logloss/token_counts.
                            for arm in ("preview", "final"):
                                src = (
                                    pairing_schedule.get(arm)
                                    if isinstance(pairing_schedule, dict)
                                    else None
                                )
                                if not isinstance(src, dict):
                                    continue
                                dst = ew.get(arm)
                                if not isinstance(dst, dict):
                                    ew[arm] = dict(src)
                                    continue
                                for key, value in src.items():
                                    dst[key] = value
                        except NON_FATAL_RUNTIME_EXCEPTIONS:
                            pass
                        # Harvest tokenizer hash provenance from baseline when present.
                        try:
                            if not tokenizer_hash:
                                tok = None
                                meta = (
                                    baseline_report_data.get("meta")
                                    if isinstance(
                                        baseline_report_data.get("meta"), dict
                                    )
                                    else {}
                                )
                                data = (
                                    baseline_report_data.get("data")
                                    if isinstance(
                                        baseline_report_data.get("data"), dict
                                    )
                                    else {}
                                )
                                if isinstance(meta, dict):
                                    tok = meta.get("tokenizer_hash")
                                if not tok and isinstance(data, dict):
                                    tok = data.get("tokenizer_hash")
                                if isinstance(tok, str) and tok:
                                    tokenizer_hash = tok
                        except NON_FATAL_RUNTIME_EXCEPTIONS:
                            pass
                        _event(
                            console,
                            "DATA",
                            "Loaded baseline evaluation schedule for pairing",
                            emoji="🧬",
                            profile=profile_normalized,
                        )
                    else:
                        msg = (
                            "PAIRING-EVIDENCE-MISSING: baseline report missing or invalid "
                            f"evaluation_windows ({baseline})"
                        )
                        if strict_baseline:
                            raise InvarlockError(code="E001", message=msg)
                        _event(
                            console,
                            "WARN",
                            f"{msg}. Falling back to dataset schedule.",
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                        baseline_report_data = None
                        pairing_schedule = None

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

        # Create run configuration
        guard_overrides = {
            "spectral": _to_serialisable_dict(getattr(cfg.guards, "spectral", {})),
            "rmt": _to_serialisable_dict(getattr(cfg.guards, "rmt", {})),
            "variance": _to_serialisable_dict(getattr(cfg.guards, "variance", {})),
            "invariants": _to_serialisable_dict(getattr(cfg.guards, "invariants", {})),
        }

        if model_profile.invariants:
            invariants_policy = guard_overrides.setdefault("invariants", {})
            existing_checks = invariants_policy.get("profile_checks", [])
            if isinstance(existing_checks, list | tuple | set):
                checks_list = [str(item) for item in existing_checks]
            elif existing_checks:
                checks_list = [str(existing_checks)]
            else:
                checks_list = []
            for invariant in model_profile.invariants:
                invariant_name = str(invariant)
                if invariant_name not in checks_list:
                    checks_list.append(invariant_name)
            invariants_policy["profile_checks"] = checks_list

        run_context = {
            "eval": _to_serialisable_dict(cfg.eval),
            "dataset": _to_serialisable_dict(cfg.dataset),
            "guards": guard_overrides,
            "profile": profile if profile else "",
            "pairing_baseline": pairing_schedule,
            "seeds": seed_bundle,
            "plugins": plugin_provenance,
            "run_id": run_id,
        }
        tiny_relax_env = str(os.environ.get("INVARLOCK_TINY_RELAX", "")).strip().lower()
        if tiny_relax_env in {"1", "true", "yes", "on"}:
            run_context.setdefault("run", {})["tiny_relax"] = True
        # Provide baseline per-window logloss to the CoreRunner for paired tail
        # evidence and (optionally) fail/rollback enforcement.
        try:
            if isinstance(baseline_report_data, dict):
                ew = baseline_report_data.get("evaluation_windows")
                if isinstance(ew, dict):
                    final = ew.get("final")
                    if (
                        isinstance(final, dict)
                        and isinstance(final.get("window_ids"), list)
                        and isinstance(final.get("logloss"), list)
                    ):
                        base_eval: dict[str, Any] = {
                            "final": {
                                "window_ids": list(final.get("window_ids") or []),
                                "logloss": list(final.get("logloss") or []),
                            }
                        }
                        if isinstance(final.get("token_counts"), list):
                            base_eval["final"]["token_counts"] = list(
                                final.get("token_counts") or []
                            )
                        run_context["baseline_eval_windows"] = base_eval
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            pass
        run_context.setdefault("primary_metric", {})["acceptance_range"] = (
            pm_acceptance_range
        )
        run_context["pm_acceptance_range"] = pm_acceptance_range
        if pm_drift_band:
            run_context.setdefault("primary_metric", {})["drift_band"] = pm_drift_band
            run_context["pm_drift_band"] = pm_drift_band
        run_context.setdefault("primary_metric", {})["overhead_threshold"] = (
            guard_overhead_threshold
        )
        run_context["guard_overhead_threshold"] = guard_overhead_threshold
        run_context["model_profile"] = {
            "family": model_profile.family,
            "default_loss": model_profile.default_loss,
            "module_selectors": model_profile.module_selectors,
            "invariants": model_profile.invariants,
            "cert_lints": model_profile.cert_lints,
        }
        extra_context = _to_serialisable_dict(getattr(cfg, "context", {}))
        if isinstance(extra_context, dict):
            run_context.update(extra_context)
        try:
            run_context.setdefault("eval", {}).setdefault("loss", {})[
                "resolved_type"
            ] = resolved_loss_type
        except (AttributeError, TypeError):
            pass
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
        baseline_meta: dict[str, Any] = {}
        window_plan: dict[str, Any] | None = None
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
            preview_window_ids = pairing_schedule["preview"].get("window_ids")
            preview_labels = pairing_schedule["preview"].get("labels")
            for idx, (input_ids, attention_mask) in enumerate(
                zip(
                    pairing_schedule["preview"]["input_ids"],
                    pairing_schedule["preview"]["attention_masks"],
                    strict=False,
                )
            ):
                window_id = (
                    preview_window_ids[idx]
                    if preview_window_ids and idx < len(preview_window_ids)
                    else idx
                )
                entry = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "window_id": f"preview::{window_id}",
                }
                if use_mlm:
                    labels_list: list[int] = []
                    if isinstance(preview_labels, list) and idx < len(preview_labels):
                        labels_list = _tensor_or_list_to_ints(preview_labels[idx])
                    if labels_list and any(token != -100 for token in labels_list):
                        entry["labels"] = labels_list
                        entry["mlm_masked"] = sum(
                            1 for token in labels_list if token != -100
                        )
                    else:
                        entry["labels"] = []
                        entry["mlm_masked"] = 0
                    # Prefer masked_token_counts if present in schedule
                    mtc = pairing_schedule["preview"].get("masked_token_counts")
                    if isinstance(mtc, list) and idx < len(mtc):
                        try:
                            entry["mlm_masked"] = int(mtc[idx])
                        except NUMERIC_EXCEPTIONS:
                            pass
                calibration_data.append(entry)
            final_window_ids = pairing_schedule["final"].get("window_ids")
            final_labels = pairing_schedule["final"].get("labels")
            for idx, (input_ids, attention_mask) in enumerate(
                zip(
                    pairing_schedule["final"]["input_ids"],
                    pairing_schedule["final"]["attention_masks"],
                    strict=False,
                )
            ):
                window_id = (
                    final_window_ids[idx]
                    if final_window_ids and idx < len(final_window_ids)
                    else idx
                )
                entry = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "window_id": f"final::{window_id}",
                }
                if use_mlm:
                    labels_list: list[int] = []
                    if isinstance(final_labels, list) and idx < len(final_labels):
                        labels_list = _tensor_or_list_to_ints(final_labels[idx])
                    if labels_list and any(token != -100 for token in labels_list):
                        entry["labels"] = labels_list
                        entry["mlm_masked"] = sum(
                            1 for token in labels_list if token != -100
                        )
                    else:
                        entry["labels"] = []
                        entry["mlm_masked"] = 0
                    # Prefer masked_token_counts if present in schedule
                    mtc = pairing_schedule["final"].get("masked_token_counts")
                    if isinstance(mtc, list) and idx < len(mtc):
                        try:
                            entry["mlm_masked"] = int(mtc[idx])
                        except NUMERIC_EXCEPTIONS:
                            pass
                calibration_data.append(entry)
            preview_count = len(pairing_schedule["preview"]["input_ids"])
            final_count = len(pairing_schedule["final"]["input_ids"])
            effective_preview = int(preview_count)
            effective_final = int(final_count)
            preview_mask_total = 0
            final_mask_total = 0
            preview_mask_counts: list[int] = []
            final_mask_counts: list[int] = []
            if use_mlm:
                preview_entries = calibration_data[:preview_count]
                final_entries = calibration_data[preview_count:]

                def _needs_masks(entries):
                    missing_any = False
                    counts = []
                    for entry in entries:
                        labels_val = entry.get("labels")
                        has_label_masks = bool(
                            isinstance(labels_val, list)
                            and any(token != -100 for token in labels_val)
                        )
                        existing_count = int(entry.get("mlm_masked", 0))
                        if not has_label_masks and existing_count <= 0:
                            missing_any = True
                        counts.append(int(entry.get("mlm_masked", 0)))
                    return missing_any, counts

                preview_missing, preview_counts_existing = _needs_masks(preview_entries)
                final_missing, final_counts_existing = _needs_masks(final_entries)

                if preview_missing:
                    preview_mask_total, preview_mask_counts = _apply_mlm_masks(
                        preview_entries,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="preview",
                    )
                else:
                    preview_mask_counts = preview_counts_existing
                    preview_mask_total = sum(preview_mask_counts)

                if final_missing:
                    final_mask_total, final_mask_counts = _apply_mlm_masks(
                        final_entries,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="final",
                    )
                else:
                    final_mask_counts = final_counts_existing
                    final_mask_total = sum(final_mask_counts)

                # Ensure counts and labels set on entries
                if preview_mask_counts:
                    for entry, count in zip(
                        preview_entries, preview_mask_counts, strict=False
                    ):
                        entry["mlm_masked"] = int(count)
                if final_mask_counts:
                    for entry, count in zip(
                        final_entries, final_mask_counts, strict=False
                    ):
                        entry["mlm_masked"] = int(count)

                if preview_count > 0 and preview_mask_total <= 0:
                    _fail_run(
                        "Baseline pairing schedule provided no masked tokens for preview windows; "
                        "ensure MLM labels are present in the baseline report."
                    )
                if final_count > 0 and final_mask_total <= 0:
                    _fail_run(
                        "Baseline pairing schedule provided no masked tokens for final windows; "
                        "ensure MLM labels are present in the baseline report."
                    )

                dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
                dataset_meta["masked_tokens_final"] = int(final_mask_total)
                dataset_meta["masked_tokens_total"] = int(
                    preview_mask_total + final_mask_total
                )
                if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                    console.print(
                        f"[debug] MLM pairing masks → preview={preview_mask_total}, final={final_mask_total}"
                    )
            if "preview_total_tokens" not in dataset_meta:
                dataset_meta["preview_total_tokens"] = sum(
                    len(_tensor_or_list_to_ints(seq))
                    for seq in pairing_schedule["preview"]["input_ids"]
                )
            if "final_total_tokens" not in dataset_meta:
                dataset_meta["final_total_tokens"] = sum(
                    len(_tensor_or_list_to_ints(seq))
                    for seq in pairing_schedule["final"]["input_ids"]
                )
            if "preview_hash" not in dataset_meta:
                preview_hash = _hash_sequences(
                    _tensor_or_list_to_ints(seq)
                    for seq in pairing_schedule["preview"]["input_ids"]
                )
                dataset_meta["preview_hash"] = preview_hash
            else:
                preview_hash = dataset_meta["preview_hash"]
            if "final_hash" not in dataset_meta:
                final_hash = _hash_sequences(
                    _tensor_or_list_to_ints(seq)
                    for seq in pairing_schedule["final"]["input_ids"]
                )
                dataset_meta["final_hash"] = final_hash
            else:
                final_hash = dataset_meta["final_hash"]
            if "dataset_hash" not in dataset_meta:
                dataset_meta["dataset_hash"] = hashlib.blake2s(
                    (str(preview_hash) + str(final_hash)).encode("utf-8"),
                    digest_size=16,
                ).hexdigest()
            if not window_plan:
                window_capacity = (
                    baseline_meta.get("window_capacity")
                    if isinstance(baseline_meta, dict)
                    else {}
                )
                window_plan = {
                    "profile": (profile or "").lower() or "baseline",
                    "requested_preview": int(preview_count),
                    "requested_final": int(final_count),
                    "actual_preview": int(preview_count),
                    "actual_final": int(final_count),
                    "coverage_ok": True,
                    "capacity": window_capacity or {},
                }
            if isinstance(window_plan, dict):
                preview_masks = pairing_schedule["preview"].get("attention_masks") or []
                final_masks = pairing_schedule["final"].get("attention_masks") or []
                preview_total_tokens = sum(
                    sum(_tensor_or_list_to_ints(mask)) for mask in preview_masks
                ) or int(dataset_meta.get("preview_total_tokens", 0) or 0)
                final_total_tokens = sum(
                    sum(_tensor_or_list_to_ints(mask)) for mask in final_masks
                ) or int(dataset_meta.get("final_total_tokens", 0) or 0)
                min_tokens_target = _resolve_pm_min_tokens_target(
                    tier=tier
                    or getattr(
                        getattr(cfg, "auto", None),
                        "tier",
                        None,
                    ),
                    profile=profile,
                )
                window_plan["preview_total_tokens"] = int(preview_total_tokens)
                window_plan["final_total_tokens"] = int(final_total_tokens)
                window_plan["min_tokens_target"] = int(min_tokens_target)
                window_plan["tokens_floor_met"] = (
                    int(preview_total_tokens) + int(final_total_tokens)
                ) >= int(min_tokens_target)
                dataset_meta["min_tokens_target"] = int(min_tokens_target)
                dataset_meta["tokens_floor_met"] = bool(window_plan["tokens_floor_met"])
            if isinstance(window_plan, dict):
                dataset_meta.setdefault("window_plan", window_plan)
                capacity_meta = window_plan.get("capacity")
                if capacity_meta and "window_capacity" not in dataset_meta:
                    dataset_meta["window_capacity"] = capacity_meta
        elif cfg.dataset.provider:
            _event(
                console,
                "DATA",
                f"Loading dataset: {cfg.dataset.provider}",
                emoji="📊",
                profile=profile_normalized,
            )
            # Pass through provider-specific kwargs when available
            provider_kwargs = {}
            for key in (
                "dataset_name",
                "config_name",
                "text_field",
                "src_field",
                "tgt_field",
                "cache_dir",
                "max_samples",
                # Local providers (e.g., local_jsonl)
                "file",
                "path",
                "data_files",
            ):
                try:
                    val = getattr(cfg.dataset, key)
                except (AttributeError, TypeError):
                    val = None
                if val is not None and val != "":
                    provider_kwargs[key] = val
            # Resolve provider kind from config (supports string or mapping with kind)
            provider_val = getattr(cfg.dataset, "provider", None)
            provider_name = None
            if isinstance(provider_val, dict):
                provider_name = provider_val.get("kind")
                # Include nested provider-specific kwargs
                for k, v in provider_val.items():
                    if k != "kind" and v is not None and v != "":
                        provider_kwargs[k] = v
            elif isinstance(provider_val, str):
                provider_name = provider_val  # noqa: F841
            else:
                # Support mapping-like provider configs (e.g., _Obj with .get)
                try:
                    _ = provider_val.get("kind")  # type: ignore[attr-defined]
                    # Try to expose nested entries
                    try:
                        for k, v in provider_val._data.items():  # type: ignore[attr-defined]
                            if k != "kind" and v is not None and v != "":
                                provider_kwargs[k] = v
                    except (AttributeError, TypeError):
                        # Fallback: if items() exists
                        try:
                            for k, v in provider_val.items():  # type: ignore[attr-defined]
                                if k != "kind" and v is not None and v != "":
                                    provider_kwargs[k] = v
                        except (AttributeError, TypeError):
                            pass
                except (AttributeError, TypeError):
                    _ = None
            data_provider, resolved_split, used_fallback_split = (
                _resolve_provider_and_split(
                    cfg,
                    model_profile,
                    get_provider_fn=get_provider,
                    provider_kwargs=provider_kwargs,
                    console=console,
                    resolved_device=resolved_device,
                    emit=_provider_event,
                )
            )

            # Load tokenizer for dataset processing
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

            dataset_stride = getattr(
                cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", 0) // 2
            )
            release_profile = (profile or "").lower() == "release"
            if release_profile and not pairing_schedule:
                estimate_fn = getattr(data_provider, "estimate_capacity", None)
                if callable(estimate_fn):
                    capacity_fast = bool(getattr(cfg.eval, "capacity_fast", False))
                    capacity_meta = estimate_fn(
                        tokenizer=tokenizer,
                        seq_len=cfg.dataset.seq_len,
                        stride=dataset_stride,
                        split=resolved_split,
                        target_total=requested_preview + requested_final,
                        fast_mode=capacity_fast,
                    )
                    variance_policy = getattr(cfg.guards, "variance", None)
                    max_calibration = (
                        getattr(variance_policy, "max_calib", 0)
                        if variance_policy is not None
                        else 0
                    )
                    try:
                        window_plan = _maybe_plan_release_windows(
                            capacity_meta,
                            requested_preview=requested_preview,
                            requested_final=requested_final,
                            max_calibration=max_calibration,
                            console=console,
                        )
                    except RuntimeError as err:
                        _event(console, "FAIL", str(err), emoji="❌", profile=profile)
                        raise typer.Exit(1) from err

                    actual_per_arm = int(window_plan["actual_preview"])
                    effective_preview = actual_per_arm
                    effective_final = actual_per_arm
                    dataset_stride = getattr(
                        cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", 0)
                    )
                else:
                    _event(
                        console,
                        "WARN",
                        "Release profile requested but dataset provider does not expose capacity estimation; using configured window counts.",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

            preview_records: list[dict[str, Any]] = []
            final_records: list[dict[str, Any]] = []

            signature_transform = None
            if use_mlm:

                def _signature_transform(
                    preview_records_in: list[dict[str, Any]],
                    final_records_in: list[dict[str, Any]],
                ) -> list[dict[str, Any]]:
                    temp_preview_records = [
                        {
                            "input_ids": list(rec["input_ids"]),
                            "attention_mask": list(rec["attention_mask"]),
                            "dataset_index": rec.get("dataset_index"),
                            "window_id": rec.get("window_id"),
                        }
                        for rec in preview_records_in
                    ]
                    temp_final_records = [
                        {
                            "input_ids": list(rec["input_ids"]),
                            "attention_mask": list(rec["attention_mask"]),
                            "dataset_index": rec.get("dataset_index"),
                            "window_id": rec.get("window_id"),
                        }
                        for rec in final_records_in
                    ]
                    _apply_mlm_masks(
                        temp_preview_records,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="preview",
                    )
                    _apply_mlm_masks(
                        temp_final_records,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="final",
                    )
                    return temp_preview_records + temp_final_records

                signature_transform = _signature_transform

            try:
                effective_windows = _resolve_effective_windows(
                    data_provider=data_provider,
                    tokenizer=tokenizer,
                    seq_len=cfg.dataset.seq_len,
                    stride=getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
                    preview_n=effective_preview,
                    final_n=effective_final,
                    seed=getattr(cfg.dataset, "seed", 42),
                    split=resolved_split,
                    requested_preview=requested_preview,
                    requested_final=requested_final,
                    profile=profile_normalized,
                    signature_transform=signature_transform,
                    event_fn=lambda message: _event(
                        console,
                        "WARN",
                        message,
                        emoji="⚠️",
                        profile=profile_normalized,
                    ),
                )
            except RuntimeError as err:
                _fail_run(str(err))

            preview_records = effective_windows["preview_records"]
            final_records = effective_windows["final_records"]
            preview_count = int(effective_windows["actual_preview"])
            final_count = int(effective_windows["actual_final"])
            effective_preview = preview_count
            effective_final = final_count

            # Optional: provider-supplied labels for seq2seq
            provider_labels_prev = None
            provider_labels_fin = None
            try:
                provider_labels_prev = getattr(
                    data_provider, "last_preview_labels", None
                )
                provider_labels_fin = getattr(data_provider, "last_final_labels", None)
            except (AttributeError, TypeError):
                provider_labels_prev = None
                provider_labels_fin = None

            for idx_local, rec in enumerate(preview_records):
                if provider_labels_prev is not None and idx_local < len(
                    provider_labels_prev
                ):
                    rec["labels"] = _tensor_or_list_to_ints(
                        provider_labels_prev[idx_local]
                    )

            min_tokens_target = _resolve_pm_min_tokens_target(
                tier=tier
                or getattr(
                    getattr(cfg, "auto", None),
                    "tier",
                    None,
                ),
                profile=profile,
            )
            tokens_floor_met = (
                int(effective_windows["preview_total_tokens"])
                + int(effective_windows["final_total_tokens"])
            ) >= int(min_tokens_target)

            if window_plan is None:
                window_plan = {
                    "profile": (profile or "").lower() or "default",
                    "requested_preview": int(requested_preview),
                    "requested_final": int(requested_final),
                    "capacity": {},
                }

            window_plan["actual_preview"] = int(preview_count)
            window_plan["actual_final"] = int(final_count)
            window_plan["coverage_ok"] = (
                window_plan.get("coverage_ok", True) and preview_count == final_count
            )
            window_plan["preview_total_tokens"] = int(
                effective_windows["preview_total_tokens"]
            )
            window_plan["final_total_tokens"] = int(
                effective_windows["final_total_tokens"]
            )
            window_plan["min_tokens_target"] = int(min_tokens_target)
            window_plan["tokens_floor_met"] = bool(tokens_floor_met)
            if effective_windows["dedupe_adjustments"]:
                window_plan["dedupe_adjustments"] = list(
                    effective_windows["dedupe_adjustments"]
                )

            calibration_data: list[dict[str, Any]] = []
            preview_mask_total = 0
            final_mask_total = 0
            preview_mask_counts: list[int] = []
            final_mask_counts: list[int] = []
            if use_mlm:
                preview_mask_total, preview_mask_counts = _apply_mlm_masks(
                    preview_records,
                    tokenizer=tokenizer,
                    mask_prob=mask_prob,
                    seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    prefix="preview",
                )
                final_mask_total, final_mask_counts = _apply_mlm_masks(
                    final_records,
                    tokenizer=tokenizer,
                    mask_prob=mask_prob,
                    seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    prefix="final",
                )
            else:
                preview_mask_counts = [0] * len(preview_records)
                final_mask_counts = [0] * len(final_records)

            preview_sequences = [record["input_ids"] for record in preview_records]
            for idx, record in enumerate(preview_records):
                entry = {
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"],
                    "window_id": f"preview::{idx}",
                    "dataset_index": record.get("dataset_index"),
                    "mlm_masked": record.get("mlm_masked", 0),
                }
                if use_mlm:
                    entry["labels"] = record.get(
                        "labels", [-100] * len(record["input_ids"])
                    )
                calibration_data.append(entry)

            final_sequences = [record["input_ids"] for record in final_records]
            for idx, record in enumerate(final_records):
                entry = {
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"],
                    "window_id": f"final::{idx}",
                    "dataset_index": record.get("dataset_index"),
                    "mlm_masked": record.get("mlm_masked", 0),
                }
                if use_mlm:
                    entry["labels"] = record.get(
                        "labels", [-100] * len(record["input_ids"])
                    )
                elif provider_labels_fin is not None and idx < len(provider_labels_fin):
                    entry["labels"] = _tensor_or_list_to_ints(provider_labels_fin[idx])
                calibration_data.append(entry)

            masked_tokens_total = preview_mask_total + final_mask_total
            preview_hash = _hash_sequences(preview_sequences)
            final_hash = _hash_sequences(final_sequences)
            dataset_meta = {
                "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
                "tokenizer_hash": tokenizer_hash
                if tokenizer_hash is not None
                else _tokenizer_digest(tokenizer),
                "vocab_size": _safe_int(getattr(tokenizer, "vocab_size", 0)),
                "bos_token": getattr(tokenizer, "bos_token", None),
                "eos_token": getattr(tokenizer, "eos_token", None),
                "pad_token": getattr(tokenizer, "pad_token", None),
                "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
                "dataset_hash": hashlib.blake2s(
                    (preview_hash + final_hash).encode("utf-8"), digest_size=16
                ).hexdigest(),
                "preview_hash": preview_hash,
                "final_hash": final_hash,
                "preview_total_tokens": int(effective_windows["preview_total_tokens"]),
                "final_total_tokens": int(effective_windows["final_total_tokens"]),
                "min_tokens_target": int(min_tokens_target),
                "tokens_floor_met": bool(tokens_floor_met),
            }
            dataset_meta["loss_type"] = resolved_loss_type
            if use_mlm:
                dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
                dataset_meta["masked_tokens_final"] = int(final_mask_total)
                dataset_meta["masked_tokens_total"] = int(masked_tokens_total)
            if window_plan:
                dataset_meta["window_plan"] = window_plan
                capacity_meta = window_plan.get("capacity")
                if capacity_meta:
                    dataset_meta["window_capacity"] = capacity_meta
            strat_stats = getattr(data_provider, "stratification_stats", None)
            if strat_stats:
                dataset_meta["stratification"] = strat_stats
            scorer_profile = getattr(data_provider, "scorer_profile", None)
            if scorer_profile:
                dataset_meta["scorer_profile"] = scorer_profile

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

        # Prepare auto configuration for tier resolution
        # Build auto configuration with safe fallbacks when section/keys are absent
        try:
            auto_enabled = bool(cfg.auto.enabled)
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            auto_enabled = False
        try:
            auto_tier = cfg.auto.tier
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            auto_tier = "balanced"
        try:
            auto_probes = int(cfg.auto.probes)
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            auto_probes = 0
        try:
            auto_target_ratio = float(cfg.auto.target_pm_ratio)
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            auto_target_ratio = 2.0

        auto_config = {
            "enabled": auto_enabled,
            "tier": auto_tier,
            "probes": auto_probes,
            "target_pm_ratio": auto_target_ratio,
        }

        # Extract edit configuration parameters
        edit_config = {}
        if hasattr(cfg.edit, "plan") and cfg.edit.plan:
            try:
                # Accept plain dicts, dict-like wrappers, or nested objects
                plan_obj = getattr(cfg.edit, "plan", {})
                if isinstance(plan_obj, dict):
                    edit_config = dict(plan_obj)
                else:
                    # Best-effort unwrap for InvarLockConfig _Obj wrapper
                    plan_data = getattr(plan_obj, "_data", None)
                    if isinstance(plan_data, dict):
                        edit_config = dict(plan_data)
                    elif hasattr(plan_obj, "items"):
                        edit_config = dict(plan_obj)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                pass

        if (
            model_profile.module_selectors
            and "module_selectors" not in edit_config
            and isinstance(model_profile.module_selectors, dict)
        ):
            edit_config["module_selectors"] = {
                key: list(values)
                for key, values in model_profile.module_selectors.items()
            }

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
                run_policy_context = (
                    dict(run_context.get("run"))
                    if isinstance(run_context.get("run"), dict)
                    else {}
                )
                eval_policy_context = (
                    dict(run_context.get("eval"))
                    if isinstance(run_context.get("eval"), dict)
                    else {}
                )
                report["context"] = {
                    "profile": profile_normalized,
                    "auto": dict(auto_config),
                    "assurance": dict(run_context.get("assurance") or {}),
                    "run": run_policy_context,
                    "eval": eval_policy_context,
                }
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

            meta_payload = {
                "model_id": cfg.model.id,
                "adapter": cfg.model.adapter,
                "device": str(resolved_device),
                "commit": commit_value,
                "seed": seed_bundle["python"],
                "seeds": seed_bundle,
                "ts": datetime.now().isoformat(),
                "auto": auto_config,
            }
            if invarlock_version:
                meta_payload["invarlock_version"] = invarlock_version
            if env_flags:
                meta_payload["env_flags"] = env_flags
            if determinism_meta:
                meta_payload["determinism"] = determinism_meta
            report["meta"].update(meta_payload)
            if pm_acceptance_range:
                report["meta"]["pm_acceptance_range"] = pm_acceptance_range
            if pm_drift_band:
                report["meta"]["pm_drift_band"] = pm_drift_band
            report["meta"]["guard_overhead_threshold"] = guard_overhead_threshold
            report["meta"]["model_profile"] = {
                "family": model_profile.family,
                "default_loss": model_profile.default_loss,
                "module_selectors": model_profile.module_selectors,
                "invariants": list(model_profile.invariants),
                "cert_lints": [dict(lint) for lint in model_profile.cert_lints],
            }

            dataset_provider = getattr(cfg.dataset, "provider", None)
            if dataset_provider is None:
                dataset_provider = getattr(cfg.dataset, "dataset", None)
            report["data"].update(
                {
                    "dataset": _canonical_dataset_id(dataset_provider),
                    # Resolved split (explicit or inferred)
                    "split": resolved_split,
                    "seq_len": cfg.dataset.seq_len,
                    "stride": getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
                    "preview_n": _safe_int(preview_count),
                    "final_n": _safe_int(final_count),
                }
            )
            dataset_meta_context = core_report.context.get("dataset_meta", {})
            if isinstance(dataset_meta_context, dict):
                report["data"].update(dataset_meta_context)
                dataset_tokenizer_hash = dataset_meta_context.get("tokenizer_hash")
                if (
                    not tokenizer_hash
                    and isinstance(dataset_tokenizer_hash, str)
                    and dataset_tokenizer_hash
                ):
                    tokenizer_hash = dataset_tokenizer_hash

            if tokenizer_hash:
                report["meta"]["tokenizer_hash"] = tokenizer_hash

            # Snapshot/restore provenance (survives retries).
            try:
                prov = report.setdefault("provenance", {})
                prov["restore_failed"] = bool(snapshot_provenance.get("restore_failed"))
                prov["reload_path_used"] = bool(
                    snapshot_provenance.get("reload_path_used")
                )
            except (TypeError, KeyError):
                pass

            # Transfer edit information
            if hasattr(core_report, "edit") and core_report.edit:
                edit_deltas = core_report.edit.get("deltas", {})
                report["edit"].update(
                    {
                        "name": edit_op.name,
                        "plan_digest": core_report.edit.get(
                            "plan_digest", str(hash(str(core_report.edit)))
                        ),
                        "deltas": {
                            "params_changed": edit_deltas.get("params_changed", 0),
                            "sparsity": edit_deltas.get("sparsity", None),
                            "bitwidth_map": edit_deltas.get("bitwidth_map", None),
                            "layers_modified": edit_deltas.get("layers_modified", 0),
                        },
                    }
                )
                for key in (
                    "algorithm",
                    "algorithm_version",
                    "implementation",
                    "scope",
                    "ranking",
                    "grouping",
                    "budgets",
                    "seed",
                    "mask_digest",
                ):
                    if key in core_report.edit:
                        report["edit"][key] = copy.deepcopy(core_report.edit[key])
                if isinstance(core_report.context, dict):
                    core_report.context.setdefault("edit", {})
                    core_report.context["edit"].update(
                        {
                            "name": edit_op.name,
                            "params_changed": edit_deltas.get("params_changed", 0),
                            "layers_modified": edit_deltas.get("layers_modified", 0),
                        }
                    )

            if edit_label:
                report.setdefault("edit", {})
                report["edit"]["name"] = edit_label
                report["edit"]["algorithm"] = edit_label
                if isinstance(core_report.context, dict):
                    core_report.context.setdefault("edit", {})
                    core_report.context["edit"]["name"] = edit_label

            mask_artifact_path = _persist_ref_masks(core_report, run_dir)
            if mask_artifact_path:
                report.setdefault("artifacts", {})
                report["artifacts"]["masks_path"] = str(mask_artifact_path)

            # Transfer metrics (PM-only: do not write ppl_* fields)
            if hasattr(core_report, "metrics") and core_report.metrics:
                if isinstance(core_report.metrics, dict):
                    core_timings = core_report.metrics.get("timings")
                    if isinstance(core_timings, dict):
                        for key in (
                            "prepare",
                            "prepare_guards",
                            "edit",
                            "guards",
                            "eval",
                            "finalize",
                        ):
                            if key in core_timings:
                                try:
                                    timings[key] = float(core_timings[key])
                                except NUMERIC_EXCEPTIONS:
                                    timings[key] = core_timings[key]
                metrics_payload = {
                    "latency_ms_per_tok": core_report.metrics.get(
                        "latency_ms_per_tok", 0.0
                    ),
                    "memory_mb_peak": core_report.metrics.get("memory_mb_peak", 0.0),
                    "spectral": {},
                    "rmt": {},
                    "invariants": {},
                }
                window_plan_ctx = core_report.context.get("window_plan")
                if isinstance(window_plan_ctx, dict):
                    metrics_payload["window_plan"] = window_plan_ctx
                    capacity_meta = window_plan_ctx.get("capacity")
                    if isinstance(capacity_meta, dict):
                        metrics_payload["window_capacity"] = capacity_meta
                    stats_section = metrics_payload.setdefault("stats", {})
                    if isinstance(stats_section, dict):
                        stats_section.update(
                            {
                                "requested_preview": window_plan_ctx.get(
                                    "requested_preview"
                                ),
                                "requested_final": window_plan_ctx.get(
                                    "requested_final"
                                ),
                                "actual_preview": window_plan_ctx.get("actual_preview"),
                                "actual_final": window_plan_ctx.get("actual_final"),
                                "coverage_ok": window_plan_ctx.get("coverage_ok"),
                                "preview_total_tokens": window_plan_ctx.get(
                                    "preview_total_tokens"
                                ),
                                "final_total_tokens": window_plan_ctx.get(
                                    "final_total_tokens"
                                ),
                                "min_tokens_target": window_plan_ctx.get(
                                    "min_tokens_target"
                                ),
                                "tokens_floor_met": window_plan_ctx.get(
                                    "tokens_floor_met"
                                ),
                                "dedupe_adjustments": window_plan_ctx.get(
                                    "dedupe_adjustments"
                                ),
                            }
                        )
                optional_keys = [
                    "logloss_preview",
                    "logloss_final",
                    "logloss_delta",
                    "logloss_preview_ci",
                    "logloss_final_ci",
                    "logloss_delta_ci",
                    "bootstrap",
                    "window_overlap_fraction",
                    "window_match_fraction",
                    "window_pairing_reason",
                    "window_pairing_preview",
                    "window_pairing_final",
                    "paired_windows",
                    "paired_delta_summary",
                    "primary_metric_tail",
                    "preview_total_tokens",
                    "final_total_tokens",
                    "masked_tokens_total",
                    "masked_tokens_preview",
                    "masked_tokens_final",
                    "timings",
                    "guard_timings",
                    "memory_snapshots",
                    "gpu_memory_mb_peak",
                    "gpu_memory_reserved_mb_peak",
                    "reduction",
                ]
                for key in optional_keys:
                    if key in core_report.metrics:
                        metrics_payload[key] = core_report.metrics[key]
                metrics_payload["loss_type"] = resolved_loss_type
                if metrics_payload.get("loss_type") is None and isinstance(
                    dataset_meta_context, dict
                ):
                    metrics_payload["loss_type"] = dataset_meta_context.get(
                        "loss_type", resolved_loss_type
                    )
                if isinstance(dataset_meta_context, dict):
                    for meta_key in (
                        "masked_tokens_total",
                        "masked_tokens_preview",
                        "masked_tokens_final",
                    ):
                        if (
                            meta_key not in metrics_payload
                            and dataset_meta_context.get(meta_key) is not None
                        ):
                            metrics_payload[meta_key] = dataset_meta_context[meta_key]
                report["metrics"].update(metrics_payload)

            if guard_overhead_payload is not None:
                if bool(guard_overhead_payload.get("skipped", False)):
                    report["guard_overhead"] = guard_overhead_payload
                else:
                    # Compute guarded primary-metric snapshot; pass structured reports into validator
                    try:
                        # Map loss type to ppl family kind
                        lk = str(resolved_loss_type or "causal").lower()
                        if lk == "mlm":
                            pm_kind_for_overhead = "ppl_mlm"
                        elif lk in {"seq2seq", "s2s", "t5"}:
                            pm_kind_for_overhead = "ppl_seq2seq"
                        else:
                            pm_kind_for_overhead = "ppl_causal"

                        # Prefer computing from the in-memory core_report windows to avoid ordering issues
                        pm_guarded = _extract_pm_snapshot_for_overhead(
                            core_report, kind=pm_kind_for_overhead
                        )
                        if not isinstance(pm_guarded, dict) or not pm_guarded:
                            pm_guarded = _extract_pm_snapshot_for_overhead(
                                report, kind=pm_kind_for_overhead
                            )

                        guard_overhead_payload["guarded_report"] = (
                            {"metrics": {"primary_metric": pm_guarded}}
                            if isinstance(pm_guarded, dict) and pm_guarded
                            else None
                        )
                    except (AttributeError, TypeError, ValueError):
                        guard_overhead_payload["guarded_report"] = None
                    bare_struct = guard_overhead_payload.get("bare_report") or {}
                    guarded_struct = guard_overhead_payload.get("guarded_report") or {}
                    # Be robust to mocks or minimal objects returned by validators
                    result = validate_guard_overhead(
                        bare_struct,
                        guarded_struct,
                        overhead_threshold=guard_overhead_payload.get(
                            "overhead_threshold", guard_overhead_threshold
                        ),
                    )
                    guard_overhead_payload = _finalize_guard_overhead_payload(
                        guard_overhead_payload,
                        result,
                    )
                    report["guard_overhead"] = guard_overhead_payload

            had_baseline = bool(baseline and Path(baseline).exists())
            serialized_evaluation_windows = _serialize_evaluation_windows(
                getattr(core_report, "evaluation_windows", None)
            )
            if serialized_evaluation_windows:
                report["evaluation_windows"] = serialized_evaluation_windows
            elif had_baseline and (profile or "").lower() in {"ci", "release"}:
                _event(
                    console,
                    "FAIL",
                    "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but evaluation windows were not produced. Check capacity/pairing config.",
                    emoji="❌",
                    profile=profile_normalized,
                )
                raise typer.Exit(3)
            else:
                # Populate evaluation_windows directly from assembled records when the
                # runner did not provide a structured window payload. This ensures
                # provenance (provider_digest) can be computed even in lightweight/dev
                # runs and unit tests that stub the runner.
                try:
                    report["evaluation_windows"] = _build_fallback_evaluation_windows(
                        preview_records,
                        final_records,
                        use_mlm=use_mlm,
                        preview_mask_counts=preview_mask_counts,
                        final_mask_counts=final_mask_counts,
                    )
                except NON_FATAL_RUNTIME_EXCEPTIONS:
                    # Best-effort: provenance digest will be skipped if windows cannot be built
                    pass

            # Attach provider digest and dataset split provenance when available
            try:
                prov = report.setdefault("provenance", {})
                # Always record dataset split provenance for visibility
                try:
                    prov["dataset_split"] = str(resolved_split)
                    prov["split_fallback"] = bool(used_fallback_split)
                except (TypeError, ValueError):
                    pass
                provider_digest = _compute_provider_digest(report)
                if provider_digest:
                    prov["provider_digest"] = provider_digest
                    # Attach digest version for future evolution
                    prov["digest_version"] = 1
                    # Strict parity checks in CI/Release when baseline present
                    try:
                        if isinstance(baseline_report_data, dict):
                            base_digest = None
                            base_prov = baseline_report_data.get("provenance")
                            if isinstance(base_prov, dict):
                                base_pd = base_prov.get("provider_digest")
                                if isinstance(base_pd, dict):
                                    base_digest = base_pd
                            if base_digest is None:
                                base_digest = _compute_provider_digest(
                                    baseline_report_data
                                )
                            _enforce_provider_parity(
                                provider_digest,
                                base_digest,
                                profile=(str(profile).lower() if profile else None),
                            )
                    except InvarlockError as ce:
                        console.print(str(ce))
                        # Map to profile-aware exit code: dev→1, ci/release→3
                        raise typer.Exit(
                            _resolve_exit_code(ce, profile=profile)
                        ) from None
                    except RuntimeError as _e:
                        _fail_run(str(_e))
                    except NON_FATAL_RUNTIME_EXCEPTIONS:
                        pass
            except (typer.Exit, SystemExit, click.exceptions.Exit):
                raise
            except NON_FATAL_RUNTIME_EXCEPTIONS:
                pass

            # Transfer guard results
            if hasattr(core_report, "guards") and core_report.guards:
                for guard_name, guard_result in core_report.guards.items():
                    guard_entry = {
                        "name": guard_name,
                        "passed": guard_result.get("passed"),
                        "action": guard_result.get("action"),
                        "policy": guard_result.get("policy", {}),
                        "metrics": guard_result.get("metrics", {}),
                        "actions": guard_result.get("actions", []),
                        "violations": guard_result.get("violations", []),
                        "warnings": guard_result.get("warnings", []),
                        "errors": guard_result.get("errors", []),
                        "details": guard_result.get("details", {}),
                    }
                    for extra_key in ("final_z_scores", "module_family_map"):
                        if extra_key in guard_result:
                            guard_entry[extra_key] = guard_result[extra_key]
                    report["guards"].append(guard_entry)

            # Set artifacts
            report["artifacts"].update(
                {
                    "events_path": str(run_config.event_path)
                    if run_config.event_path
                    else "",
                    "logs_path": "",
                    "checkpoint_path": None,
                }
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

            # Set flags
            report["flags"].update(
                {
                    "guard_recovered": any(
                        not g.get("passed", True)
                        for g in core_report.guards.values()
                        if hasattr(core_report, "guards") and core_report.guards
                    ),
                    "rollback_reason": None,
                }
            )

            metrics_section = report.get("metrics", {}) or {}
            data_section = report.get("data", {}) or {}
            preview_count_report = data_section.get("preview_n")
            final_count_report = data_section.get("final_n")

            # Classification metric (accuracy) — deterministic smoke path
            # If loss type is explicitly 'classification', derive accuracy
            # counts from evaluation windows using a deterministic label rule.
            try:
                loss_type_ctx = (
                    run_config.context.get("eval", {})
                    .get("loss", {})
                    .get("resolved_type")
                )
            except (AttributeError, TypeError, KeyError):
                loss_type_ctx = None
            if str(loss_type_ctx).lower() == "classification":
                try:
                    from invarlock.eval.primary_metric import compute_accuracy_counts

                    # Prefer in-memory core_report.evaluation_windows (includes input_ids)
                    ew = {}
                    try:
                        if hasattr(core_report, "evaluation_windows") and isinstance(
                            core_report.evaluation_windows, dict
                        ):
                            ew = core_report.evaluation_windows  # type: ignore[assignment]
                    except (AttributeError, TypeError):
                        ew = {}
                    if not ew:
                        # Fallback to the soon-to-be persisted report windows (may lack input_ids)
                        ew = (
                            report.get("evaluation_windows", {})
                            if isinstance(report.get("evaluation_windows"), dict)
                            else {}
                        )
                    prev_rec = []
                    fin_rec = []
                    if isinstance(ew, dict):
                        prev = ew.get("preview", {})
                        fin = ew.get("final", {})
                        if isinstance(prev, dict):
                            prev_rec = [
                                {"input_ids": seq}
                                for seq in prev.get("input_ids", []) or []
                                if isinstance(seq, list)
                            ]
                        if isinstance(fin, dict):
                            fin_rec = [
                                {"input_ids": seq}
                                for seq in fin.get("input_ids", []) or []
                                if isinstance(seq, list)
                            ]
                    c_prev, n_prev = compute_accuracy_counts(prev_rec)
                    c_fin, n_fin = compute_accuracy_counts(fin_rec)
                    # If we could not derive counts (no windows persisted), fall back to
                    # deterministic pseudo-accuracy based on configured window counts.
                    used_pseudo_counts = False
                    if n_prev == 0 and n_fin == 0:
                        try:
                            prev_n_cfg = getattr(cfg.dataset, "preview_n", None)
                            fin_n_cfg = getattr(cfg.dataset, "final_n", None)
                        except (AttributeError, TypeError):
                            prev_n_cfg = None
                            fin_n_cfg = None
                        try:
                            prev_n = int(preview_count_report or prev_n_cfg or 0)
                            fin_n = int(final_count_report or fin_n_cfg or 0)
                        except NUMERIC_EXCEPTIONS:
                            prev_n = 0
                            fin_n = 0
                        c_prev, n_prev = (prev_n, prev_n) if prev_n > 0 else (0, 0)
                        c_fin, n_fin = (fin_n, fin_n) if fin_n > 0 else (0, 0)
                        used_pseudo_counts = prev_n > 0 or fin_n > 0
                    classification_metrics = {
                        "preview": {"correct_total": int(c_prev), "total": int(n_prev)},
                        "final": {"correct_total": int(c_fin), "total": int(n_fin)},
                    }
                    # Tag source of counts for downstream rendering/doctor
                    if used_pseudo_counts:
                        classification_metrics["counts_source"] = "pseudo_config"
                        # Add a provenance crumb for transparency
                        try:
                            prov = report.setdefault("provenance", {})
                            notes = prov.setdefault("metric_notes", [])
                            if isinstance(notes, list):
                                notes.append(
                                    "accuracy: pseudo counts from preview_n/final_n"
                                )
                        except (TypeError, KeyError, AttributeError):
                            pass
                    else:
                        classification_metrics["counts_source"] = "measured"
                    report.setdefault("metrics", {})["classification"] = (
                        classification_metrics
                    )
                    # Convenience: top-level accuracy (final)
                    if n_fin > 0:
                        report["metrics"]["accuracy"] = float(c_fin / n_fin)
                except (
                    ImportError,
                    ModuleNotFoundError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ):
                    pass

            match_fraction = metrics_section.get("window_match_fraction")
            if match_fraction is not None and not math.isclose(
                match_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9
            ):
                err = InvarlockError(
                    code="E001",
                    message=(
                        f"PAIRING-SCHEDULE-MISMATCH: window_match_fraction={match_fraction:.3f}"
                    ),
                    details={"window_match_fraction": float(match_fraction)},
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)

            overlap_fraction = metrics_section.get("window_overlap_fraction")
            if overlap_fraction is not None and overlap_fraction > 1e-9:
                err = InvarlockError(
                    code="E001",
                    message=(
                        f"PAIRING-SCHEDULE-MISMATCH: window_overlap_fraction={overlap_fraction:.3f}"
                    ),
                    details={"window_overlap_fraction": float(overlap_fraction)},
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)

            # Paired-run enforcement: baseline provided must be truly paired in CI/Release.
            if baseline and profile_normalized in {"ci", "release"}:
                pairing_reason = metrics_section.get("window_pairing_reason")
                if pairing_reason is not None:
                    err = InvarlockError(
                        code="E001",
                        message=(
                            "PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but run was not paired "
                            f"(window_pairing_reason={pairing_reason})"
                        ),
                        details={"window_pairing_reason": pairing_reason},
                    )
                    code = _resolve_exit_code(err, profile=profile_normalized)
                    console.print(f"[red]{err}[/red]")
                    raise typer.Exit(code)

                paired_windows_val = metrics_section.get("paired_windows")
                paired_windows_int = None
                try:
                    if paired_windows_val is not None and not isinstance(
                        paired_windows_val, bool
                    ):
                        paired_windows_int = int(paired_windows_val)
                except NUMERIC_EXCEPTIONS:
                    paired_windows_int = None
                if paired_windows_int is None or paired_windows_int <= 0:
                    err = InvarlockError(
                        code="E001",
                        message=(
                            "PAIRED-WINDOWS-COLLAPSED: paired_windows<=0 under paired baseline. "
                            "Check device stability, dataset windows, or edit scope."
                        ),
                        details={
                            "paired_windows": paired_windows_val,
                            "profile": profile_normalized,
                        },
                    )
                    code = _resolve_exit_code(err, profile=profile_normalized)
                    console.print(f"[red]{err}[/red]")
                    raise typer.Exit(code)

            expected_preview = effective_preview or getattr(
                cfg.dataset, "preview_n", preview_count_report
            )
            expected_final = effective_final or getattr(
                cfg.dataset, "final_n", final_count_report
            )
            if (
                preview_count_report is not None
                and expected_preview is not None
                and int(preview_count_report) != int(expected_preview)
            ) or (
                final_count_report is not None
                and expected_final is not None
                and int(final_count_report) != int(expected_final)
            ):
                err = InvarlockError(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: counts do not match configuration after stratification"
                    ),
                    details={
                        "preview_used": int(preview_count_report or -1),
                        "preview_expected": int(expected_preview or -1),
                        "final_used": int(final_count_report or -1),
                        "final_expected": int(expected_final or -1),
                    },
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)

            # Compute metric-v1 snapshot (primary_metric) — canonical path
            try:
                metric_kind_resolved, _provider_kind, metric_opts = (
                    _resolve_metric_and_provider(
                        cfg,
                        model_profile,
                        resolved_loss_type=resolved_loss_type,
                        metric_kind_override=metric_kind,
                    )
                )
                if metric_kind_resolved:
                    from invarlock.eval.primary_metric import (
                        compute_primary_metric_from_report,
                    )

                    pm = compute_primary_metric_from_report(
                        report, kind=metric_kind_resolved, baseline=baseline_report_data
                    )
                    core_primary_metric = None
                    if hasattr(core_report, "metrics") and isinstance(
                        core_report.metrics, dict
                    ):
                        core_primary_metric = core_report.metrics.get("primary_metric")
                    pm = _merge_primary_metric_health(pm, core_primary_metric)
                    report.setdefault("metrics", {})["primary_metric"] = pm
                    # Attach configured reps/ci_level when provided
                    if metric_opts:
                        try:
                            if "reps" in metric_opts:
                                report["metrics"]["primary_metric"]["reps"] = int(
                                    metric_opts["reps"]
                                )  # type: ignore[index]
                            if "ci_level" in metric_opts:
                                report["metrics"]["primary_metric"]["ci_level"] = float(
                                    metric_opts["ci_level"]
                                )  # type: ignore[index]
                        except (TypeError, ValueError, KeyError):
                            pass
                # Shadow parity check against ppl_* fields (best-effort)
                try:
                    pm_blk = report.get("metrics", {}).get("primary_metric", {})
                    ppl_final_v1 = float(pm_blk.get("final"))
                    ppl_final_v2 = float(pm.get("final", float("nan")))
                    if math.isfinite(ppl_final_v1) and math.isfinite(ppl_final_v2):
                        if not math.isclose(
                            ppl_final_v1, ppl_final_v2, rel_tol=1e-9, abs_tol=1e-9
                        ):
                            report.setdefault("metrics", {}).setdefault(
                                "_metric_v1_mismatch", {}
                            )["ppl_final_diff"] = ppl_final_v2 - ppl_final_v1
                    # Optional: dual-write diffs logging for ppl_* metrics
                    debug_diffs = str(
                        os.environ.get("DEBUG_METRIC_DIFFS", "")
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    if debug_diffs and str(pm.get("kind", "")).startswith("ppl"):
                        diffs_line = _format_debug_metric_diffs(
                            pm, report.get("metrics", {}), baseline_report_data
                        )
                        if diffs_line:
                            console.print(
                                "[dim]DEBUG_METRIC_DIFFS: " + diffs_line + "[/dim]"
                            )
                except NON_FATAL_RUNTIME_EXCEPTIONS:
                    pass
            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
            ):
                # Non-fatal: metric-v1 snapshot should not break runs
                pass

            # No deprecation notices in dev-phase: primary_metric is canonical.

            # Derive dataset.windows.stats (PM-only surface)
            try:
                ds = report.setdefault("dataset", {}).setdefault("windows", {})
                stats = ds.setdefault("stats", {})
                if match_fraction is not None:
                    stats["window_match_fraction"] = float(match_fraction)
                if overlap_fraction is not None:
                    stats["window_overlap_fraction"] = float(overlap_fraction)
                try:
                    if isinstance(window_plan, dict) and "coverage_ok" in window_plan:
                        stats["coverage"] = bool(window_plan.get("coverage_ok"))
                        stats["preview_total_tokens"] = window_plan.get(
                            "preview_total_tokens"
                        )
                        stats["final_total_tokens"] = window_plan.get(
                            "final_total_tokens"
                        )
                        stats["min_tokens_target"] = window_plan.get(
                            "min_tokens_target"
                        )
                        stats["tokens_floor_met"] = window_plan.get("tokens_floor_met")
                except (AttributeError, KeyError, TypeError):
                    pass
            except (AttributeError, KeyError, TypeError):
                pass

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
                window_plan=window_plan,
                dataset_meta=dataset_meta,
                match_fraction=match_fraction,
                overlap_fraction=overlap_fraction,
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
                from invarlock.reporting.report_builder import make_report
                from invarlock.reporting.report_telemetry import (
                    telemetry_output_enabled,
                    telemetry_summary_line,
                )

                try:
                    baseline_report = baseline_report_data
                    if baseline_report is None and baseline:
                        baseline_path = Path(baseline)
                        with baseline_path.open(encoding="utf-8") as f:
                            baseline_report = json.load(f)

                    if baseline_report is None:
                        raise FileNotFoundError("Baseline report unavailable")

                    _event(
                        console,
                        "EXEC",
                        "Generating evaluation report...",
                        emoji="📜",
                        profile=profile_normalized,
                    )
                    evaluation_report = make_report(report, baseline_report)
                    if telemetry_output_enabled():
                        summary_line = telemetry_summary_line(evaluation_report)
                        if summary_line:
                            console.print(summary_line, markup=False)

                    validation = evaluation_report.get("validation", {})
                    result_summary = _build_retry_result_summary(validation)
                    report_passed = bool(result_summary["passed"])
                    failed_gates = list(result_summary["failures"])
                    retry_controller.record_attempt(
                        attempt, result_summary, edit_config
                    )

                    if report_passed:
                        _event(
                            console,
                            "PASS",
                            "Evaluation report PASSED all gates!",
                            emoji="✅",
                            profile=profile_normalized,
                        )
                        break
                    else:
                        _event(
                            console,
                            "FAIL",
                            f"Evaluation report FAILED gates: {', '.join(failed_gates)}",
                            emoji="⚠️",
                            profile=profile_normalized,
                        )

                        edit_config, head_adjustment = _apply_mask_only_head_autotune(
                            edit_config, validation
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

                        should_retry = retry_controller.should_retry(report_passed)
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
                        else:
                            _event(
                                console,
                                "FAIL",
                                f"Exhausted retry budget after {attempt} attempts",
                                emoji="❌",
                                profile=profile_normalized,
                            )
                            break

                except Exception as report_error:
                    _event(
                        console,
                        "WARN",
                        f"Evaluation report validation failed: {report_error}",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                    if retry_controller:
                        retry_controller.record_attempt(
                            attempt,
                            {
                                "passed": False,
                                "failures": ["report_error"],
                                "validation": {},
                            },
                            edit_config,
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
