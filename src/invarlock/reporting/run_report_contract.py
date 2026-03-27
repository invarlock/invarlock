from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunReportAssemblyResult:
    report: dict[str, Any]
    timings: dict[str, float]
    provenance_result: Any
    metrics_enrichment: Any


@dataclass(frozen=True)
class RunReportPersistenceResult:
    saved_files: dict[str, str]
    report_path_out: str | None
    telemetry_saved_path: str | None = None
    telemetry_error: str | None = None


def _detect_commit_value(cfg: Any) -> str:
    commit_value = getattr(getattr(cfg, "meta", None), "commit", "") or ""
    if commit_value:
        return str(commit_value)
    try:
        git_path = shutil.which("git")
        if git_path:
            return (
                subprocess.check_output(
                    [git_path, "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8", "ignore")
                .strip()
            )
    except (OSError, subprocess.SubprocessError):
        return ""
    return ""


def _resolve_version() -> str | None:
    try:
        from invarlock import __version__ as invarlock_version

        return invarlock_version
    except ImportError:
        return None


def _collect_env_flags(
    optional_torch_fn: Any, environ: dict[str, str]
) -> dict[str, object]:
    env_flags: dict[str, object] = {}
    try:
        torch_mod = optional_torch_fn()
        if torch_mod is not None:
            try:
                det_enabled = getattr(
                    torch_mod, "are_deterministic_algorithms_enabled", None
                )
                if callable(det_enabled):
                    env_flags["torch_deterministic_algorithms"] = bool(det_enabled())
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                tf32_matmul = getattr(
                    getattr(torch_mod.backends, "cuda", object()),
                    "matmul",
                    None,
                )
                if tf32_matmul is not None and hasattr(tf32_matmul, "allow_tf32"):
                    env_flags["cuda_matmul_allow_tf32"] = bool(tf32_matmul.allow_tf32)
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
        env_flags["CUBLAS_WORKSPACE_CONFIG"] = environ.get("CUBLAS_WORKSPACE_CONFIG")
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return {}
    return env_flags


def assemble_run_report(
    *,
    core_report: Any,
    cfg: Any,
    run_context: dict[str, Any] | None,
    profile_normalized: str | None,
    auto_config: dict[str, Any] | None,
    resolved_device: str,
    seed_bundle: dict[str, Any],
    guard_overhead_threshold: float,
    model_profile: Any,
    determinism_meta: dict[str, Any],
    pm_acceptance_range: tuple[float, float] | None,
    pm_drift_band: tuple[float, float] | None,
    tokenizer_hash: str | None,
    resolved_split: str | None,
    preview_count: Any,
    final_count: Any,
    snapshot_provenance: dict[str, bool],
    edit_op: Any,
    edit_label: str | None,
    run_dir: Path,
    run_config: Any,
    resolved_loss_type: str,
    timings: dict[str, float],
    guard_overhead_payload: dict[str, Any] | None,
    baseline: str | None,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    use_mlm: bool,
    preview_mask_counts: list[int] | None,
    final_mask_counts: list[int] | None,
    profile: str | None,
    used_fallback_split: bool,
    baseline_report_data: dict[str, Any] | None,
    effective_preview: Any,
    effective_final: Any,
    metric_kind: str | None,
    window_plan: dict[str, Any] | None,
    debug_metric_diffs_enabled: bool,
    create_empty_report_fn: Any,
    build_run_report_context_fn: Any,
    build_run_report_meta_fn: Any,
    canonical_dataset_id_fn: Any,
    safe_int_fn: Any,
    build_run_report_data_fn: Any,
    build_snapshot_provenance_fn: Any,
    build_edit_payload_fn: Any,
    persist_ref_masks_fn: Any,
    build_artifacts_payload_fn: Any,
    merge_core_timing_metrics_fn: Any,
    build_metrics_payload_fn: Any,
    prepare_guard_overhead_report_fn: Any,
    finalize_run_provenance_fn: Any,
    build_guard_entries_fn: Any,
    build_flags_payload_fn: Any,
    enrich_run_report_metrics_fn: Any,
    optional_torch_fn: Any,
    environ: dict[str, str],
) -> RunReportAssemblyResult:
    if not hasattr(core_report, "context") or core_report.context is None:
        core_report.context = {}

    report = create_empty_report_fn()

    try:
        report["context"] = build_run_report_context_fn(
            profile_normalized=profile_normalized,
            auto_config=auto_config,
            run_context=run_context,
        )
    except (TypeError, ValueError, KeyError):
        pass

    report["meta"].update(
        build_run_report_meta_fn(
            model_id=cfg.model.id,
            adapter=cfg.model.adapter,
            resolved_device=resolved_device,
            commit_value=_detect_commit_value(cfg),
            seed_bundle=seed_bundle,
            auto_config=auto_config,
            guard_overhead_threshold=guard_overhead_threshold,
            model_profile=model_profile,
            timestamp=datetime.now().isoformat(),
            invarlock_version=_resolve_version(),
            env_flags=_collect_env_flags(optional_torch_fn, environ),
            determinism_meta=determinism_meta,
            pm_acceptance_range=pm_acceptance_range,
            pm_drift_band=pm_drift_band,
        )
    )

    dataset_provider = getattr(cfg.dataset, "provider", None)
    if dataset_provider is None:
        dataset_provider = getattr(cfg.dataset, "dataset", None)
    dataset_meta_context = core_report.context.get("dataset_meta", {})
    data_payload, tokenizer_hash = build_run_report_data_fn(
        canonical_dataset_id=canonical_dataset_id_fn(dataset_provider),
        resolved_split=resolved_split,
        seq_len=cfg.dataset.seq_len,
        stride=getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
        preview_count=safe_int_fn(preview_count),
        final_count=safe_int_fn(final_count),
        dataset_meta_context=dataset_meta_context,
        tokenizer_hash=tokenizer_hash,
    )
    report["data"].update(data_payload)

    if tokenizer_hash:
        report["meta"]["tokenizer_hash"] = tokenizer_hash

    try:
        provenance = report.setdefault("provenance", {})
        provenance.update(build_snapshot_provenance_fn(snapshot_provenance))
    except (TypeError, KeyError):
        pass

    edit_payload, context_edit = build_edit_payload_fn(
        core_edit=(
            core_report.edit
            if hasattr(core_report, "edit") and isinstance(core_report.edit, dict)
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

    mask_artifact_path = persist_ref_masks_fn(core_report, run_dir)
    report["artifacts"].update(
        build_artifacts_payload_fn(
            event_path=run_config.event_path,
            mask_artifact_path=mask_artifact_path,
        )
    )

    current_timings = dict(timings)
    if hasattr(core_report, "metrics") and core_report.metrics:
        current_timings = merge_core_timing_metrics_fn(
            current_timings, core_report.metrics
        )
        metrics_payload = build_metrics_payload_fn(
            core_metrics=core_report.metrics,
            window_plan_context=core_report.context.get("window_plan"),
            dataset_meta_context=dataset_meta_context,
            resolved_loss_type=resolved_loss_type,
        )
        report["metrics"].update(metrics_payload)

    if guard_overhead_payload is not None:
        report["guard_overhead"] = prepare_guard_overhead_report_fn(
            guard_overhead_payload,
            resolved_loss_type=resolved_loss_type,
            core_report=core_report,
            report=report,
            default_threshold=guard_overhead_threshold,
        )

    provenance_result = finalize_run_provenance_fn(
        report=report,
        core_report=core_report,
        preview_records=preview_records,
        final_records=final_records,
        use_mlm=use_mlm,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
        had_baseline=bool(baseline and Path(baseline).exists()),
        profile=profile,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
        baseline_report_data=baseline_report_data,
    )

    report["guards"].extend(
        build_guard_entries_fn(
            core_report.guards
            if hasattr(core_report, "guards") and isinstance(core_report.guards, dict)
            else None
        )
    )

    report["flags"].update(
        build_flags_payload_fn(
            core_report.guards
            if hasattr(core_report, "guards") and isinstance(core_report.guards, dict)
            else None
        )
    )

    metrics_enrichment = enrich_run_report_metrics_fn(
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

    return RunReportAssemblyResult(
        report=report,
        timings=current_timings,
        provenance_result=provenance_result,
        metrics_enrichment=metrics_enrichment,
    )


def persist_run_report_outputs(
    *,
    report: dict[str, Any],
    run_dir: Path,
    run_config: Any,
    console: Any,
    telemetry: bool,
    postprocess_and_summarize_fn: Any,
    save_telemetry_report_fn: Any,
) -> RunReportPersistenceResult:
    telemetry_path: Path | None = None
    if telemetry:
        telemetry_path = run_dir / "telemetry.json"
        report.setdefault("artifacts", {})["telemetry_path"] = str(telemetry_path)

    saved_files = postprocess_and_summarize_fn(
        report=report,
        run_dir=run_dir,
        run_config=run_config,
        console=console,
    )

    report_path_out = None
    try:
        if isinstance(saved_files, dict) and saved_files.get("json"):
            report_path_out = str(saved_files["json"])
    except (TypeError, KeyError):
        report_path_out = None

    telemetry_saved_path = None
    telemetry_error = None
    if telemetry and telemetry_path is not None:
        try:
            saved_path = save_telemetry_report_fn(
                report, run_dir, filename=telemetry_path.name
            )
            telemetry_saved_path = str(saved_path)
            if isinstance(saved_files, dict):
                saved_files["telemetry"] = telemetry_saved_path
        except Exception as exc:  # pragma: no cover - best-effort
            telemetry_error = str(exc)

    return RunReportPersistenceResult(
        saved_files=saved_files,
        report_path_out=report_path_out,
        telemetry_saved_path=telemetry_saved_path,
        telemetry_error=telemetry_error,
    )


__all__ = [
    "RunReportAssemblyResult",
    "RunReportPersistenceResult",
    "assemble_run_report",
    "persist_run_report_outputs",
]
