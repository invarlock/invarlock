from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from invarlock.core.backend_inventory import (
    BACKEND_INVENTORY_FILENAME,
    write_backend_inventory_sidecar,
)

from .report_types import RunReport
from .run_report_payloads import build_artifacts_payload as build_artifacts_payload
from .run_report_payloads import build_edit_payload as build_edit_payload
from .run_report_payloads import build_flags_payload as build_flags_payload
from .run_report_payloads import build_guard_entries as build_guard_entries
from .run_report_payloads import build_metrics_payload as build_metrics_payload
from .run_report_payloads import build_run_report_context as build_run_report_context
from .run_report_payloads import build_run_report_data as build_run_report_data
from .run_report_payloads import build_run_report_meta as build_run_report_meta
from .run_report_payloads import build_snapshot_provenance as build_snapshot_provenance
from .run_report_payloads import merge_core_timing_metrics as merge_core_timing_metrics

report_files = cast(Any, importlib.import_module("invarlock.reporting.report_files"))
_NON_FATAL_EXCEPTIONS = (AttributeError, KeyError, OSError, TypeError, ValueError)


@dataclass(frozen=True)
class RunReportAssemblyResult:
    report: RunReport
    timings: dict[str, float]
    provenance_result: Any
    metrics_enrichment: Any


@dataclass(frozen=True)
class RunReportPersistenceResult:
    saved_files: dict[str, str]
    report_path_out: str | None
    telemetry_saved_path: str | None = None
    telemetry_error: str | None = None


@dataclass(frozen=True)
class RunProvenanceResult:
    missing_evaluation_windows_for_baseline: bool = False
    missing_evaluation_windows_message: str | None = None


def _detect_commit_value(cfg: Any) -> str:
    commit_value = getattr(getattr(cfg, "meta", None), "commit", "") or ""
    return str(commit_value) if commit_value else ""


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


def _window_payload_has_signal(window_payload: Any) -> bool:
    if not isinstance(window_payload, Mapping):
        return False
    for key in (
        "window_ids",
        "example_ids",
        "logloss",
        "input_ids",
        "attention_masks",
        "token_counts",
        "masked_token_counts",
        "actual_token_counts",
        "labels",
        "records",
    ):
        value = window_payload.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _evaluation_windows_have_signal(serialized_windows: Any) -> bool:
    if not isinstance(serialized_windows, Mapping):
        return False
    return _window_payload_has_signal(serialized_windows.get("preview")) or (
        _window_payload_has_signal(serialized_windows.get("final"))
    )


def finalize_run_provenance(
    *,
    report: dict[str, Any],
    core_report: Any,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    use_mlm: bool,
    preview_mask_counts: list[int] | None,
    final_mask_counts: list[int] | None,
    had_baseline: bool,
    profile: str | None,
    resolved_split: str | None,
    used_fallback_split: bool,
    baseline_report_data: dict[str, Any] | None,
    serialize_evaluation_windows_fn: Any,
    build_fallback_evaluation_windows_fn: Any,
    compute_provider_digest_fn: Any,
    enforce_provider_parity_fn: Any,
) -> RunProvenanceResult:
    """Finalize evaluation windows plus run provenance and provider parity."""

    serialized_evaluation_windows = serialize_evaluation_windows_fn(
        getattr(core_report, "evaluation_windows", None)
    )
    if _evaluation_windows_have_signal(serialized_evaluation_windows):
        report["evaluation_windows"] = serialized_evaluation_windows
    else:
        try:
            fallback_evaluation_windows = build_fallback_evaluation_windows_fn(
                preview_records,
                final_records,
                use_mlm=use_mlm,
                preview_mask_counts=preview_mask_counts,
                final_mask_counts=final_mask_counts,
            )
            if fallback_evaluation_windows:
                report["evaluation_windows"] = fallback_evaluation_windows
        except _NON_FATAL_EXCEPTIONS:
            pass
        if (
            "evaluation_windows" not in report
            and had_baseline
            and (profile or "").lower() in {"ci", "release"}
        ):
            return RunProvenanceResult(
                missing_evaluation_windows_for_baseline=True,
                missing_evaluation_windows_message=(
                    "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing "
                    "requested but evaluation windows were not produced. Check "
                    "capacity/pairing config."
                ),
            )

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        report["provenance"] = provenance

    try:
        provenance["dataset_split"] = str(resolved_split)
        provenance["split_fallback"] = bool(used_fallback_split)
    except _NON_FATAL_EXCEPTIONS:
        pass

    try:
        provider_digest = compute_provider_digest_fn(report)
    except _NON_FATAL_EXCEPTIONS:
        provider_digest = None
    if not provider_digest:
        return RunProvenanceResult()

    provenance["provider_digest"] = provider_digest
    provenance["digest_version"] = 1

    if not isinstance(baseline_report_data, dict):
        return RunProvenanceResult()

    base_digest = None
    base_provenance = baseline_report_data.get("provenance")
    if isinstance(base_provenance, dict):
        base_provider_digest = base_provenance.get("provider_digest")
        if isinstance(base_provider_digest, dict):
            base_digest = base_provider_digest
    if base_digest is None:
        try:
            base_digest = compute_provider_digest_fn(baseline_report_data)
        except _NON_FATAL_EXCEPTIONS:
            base_digest = None

    enforce_provider_parity_fn(
        provider_digest,
        base_digest,
        profile=(str(profile).lower() if profile else None),
    )
    return RunProvenanceResult()


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
    report: RunReport,
    run_dir: Path,
    run_config: Any,
    telemetry: bool,
    save_telemetry_report_fn: Any,
) -> RunReportPersistenceResult:
    telemetry_path: Path | None = None
    if telemetry:
        telemetry_path = run_dir / "telemetry.json"
        report["artifacts"]["telemetry_path"] = str(telemetry_path)

    saved_paths = report_files.save_report(
        report,
        run_dir,
        formats=["json"],
        filename_prefix="report",
    )
    saved_files = {key: str(value) for key, value in saved_paths.items()}
    run_context = getattr(run_config, "context", None)
    backend_inventory = (
        run_context.get("_backend_inventory") if isinstance(run_context, dict) else None
    )
    if isinstance(backend_inventory, dict):
        backend_inventory = dict(backend_inventory)
        backend_inventory["load_smoke"] = backend_inventory.get("load_smoke") is True
        backend_inventory["inference_smoke"] = True
    existing_backend_inventory_path = run_dir / BACKEND_INVENTORY_FILENAME
    existing_backend_inventory = None
    if existing_backend_inventory_path.is_file():
        try:
            existing_backend_inventory = json.loads(
                existing_backend_inventory_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing_backend_inventory = None

    backend_inventory_path = write_backend_inventory_sidecar(
        report,
        run_dir,
        inventory=backend_inventory or existing_backend_inventory,
    )
    if backend_inventory_path is not None:
        saved_files["backend_inventory"] = str(backend_inventory_path)

    report_path_out = saved_files.get("json")
    if report_path_out:
        report_path_out = str(report_path_out)
    if not report_path_out:
        raise RuntimeError("run report persistence did not return a json artifact path")

    telemetry_saved_path = None
    telemetry_error = None
    if telemetry and telemetry_path is not None:
        try:
            saved_path = save_telemetry_report_fn(
                report, run_dir, filename=telemetry_path.name
            )
            telemetry_saved_path = str(saved_path)
            saved_files["telemetry"] = telemetry_saved_path
        except _NON_FATAL_EXCEPTIONS as exc:  # pragma: no cover - best-effort
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
    "RunProvenanceResult",
    "assemble_run_report",
    "build_artifacts_payload",
    "build_edit_payload",
    "build_flags_payload",
    "build_guard_entries",
    "build_metrics_payload",
    "build_run_report_context",
    "build_run_report_data",
    "build_run_report_meta",
    "build_snapshot_provenance",
    "finalize_run_provenance",
    "merge_core_timing_metrics",
    "persist_run_report_outputs",
]
