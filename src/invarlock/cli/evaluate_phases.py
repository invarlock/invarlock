"""Baseline/subject evaluation phases for the evaluate CLI command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from invarlock.cli import output as cli_output
from invarlock.core.evaluate_contract import (
    load_validated_baseline_report,
    require_run_report_artifact,
)

_PROFILE_DATASET_ERRORS = (RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class EvaluatePhaseRuntime:
    console: Any
    output_style: Any
    timings: dict[str, float]
    verbosity: int
    progress: bool
    info_fn: Any
    debug_fn: Any
    phase_fn: Any
    fail_fn: Any
    suppress_child_output_fn: Any
    load_yaml_fn: Any
    dump_yaml_fn: Any
    run_command_fn: Any
    json_load_fn: Any


@dataclass(frozen=True)
class BaselineEvaluationRequest:
    baseline_report: str | None
    profile_name: str
    tier_name: str
    adapter: str
    out: str
    device: str | None
    allow_network: bool
    allow_host_execution: bool
    allow_third_party_plugins: bool
    allow_remote_code: bool
    allow_unverified_provenance: bool
    prefer_local_files_only: bool
    no_color: bool
    baseline_cfg: dict[str, Any]
    baseline_label: str
    tmp_dir: Path


@dataclass(frozen=True)
class SubjectEvaluationRequest:
    baseline_report_path: Path
    preset_data: dict[str, Any]
    subject_model_id: str
    adapter: str
    out: str
    device: str | None
    profile_name: str
    tier_name: str
    guards_order: Any
    assurance_mode: str
    subject_label: str | None
    edit_config: str | None
    edit_label: str | None
    execution_mode: str
    allow_network: bool
    allow_host_execution: bool
    allow_third_party_plugins: bool
    allow_remote_code: bool
    allow_unverified_provenance: bool
    prefer_local_files_only: bool
    no_color: bool
    tmp_dir: Path


def _profile_effective_dataset_config(
    baseline_cfg: dict[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any] | None:
    dataset_cfg = baseline_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return None
    normalized_profile = str(profile_name or "").strip().lower()
    if normalized_profile in {"", "dev"}:
        return dataset_cfg
    try:
        from invarlock.core.config_loader import apply_profile
        from invarlock.core.config_runtime import InvarLockConfig

        effective_cfg = apply_profile(InvarLockConfig(baseline_cfg), normalized_profile)
        effective_payload = effective_cfg.model_dump()
    except _PROFILE_DATASET_ERRORS:
        return dataset_cfg
    effective_dataset = effective_payload.get("dataset")
    return effective_dataset if isinstance(effective_dataset, dict) else dataset_cfg


def run_baseline_evaluation_phase(
    request: BaselineEvaluationRequest,
    runtime: EvaluatePhaseRuntime,
) -> Path:
    if request.baseline_report:
        runtime.info_fn(
            "Using provided baseline report (skipping baseline evaluation)",
            tag="EXEC",
            emoji="♻️",
        )
        try:
            model_cfg = request.baseline_cfg.get("model")
            baseline_model_id = (
                model_cfg.get("id") if isinstance(model_cfg, dict) else ""
            )
            assurance_cfg = request.baseline_cfg.get("assurance")
            assurance_mode = (
                assurance_cfg.get("mode") if isinstance(assurance_cfg, dict) else None
            )
            dataset_cfg = _profile_effective_dataset_config(
                request.baseline_cfg,
                profile_name=request.profile_name,
            )
            baseline_report_path, _ = load_validated_baseline_report(
                Path(request.baseline_report),
                expected_model_id=str(baseline_model_id or ""),
                expected_profile=request.profile_name,
                expected_tier=request.tier_name,
                expected_adapter=str(request.adapter),
                expected_assurance_mode=str(assurance_mode or "off"),
                expected_dataset=dataset_cfg,
            )
        except typer.BadParameter as exc:
            runtime.fail_fn(str(getattr(exc, "message", exc)), exit_code=2)
        except Exception as exc:  # noqa: BLE001 - preserve existing failure surface
            if isinstance(exc, (typer.Exit, SystemExit)):
                raise
            runtime.fail_fn(str(getattr(exc, "message", exc)), exit_code=2)
        runtime.debug_fn(f"Baseline report: {baseline_report_path}")
        return baseline_report_path

    baseline_yaml = request.tmp_dir / "baseline_noop.yaml"
    runtime.dump_yaml_fn(baseline_yaml, request.baseline_cfg)

    runtime.phase_fn(1, 3, "BASELINE EVALUATION")
    runtime.info_fn("Running baseline (no-op edit)", tag="EXEC", emoji="🏁")
    runtime.debug_fn(f"Baseline config: {baseline_yaml}")

    with runtime.suppress_child_output_fn(runtime.verbosity == 0) as quiet_buffer:
        try:
            with cli_output.timed_step(
                console=runtime.console,
                style=runtime.output_style,
                timings=runtime.timings,
                key="baseline",
                tag="EXEC",
                message="Baseline",
                emoji="🏁",
            ):
                baseline_run_result = runtime.run_command_fn(
                    config=str(baseline_yaml),
                    profile=request.profile_name,
                    out=str(Path(request.out) / "source"),
                    tier=request.tier_name,
                    device=request.device,
                    until_pass=False,
                    max_attempts=1,
                    timeout=None,
                    edit_label=request.baseline_label,
                    style=runtime.output_style.name,
                    progress=runtime.progress,
                    timing=False,
                    allow_network=request.allow_network,
                    allow_host_execution=request.allow_host_execution,
                    allow_third_party_plugins=request.allow_third_party_plugins,
                    allow_remote_code=request.allow_remote_code,
                    allow_unverified_provenance=request.allow_unverified_provenance,
                    prefer_local_files_only=request.prefer_local_files_only,
                    no_color=request.no_color,
                )
        except typer.Exit:
            if quiet_buffer is not None:
                runtime.console.print(quiet_buffer.getvalue(), markup=False)
            raise
        except Exception:
            if quiet_buffer is not None:
                runtime.console.print(quiet_buffer.getvalue(), markup=False)
            raise

    try:
        baseline_report_path = require_run_report_artifact(
            baseline_run_result,
            stage="Baseline",
        )
    except Exception as exc:
        runtime.fail_fn(str(getattr(exc, "message", exc)), exit_code=1)
    runtime.debug_fn(f"Baseline report: {baseline_report_path}")
    return baseline_report_path


def run_subject_evaluation_phase(
    request: SubjectEvaluationRequest,
    runtime: EvaluatePhaseRuntime,
) -> tuple[Path, dict[str, Any]]:
    from invarlock.core.evaluate_plan import (
        build_subject_edit_run_config,
        build_subject_noop_run_config,
    )

    runtime.phase_fn(2, 3, "SUBJECT EVALUATION")
    baseline_report_str = str(request.baseline_report_path)
    if request.edit_config:
        edited_yaml = Path(request.edit_config)
        if not edited_yaml.exists():
            cli_output.print_event(
                runtime.console,
                "FAIL",
                f"Edit config not found: {edited_yaml}",
                style=runtime.output_style,
                emoji="❌",
            )
            raise typer.Exit(1)
        runtime.info_fn(
            "Running edited (demo edit via --edit-config)", tag="EXEC", emoji="✂️"
        )
        try:
            cfg_loaded: dict[str, Any] = runtime.load_yaml_fn(edited_yaml)
        except Exception as exc:
            cli_output.print_event(
                runtime.console,
                "FAIL",
                f"Failed to load edit config: {exc}",
                style=runtime.output_style,
                emoji="❌",
            )
            raise typer.Exit(1) from exc

        merged_edited_cfg = build_subject_edit_run_config(
            request.preset_data,
            cfg_loaded,
            subject_model_id=request.subject_model_id,
            adapter_name=str(request.adapter),
            output_dir=str(Path(request.out) / "edited"),
            profile=request.profile_name,
            tier=request.tier_name,
            guards_order=request.guards_order,
            assurance_mode=request.assurance_mode,
            execution_mode=request.execution_mode,
        )

        edited_merged_yaml = request.tmp_dir / "edited_merged.yaml"
        runtime.dump_yaml_fn(edited_merged_yaml, merged_edited_cfg)
        runtime.debug_fn(f"Edited config (merged): {edited_merged_yaml}")

        with runtime.suppress_child_output_fn(runtime.verbosity == 0) as quiet_buffer:
            try:
                with cli_output.timed_step(
                    console=runtime.console,
                    style=runtime.output_style,
                    timings=runtime.timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="✂️",
                ):
                    edited_run_result = runtime.run_command_fn(
                        config=str(edited_merged_yaml),
                        profile=request.profile_name,
                        out=str(Path(request.out) / "edited"),
                        tier=request.tier_name,
                        baseline=baseline_report_str,
                        device=request.device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=(
                            request.subject_label if request.edit_label else None
                        ),
                        style=runtime.output_style.name,
                        progress=runtime.progress,
                        timing=False,
                        allow_network=request.allow_network,
                        allow_host_execution=request.allow_host_execution,
                        allow_third_party_plugins=request.allow_third_party_plugins,
                        allow_remote_code=request.allow_remote_code,
                        allow_unverified_provenance=request.allow_unverified_provenance,
                        prefer_local_files_only=request.prefer_local_files_only,
                        no_color=request.no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    runtime.console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except Exception:
                if quiet_buffer is not None:
                    runtime.console.print(quiet_buffer.getvalue(), markup=False)
                raise
    else:
        edited_cfg = build_subject_noop_run_config(
            request.preset_data,
            model_id=request.subject_model_id,
            adapter_name=str(request.adapter),
            output_dir=str(Path(request.out) / "edited"),
            profile=request.profile_name,
            tier=request.tier_name,
            guards_order=request.guards_order,
            assurance_mode=request.assurance_mode,
            execution_mode=request.execution_mode,
        )
        edited_yaml = request.tmp_dir / "edited_noop.yaml"
        runtime.dump_yaml_fn(edited_yaml, edited_cfg)
        runtime.info_fn(
            "Running edited (no-op, Compare & Evaluate)", tag="EXEC", emoji="🧪"
        )
        runtime.debug_fn(f"Edited config: {edited_yaml}")

        with runtime.suppress_child_output_fn(runtime.verbosity == 0) as quiet_buffer:
            try:
                with cli_output.timed_step(
                    console=runtime.console,
                    style=runtime.output_style,
                    timings=runtime.timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="🧪",
                ):
                    edited_run_result = runtime.run_command_fn(
                        config=str(edited_yaml),
                        profile=request.profile_name,
                        out=str(Path(request.out) / "edited"),
                        tier=request.tier_name,
                        baseline=baseline_report_str,
                        device=request.device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=request.subject_label,
                        style=runtime.output_style.name,
                        progress=runtime.progress,
                        timing=False,
                        allow_network=request.allow_network,
                        allow_host_execution=request.allow_host_execution,
                        allow_third_party_plugins=request.allow_third_party_plugins,
                        allow_remote_code=request.allow_remote_code,
                        allow_unverified_provenance=request.allow_unverified_provenance,
                        prefer_local_files_only=request.prefer_local_files_only,
                        no_color=request.no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    runtime.console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except Exception:
                if quiet_buffer is not None:
                    runtime.console.print(quiet_buffer.getvalue(), markup=False)
                raise

    try:
        edited_report = require_run_report_artifact(
            edited_run_result,
            stage="Edited",
        )
    except Exception as exc:
        runtime.fail_fn(str(getattr(exc, "message", exc)), exit_code=1)
    runtime.debug_fn(f"Edited report: {edited_report}")

    try:
        with Path(edited_report).open("r", encoding="utf-8") as fh:
            edited_payload = runtime.json_load_fn(fh)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        cli_output.print_event(
            runtime.console,
            "FAIL",
            f"Failed to read edited report: {exc}",
            style=runtime.output_style,
            emoji="❌",
        )
        raise typer.Exit(1) from exc
    if not isinstance(edited_payload, dict):
        runtime.fail_fn("Edited run returned a non-object report payload.", exit_code=1)

    edited_status = str(edited_payload.get("status") or "").strip().lower()
    if edited_status == "failed":
        failure_detail = edited_payload.get("error")
        failure_suffix = (
            f" {failure_detail}"
            if isinstance(failure_detail, str) and failure_detail.strip()
            else ""
        )
        runtime.fail_fn(
            f"Edited run failed before evaluation report generation.{failure_suffix}",
            exit_code=1,
        )

    return edited_report, edited_payload


__all__ = [
    "BaselineEvaluationRequest",
    "EvaluatePhaseRuntime",
    "SubjectEvaluationRequest",
    "run_baseline_evaluation_phase",
    "run_subject_evaluation_phase",
]
