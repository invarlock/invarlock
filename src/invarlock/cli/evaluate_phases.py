"""Baseline/subject evaluation phases for the evaluate CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from invarlock.cli import output as cli_output
from invarlock.core.evaluate_contract import (
    load_validated_baseline_report,
    require_run_report_artifact,
)


def _run_baseline_evaluation_phase(
    *,
    baseline_report: str | None,
    profile_name: str,
    tier_name: str,
    eff_adapter: str,
    out: str,
    device: str | None,
    allow_network: bool,
    allow_host_execution: bool,
    allow_third_party_plugins: bool,
    allow_remote_code: bool,
    allow_unverified_provenance: bool,
    prefer_local_files_only: bool,
    no_color: bool,
    baseline_cfg: dict[str, Any],
    baseline_label: str,
    tmp_dir: Path,
    console: Any,
    output_style: Any,
    timings: dict[str, float],
    verbosity: int,
    progress: bool,
    info_fn: Any,
    debug_fn: Any,
    phase_fn: Any,
    fail_fn: Any,
    suppress_child_output_fn: Any,
    dump_yaml_fn: Any,
    run_command_fn: Any,
) -> Path:
    if baseline_report:
        info_fn(
            "Using provided baseline report (skipping baseline evaluation)",
            tag="EXEC",
            emoji="♻️",
        )
        try:
            model_cfg = baseline_cfg.get("model")
            baseline_model_id = (
                model_cfg.get("id") if isinstance(model_cfg, dict) else ""
            )
            assurance_cfg = baseline_cfg.get("assurance")
            assurance_mode = (
                assurance_cfg.get("mode") if isinstance(assurance_cfg, dict) else None
            )
            dataset_cfg = baseline_cfg.get("dataset")
            baseline_report_path, _ = load_validated_baseline_report(
                Path(baseline_report),
                expected_model_id=str(baseline_model_id or ""),
                expected_profile=profile_name,
                expected_tier=tier_name,
                expected_adapter=str(eff_adapter),
                expected_assurance_mode=str(assurance_mode or "off"),
                expected_dataset=dataset_cfg if isinstance(dataset_cfg, dict) else None,
            )
        except typer.BadParameter as exc:
            fail_fn(str(getattr(exc, "message", exc)), exit_code=2)
        except Exception as exc:  # noqa: BLE001 - preserve existing failure surface
            if isinstance(exc, (typer.Exit, SystemExit)):
                raise
            fail_fn(str(getattr(exc, "message", exc)), exit_code=2)
        debug_fn(f"Baseline report: {baseline_report_path}")
        return baseline_report_path

    baseline_yaml = tmp_dir / "baseline_noop.yaml"
    dump_yaml_fn(baseline_yaml, baseline_cfg)

    phase_fn(1, 3, "BASELINE EVALUATION")
    info_fn("Running baseline (no-op edit)", tag="EXEC", emoji="🏁")
    debug_fn(f"Baseline config: {baseline_yaml}")

    with suppress_child_output_fn(verbosity == 0) as quiet_buffer:
        try:
            with cli_output.timed_step(
                console=console,
                style=output_style,
                timings=timings,
                key="baseline",
                tag="EXEC",
                message="Baseline",
                emoji="🏁",
            ):
                baseline_run_result = run_command_fn(
                    config=str(baseline_yaml),
                    profile=profile_name,
                    out=str(Path(out) / "source"),
                    tier=tier_name,
                    device=device,
                    until_pass=False,
                    max_attempts=1,
                    timeout=None,
                    edit_label=baseline_label,
                    style=output_style.name,
                    progress=progress,
                    timing=False,
                    allow_network=allow_network,
                    allow_host_execution=allow_host_execution,
                    allow_third_party_plugins=allow_third_party_plugins,
                    allow_remote_code=allow_remote_code,
                    allow_unverified_provenance=allow_unverified_provenance,
                    prefer_local_files_only=prefer_local_files_only,
                    no_color=no_color,
                )
        except typer.Exit:
            if quiet_buffer is not None:
                console.print(quiet_buffer.getvalue(), markup=False)
            raise
        except Exception:
            if quiet_buffer is not None:
                console.print(quiet_buffer.getvalue(), markup=False)
            raise

    try:
        baseline_report_path = require_run_report_artifact(
            baseline_run_result,
            stage="Baseline",
        )
    except Exception as exc:
        fail_fn(str(getattr(exc, "message", exc)), exit_code=1)
    debug_fn(f"Baseline report: {baseline_report_path}")
    return baseline_report_path


def _run_subject_evaluation_phase(
    *,
    baseline_report_path: Path,
    preset_data: dict[str, Any],
    norm_edt_id: str,
    eff_adapter: str,
    out: str,
    device: str | None,
    profile_name: str,
    tier_name: str,
    guards_order: Any,
    assurance_mode: str,
    subject_label: str | None,
    edit_config: str | None,
    edit_label: str | None,
    console: Any,
    output_style: Any,
    timings: dict[str, float],
    verbosity: int,
    progress: bool,
    execution_mode: str,
    allow_network: bool,
    allow_host_execution: bool,
    allow_third_party_plugins: bool,
    allow_remote_code: bool,
    allow_unverified_provenance: bool,
    prefer_local_files_only: bool,
    no_color: bool,
    tmp_dir: Path,
    info_fn: Any,
    debug_fn: Any,
    phase_fn: Any,
    fail_fn: Any,
    load_yaml_fn: Any,
    dump_yaml_fn: Any,
    suppress_child_output_fn: Any,
    run_command_fn: Any,
    json_load_fn: Any,
) -> tuple[Path, dict[str, Any]]:
    from invarlock.core.evaluate_plan import (
        build_subject_edit_run_config,
        build_subject_noop_run_config,
    )

    phase_fn(2, 3, "SUBJECT EVALUATION")
    baseline_report_str = str(baseline_report_path)
    if edit_config:
        edited_yaml = Path(edit_config)
        if not edited_yaml.exists():
            cli_output.print_event(
                console,
                "FAIL",
                f"Edit config not found: {edited_yaml}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1)
        info_fn("Running edited (demo edit via --edit-config)", tag="EXEC", emoji="✂️")
        try:
            cfg_loaded: dict[str, Any] = load_yaml_fn(edited_yaml)
        except Exception as exc:
            cli_output.print_event(
                console,
                "FAIL",
                f"Failed to load edit config: {exc}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1) from exc

        merged_edited_cfg = build_subject_edit_run_config(
            preset_data,
            cfg_loaded,
            subject_model_id=norm_edt_id,
            adapter_name=str(eff_adapter),
            output_dir=str(Path(out) / "edited"),
            profile=profile_name,
            tier=tier_name,
            guards_order=guards_order,
            assurance_mode=assurance_mode,
            execution_mode=execution_mode,
        )

        edited_merged_yaml = tmp_dir / "edited_merged.yaml"
        dump_yaml_fn(edited_merged_yaml, merged_edited_cfg)
        debug_fn(f"Edited config (merged): {edited_merged_yaml}")

        with suppress_child_output_fn(verbosity == 0) as quiet_buffer:
            try:
                with cli_output.timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="✂️",
                ):
                    edited_run_result = run_command_fn(
                        config=str(edited_merged_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "edited"),
                        tier=tier_name,
                        baseline=baseline_report_str,
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=subject_label if edit_label else None,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        allow_unverified_provenance=allow_unverified_provenance,
                        prefer_local_files_only=prefer_local_files_only,
                        no_color=no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
    else:
        edited_cfg = build_subject_noop_run_config(
            preset_data,
            model_id=norm_edt_id,
            adapter_name=str(eff_adapter),
            output_dir=str(Path(out) / "edited"),
            profile=profile_name,
            tier=tier_name,
            guards_order=guards_order,
            assurance_mode=assurance_mode,
            execution_mode=execution_mode,
        )
        edited_yaml = tmp_dir / "edited_noop.yaml"
        dump_yaml_fn(edited_yaml, edited_cfg)
        info_fn("Running edited (no-op, Compare & Evaluate)", tag="EXEC", emoji="🧪")
        debug_fn(f"Edited config: {edited_yaml}")

        with suppress_child_output_fn(verbosity == 0) as quiet_buffer:
            try:
                with cli_output.timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="🧪",
                ):
                    edited_run_result = run_command_fn(
                        config=str(edited_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "edited"),
                        tier=tier_name,
                        baseline=baseline_report_str,
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=subject_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        allow_unverified_provenance=allow_unverified_provenance,
                        prefer_local_files_only=prefer_local_files_only,
                        no_color=no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

    try:
        edited_report = require_run_report_artifact(
            edited_run_result,
            stage="Edited",
        )
    except Exception as exc:
        fail_fn(str(getattr(exc, "message", exc)), exit_code=1)
    debug_fn(f"Edited report: {edited_report}")

    try:
        with Path(edited_report).open("r", encoding="utf-8") as fh:
            edited_payload = json_load_fn(fh)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        cli_output.print_event(
            console,
            "FAIL",
            f"Failed to read edited report: {exc}",
            style=output_style,
            emoji="❌",
        )
        raise typer.Exit(1) from exc
    if not isinstance(edited_payload, dict):
        fail_fn("Edited run returned a non-object report payload.", exit_code=1)

    edited_status = str(edited_payload.get("status") or "").strip().lower()
    if edited_status == "failed":
        failure_detail = edited_payload.get("error")
        failure_suffix = (
            f" {failure_detail}"
            if isinstance(failure_detail, str) and failure_detail.strip()
            else ""
        )
        fail_fn(
            f"Edited run failed before evaluation report generation.{failure_suffix}",
            exit_code=1,
        )

    return edited_report, edited_payload
