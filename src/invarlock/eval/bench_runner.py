from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from invarlock.reporting.report_types import create_empty_report

from .bench_policy import (
    BenchmarkConfig,
    BenchmarkSummary,
    ConfigurationManager,
    MetricsAggregator,
    RunResult,
    ScenarioConfig,
    ScenarioResult,
    ValidationGates,
    config_to_dict,
    generate_scenarios,
    generate_step14_markdown,
    resolve_epsilon_from_runtime,
    scenario_result_to_dict,
    summary_to_step14_json,
)

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check for optional dependencies required by specific edit types."""

    @staticmethod
    def check_external_deps() -> tuple[bool, str]:
        return True, "Available"

    @staticmethod
    def check_peft() -> tuple[bool, str]:
        return False, "unsupported edit"

    @classmethod
    def check_edit_dependencies(cls, edit_name: str) -> tuple[bool, str]:
        if edit_name.lower() == "quant_rtn":
            return True, "Available"
        return False, "unsupported edit"


def execute_single_run(
    run_config: dict[str, Any],
    scenario: ScenarioConfig,
    run_type: str,
    output_dir: Path,
    *,
    runtime: dict[str, Any] | None = None,
) -> RunResult:
    try:
        from invarlock.core.api import RunConfig as _RunConfig
        from invarlock.core.auto_tuning import get_tier_policies as _get_tier_policies
        from invarlock.core.registry import get_registry as _get_registry
        from invarlock.core.runner import CoreRunner as _CoreRunner
        from invarlock.eval.data import get_provider as _get_provider
        from invarlock.guards.rmt_legacy import (
            capture_baseline_mp_stats as _capture_mp_stats,
        )
        from invarlock.guards.rmt_legacy import rmt_detect as _rmt_detect
        from invarlock.model_profile import detect_model_profile as _detect_profile

        def _ensure_dir(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

        def _write_json(path: Path, payload: dict[str, Any]) -> None:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if runtime is None:
            runtime = {}

        adapter = runtime.get("adapter")
        model = runtime.get("model")
        baseline_snapshot = runtime.get("baseline_snapshot")
        pairing_schedule = runtime.get("pairing_schedule")
        calibration_data = runtime.get("calibration_data")
        tokenizer_hash = runtime.get("tokenizer_hash")
        split = runtime.get("split", "validation")
        dataset_name = runtime.get("dataset_name")

        if not isinstance(dataset_name, str) or not dataset_name:
            dataset_name = str(
                run_config.get("dataset", {}).get("provider", "wikitext2")
            )

        if not (
            isinstance(pairing_schedule, dict) and isinstance(calibration_data, list)
        ):
            profile = _detect_profile(scenario.model_id, adapter=scenario.adapter)
            tokenizer, tokenizer_hash = profile.make_tokenizer()
            provider_kwargs: dict[str, Any] = {}
            if scenario.device != "auto" and dataset_name == "wikitext2":
                provider_kwargs["device_hint"] = str(scenario.device)
            provider = _get_provider(dataset_name, **provider_kwargs)
            preview_window, final_window = provider.windows(
                tokenizer=tokenizer,
                seq_len=scenario.seq_len,
                stride=scenario.stride,
                preview_n=scenario.preview_n or 0,
                final_n=scenario.final_n or 0,
                seed=scenario.seed,
                split=split,
            )
            prev_ids = list(range(len(preview_window.input_ids)))
            fin_ids = list(
                range(
                    len(preview_window.input_ids),
                    len(preview_window.input_ids) + len(final_window.input_ids),
                )
            )
            pairing_schedule = {
                "preview": {
                    "window_ids": prev_ids,
                    "input_ids": preview_window.input_ids,
                    "attention_masks": preview_window.attention_masks,
                },
                "final": {
                    "window_ids": fin_ids,
                    "input_ids": final_window.input_ids,
                    "attention_masks": final_window.attention_masks,
                },
            }
            calibration_data = []
            for idx, (input_ids, attention_mask) in enumerate(
                zip(
                    preview_window.input_ids,
                    preview_window.attention_masks,
                    strict=False,
                )
            ):
                calibration_data.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "window_id": f"preview::{idx}",
                    }
                )
            for idx, (input_ids, attention_mask) in enumerate(
                zip(final_window.input_ids, final_window.attention_masks, strict=False)
            ):
                calibration_data.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "window_id": f"final::{idx}",
                    }
                )
            runtime["pairing_schedule"] = pairing_schedule
            runtime["calibration_data"] = calibration_data
            runtime["tokenizer_hash"] = tokenizer_hash
            runtime["split"] = split
            runtime["dataset_name"] = dataset_name

        if adapter is None or model is None or baseline_snapshot is None:
            registry = _get_registry()
            adapter = registry.get_adapter(scenario.adapter)
            model = adapter.load_model(scenario.model_id, device=scenario.device)
            baseline_snapshot = adapter.snapshot(model)
            runtime["adapter"] = adapter
            runtime["model"] = model
            runtime["baseline_snapshot"] = baseline_snapshot

        rmt_baseline_mp_stats = runtime.get("rmt_baseline_mp_stats")
        rmt_baseline_sigmas = runtime.get("rmt_baseline_sigmas")
        if not isinstance(rmt_baseline_mp_stats, dict) or not isinstance(
            rmt_baseline_sigmas, dict
        ):
            adapter.restore(model, baseline_snapshot)
            rmt_baseline_mp_stats = _capture_mp_stats(model)
            rmt_baseline_sigmas = {
                name: float(stats.get("sigma_base", 0.0) or 0.0)
                for name, stats in rmt_baseline_mp_stats.items()
                if isinstance(stats, dict)
            }
            runtime["rmt_baseline_mp_stats"] = rmt_baseline_mp_stats
            runtime["rmt_baseline_sigmas"] = rmt_baseline_sigmas

        tier_policies = _get_tier_policies()
        tier_policy = tier_policies.get(
            scenario.tier, tier_policies.get("balanced", {})
        )
        rmt_policy = tier_policy.get("rmt", {}) if isinstance(tier_policy, dict) else {}
        rmt_margin = float(rmt_policy.get("margin", 1.5) or 1.5)
        rmt_deadband = float(rmt_policy.get("deadband", 0.10) or 0.10)

        adapter.restore(model, baseline_snapshot)

        run_dir = output_dir / run_type
        _ensure_dir(run_dir)
        event_path = run_dir / "events.jsonl"

        registry = _get_registry()
        edit_op = registry.get_edit(scenario.edit)

        guards: list[Any] = []
        auto_config = None
        if run_type == "guarded":
            for guard_name in ("invariants", "spectral", "rmt", "variance"):
                try:
                    guards.append(registry.get_guard(guard_name))
                except Exception:
                    continue
            auto_config = {
                "tier": scenario.tier,
                "probes": scenario.probes,
                "enabled": True,
            }

        run_context = {
            "profile": scenario.profile,
            "dataset": {"provider": dataset_name, "seed": scenario.seed},
            "pairing_baseline": pairing_schedule,
            "eval": {"loss": {"resolved_type": "causal"}},
            "run_id": f"{scenario.edit}-{scenario.tier}-p{scenario.probes}-{run_type}",
        }

        spike_threshold = float(
            run_config.get("eval", {}).get("spike_threshold", 2.0) or 2.0
        )
        cfg = _RunConfig(
            device=scenario.device,
            max_pm_ratio=spike_threshold,
            spike_threshold=spike_threshold,
            event_path=event_path,
            context=run_context,
        )

        runner = _CoreRunner()
        core_report = runner.execute(
            model=model,
            adapter=adapter,
            edit=edit_op,
            guards=guards,
            config=cfg,
            calibration_data=calibration_data,
            auto_config=auto_config,
            edit_config=run_config.get("edit", {}).get("plan", {}),
            preview_n=scenario.preview_n,
            final_n=scenario.final_n,
        )

        report = create_empty_report()
        report["meta"].update(
            {
                "model_id": scenario.model_id,
                "adapter": scenario.adapter,
                "device": str(scenario.device),
                "commit": "",
                "seed": scenario.seed,
                "ts": datetime.now().isoformat(),
            }
        )
        if tokenizer_hash:
            report["meta"]["tokenizer_hash"] = tokenizer_hash
        dur = core_report.meta.get("duration") if hasattr(core_report, "meta") else None
        if isinstance(dur, int | float):
            report["meta"]["duration_s"] = float(dur)

        report["data"].update(
            {
                "dataset": dataset_name,
                "split": split,
                "seq_len": scenario.seq_len,
                "stride": scenario.stride,
                "preview_n": int(scenario.preview_n or 0),
                "final_n": int(scenario.final_n or 0),
            }
        )

        edit_meta = core_report.edit if hasattr(core_report, "edit") else {}
        plan_digest = ""
        try:
            if isinstance(edit_meta, dict):
                plan_digest = str(edit_meta.get("plan_digest", ""))
        except Exception:
            plan_digest = ""
        report["edit"].update(
            {
                "name": scenario.edit,
                "plan_digest": plan_digest,
                "deltas": (
                    edit_meta.get("deltas", report["edit"]["deltas"])
                    if isinstance(edit_meta, dict)
                    else report["edit"]["deltas"]
                ),
            }
        )

        if hasattr(core_report, "metrics") and isinstance(core_report.metrics, dict):
            report["metrics"].update(core_report.metrics)

        if hasattr(core_report, "evaluation_windows") and isinstance(
            core_report.evaluation_windows, dict
        ):
            report["evaluation_windows"] = core_report.evaluation_windows

        if hasattr(core_report, "guards") and isinstance(core_report.guards, dict):
            for name, guard_result in core_report.guards.items():
                if not isinstance(guard_result, dict):
                    continue
                report["guards"].append(
                    {
                        "name": name,
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
                )

        try:
            detection = _rmt_detect(
                model=model,
                threshold=rmt_margin,
                detect_only=True,
                baseline_sigmas=rmt_baseline_sigmas,
                baseline_mp_stats=rmt_baseline_mp_stats,
                deadband=rmt_deadband,
            )
            report["metrics"].setdefault("rmt", {})
            if isinstance(report["metrics"].get("rmt"), dict):
                report["metrics"]["rmt"]["outliers"] = int(
                    detection.get("n_layers_flagged", 0) or 0
                )
        except Exception:
            pass

        status = getattr(core_report, "status", "")
        rollback_reason = (
            core_report.meta.get("rollback_reason")
            if hasattr(core_report, "meta") and isinstance(core_report.meta, dict)
            else None
        )
        report["flags"].update(
            {
                "guard_recovered": bool(
                    (
                        hasattr(core_report, "meta")
                        and core_report.meta.get("guard_recovered")
                    )
                    or str(status).lower() == "rollback"
                ),
                "rollback_reason": rollback_reason,
            }
        )
        report["artifacts"].update(
            {
                "events_path": str(event_path),
                "logs_path": "",
                "checkpoint_path": None,
                "report_path": str(run_dir / "report.json"),
            }
        )
        _write_json(Path(report["artifacts"]["report_path"]), report)

        success = str(status).lower() != "failed"
        return RunResult(run_type=run_type, report=report, success=success)
    except Exception as exc:
        logger.error(f"Run failed for {scenario.edit} ({run_type}): {exc}")
        return RunResult(
            run_type=run_type,
            report=create_empty_report(),
            success=False,
            error_message=str(exc),
        )


def execute_scenario(
    scenario: ScenarioConfig, config: BenchmarkConfig, output_dir: Path
) -> ScenarioResult:
    logger.info(
        f"Executing scenario: {scenario.edit} × {scenario.tier} × {scenario.probes} probes"
    )

    deps_available, deps_message = DependencyChecker.check_edit_dependencies(
        scenario.edit
    )
    if not deps_available:
        logger.warning(f"Skipping scenario: {deps_message}")
        return ScenarioResult(config=scenario, skipped=True, skip_reason=deps_message)

    config_manager = ConfigurationManager()
    metrics_aggregator = MetricsAggregator()

    scenario_slug = f"{scenario.edit}__{scenario.tier}__p{scenario.probes}"
    scenario_dir = output_dir / "scenarios" / scenario_slug
    scenario_dir.mkdir(parents=True, exist_ok=True)

    runtime: dict[str, Any] = {"dataset_name": config.dataset}

    bare_config = config_manager.create_bare_config(scenario)
    try:
        bare_config.setdefault("dataset", {})["provider"] = config.dataset
    except Exception:
        pass
    bare_result = execute_single_run(
        bare_config, scenario, "bare", scenario_dir, runtime=runtime
    )

    guarded_config = config_manager.create_guarded_config(scenario)
    try:
        guarded_config.setdefault("dataset", {})["provider"] = config.dataset
    except Exception:
        pass
    guarded_result = execute_single_run(
        guarded_config, scenario, "guarded", scenario_dir, runtime=runtime
    )

    artifacts: dict[str, Any] = {"scenario_dir": str(scenario_dir)}
    pairing_schedule = runtime.get("pairing_schedule")
    if isinstance(pairing_schedule, dict):
        pairing_path = scenario_dir / "pairing_schedule.json"
        pairing_path.write_text(
            json.dumps(pairing_schedule, indent=2), encoding="utf-8"
        )
        artifacts["pairing_schedule"] = str(pairing_path)
    try:
        if bare_result and bare_result.report:
            artifacts["bare_report"] = bare_result.report.get("artifacts", {}).get(
                "report_path"
            )
    except Exception:
        pass
    try:
        if guarded_result and guarded_result.report:
            artifacts["guarded_report"] = guarded_result.report.get(
                "artifacts", {}
            ).get("report_path")
    except Exception:
        pass

    try:
        if bare_result.success and guarded_result.success:
            from invarlock.reporting.report_builder import make_report
            from invarlock.reporting.report_telemetry import (
                telemetry_output_enabled,
                telemetry_summary_line,
            )

            evaluation_report = make_report(guarded_result.report, bare_result.report)
            if telemetry_output_enabled():
                summary_line = telemetry_summary_line(evaluation_report)
                if summary_line:
                    logger.info(summary_line)
            report_path = scenario_dir / "evaluation.report.json"
            report_path.write_text(
                json.dumps(evaluation_report, indent=2), encoding="utf-8"
            )
            artifacts["evaluation_report"] = str(report_path)
    except Exception as exc:
        logger.warning(
            f"Evaluation report generation failed for {scenario_slug}: {exc}"
        )

    epsilon_used = config.epsilon
    if epsilon_used is None and guarded_result.success:
        epsilon_used = resolve_epsilon_from_runtime(guarded_result.report)
    elif epsilon_used is None:
        epsilon_used = 0.10

    comparison_metrics = metrics_aggregator.compute_comparison_metrics(
        bare_result, guarded_result
    )
    if not (bare_result.success and guarded_result.success):
        comparison_metrics = {
            "error_bare": bare_result.error_message,
            "error_guarded": guarded_result.error_message,
        }
        gates = dict.fromkeys(
            ("spike", "tying", "rmt", "quality", "time", "mem"), False
        )
    else:
        gates = ValidationGates.validate_all_gates(
            comparison_metrics, config, epsilon_used
        )

    probes_used = min(scenario.probes, scenario.probes)

    return ScenarioResult(
        config=scenario,
        bare_result=bare_result,
        guarded_result=guarded_result,
        artifacts=artifacts,
        metrics=comparison_metrics,
        gates=gates,
        probes_used=probes_used,
        epsilon_used=epsilon_used,
    )


def generate_outputs(summary: BenchmarkSummary) -> None:
    results_dir = summary.config.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "guard_effect.json"
    json_data = summary_to_step14_json(summary)
    with open(json_path, "w") as handle:
        json.dump(json_data, handle, indent=2)
    logger.info(f"JSON artifact saved: {json_path}")

    md_path = results_dir / "guard_effect.md"
    with open(md_path, "w") as handle:
        handle.write(generate_step14_markdown(summary))
    logger.info(f"Markdown report saved: {md_path}")


def run_guard_effect_benchmark(
    edits: list[str],
    tiers: list[str],
    probes: list[int],
    profile: str = "ci",
    output_dir: str | Path = "benchmarks",
    epsilon: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    start_time = datetime.now()

    config = BenchmarkConfig(
        edits=edits,
        tiers=tiers,
        probes=probes,
        profile=profile,
        output_dir=Path(output_dir),
        epsilon=epsilon,
        **kwargs,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting guard effect benchmark with profile={profile}")
    logger.info(
        f"Scenario grid: {len(edits)} edits × {len(tiers)} tiers × {len(probes)} probes = {len(edits) * len(tiers) * len(probes)} scenarios"
    )
    logger.info(f"Output directory: {config.output_dir}")

    scenarios = generate_scenarios(config)
    scenario_results = [
        execute_scenario(scenario, config, config.output_dir) for scenario in scenarios
    ]

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    overall_pass = True
    for result in scenario_results:
        if not result.skipped and result.gates and not all(result.gates.values()):
            overall_pass = False
            break

    summary = BenchmarkSummary(
        config=config,
        scenarios=scenario_results,
        overall_pass=overall_pass,
        timestamp=start_time.isoformat(),
        execution_time_seconds=execution_time,
    )
    generate_outputs(summary)

    logger.info(f"Benchmark completed in {execution_time:.1f}s")
    logger.info(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "overall_pass": overall_pass,
        "execution_time_seconds": execution_time,
        "timestamp": start_time.isoformat(),
        "scenarios": [scenario_result_to_dict(result) for result in scenario_results],
        "config": config_to_dict(config),
    }


__all__ = [
    "DependencyChecker",
    "execute_scenario",
    "execute_single_run",
    "generate_outputs",
    "run_guard_effect_benchmark",
]
