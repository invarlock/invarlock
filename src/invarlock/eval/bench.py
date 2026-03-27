"""
InvarLock Guard Effect Benchmark - Step 14 Implementation
=====================================================

Benchmark harness for comparing "bare" vs "guarded" runs across different edit types,
tiers, and probes configurations. Provides comprehensive analysis of guard effectiveness
and overhead with precise validation gates.

Usage:
    python -m invarlock.eval.bench --edits quant_rtn --tiers balanced --probes 0,2,4 --profile ci

Key Features:
- Edit × Tier × Probes scenario grid
- Paired runs (bare vs guarded) with identical windows
- Comprehensive metrics with validation gates
- Support for CI (50/50) and Release (100/100) profiles
- Optional dependency checking (e.g., GPTQ)
- JSON artifacts and Markdown summary tables
- Exit non-zero on any gate failure
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Import InvarLock components
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
    generate_scenarios,
    resolve_epsilon_from_runtime,
)
from .bench_policy import (
    config_to_dict as _config_to_dict,
)
from .bench_policy import (
    generate_step14_markdown as _generate_step14_markdown,
)
from .bench_policy import (
    scenario_result_to_dict as _scenario_result_to_dict,
)
from .bench_policy import (
    summary_to_step14_json as _summary_to_step14_json,
)

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check for optional dependencies required by specific edit types."""

    @staticmethod
    def check_external_deps() -> tuple[bool, str]:
        """Check if external dependencies are available (placeholder for future use)."""
        # Placeholder for when external edit plugins are re-enabled
        return True, "Available"

    @staticmethod
    def check_peft() -> tuple[bool, str]:
        """Deprecated: external fine-tuning adapters are not supported in this profile."""
        return False, "unsupported edit"

    @classmethod
    def check_edit_dependencies(cls, edit_name: str) -> tuple[bool, str]:
        """Check dependencies for a specific edit type."""
        # Only quant_rtn is supported
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
    """Execute a single benchmark run and return results."""
    try:
        # Deferred imports: heavy deps only when executing real pipeline
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

        # Resolve shared runtime resources (tokenizer/windows/model snapshot) when absent.
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

        # Tokenizer + pairing schedule
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

        # Adapter/model snapshot
        if adapter is None or model is None or baseline_snapshot is None:
            registry = _get_registry()
            adapter = registry.get_adapter(scenario.adapter)
            model = adapter.load_model(scenario.model_id, device=scenario.device)
            baseline_snapshot = adapter.snapshot(model)
            runtime["adapter"] = adapter
            runtime["model"] = model
            runtime["baseline_snapshot"] = baseline_snapshot

        # Baseline RMT stats (used to compute comparable outlier counts for bare vs guarded)
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

        # Restore baseline model for this run
        adapter.restore(model, baseline_snapshot)

        run_dir = output_dir / run_type
        _ensure_dir(run_dir)
        event_path = run_dir / "events.jsonl"

        # Core objects
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

        # Wire run context for pairing verification
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

        # Convert to evaluation RunReport (dict) for downstream tooling
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

        # Transfer metrics
        if hasattr(core_report, "metrics") and isinstance(core_report.metrics, dict):
            report["metrics"].update(core_report.metrics)

        if hasattr(core_report, "evaluation_windows") and isinstance(
            core_report.evaluation_windows, dict
        ):
            report["evaluation_windows"] = core_report.evaluation_windows

        # Transfer guards
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

        # Compute comparable RMT outliers for both bare and guarded models
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

        # Flags and artifacts
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

    except Exception as e:
        logger.error(f"Run failed for {scenario.edit} ({run_type}): {e}")
        return RunResult(
            run_type=run_type,
            report=create_empty_report(),
            success=False,
            error_message=str(e),
        )


def execute_scenario(
    scenario: ScenarioConfig, config: BenchmarkConfig, output_dir: Path
) -> ScenarioResult:
    """Execute a single benchmark scenario (both bare and guarded runs)."""
    logger.info(
        f"Executing scenario: {scenario.edit} × {scenario.tier} × {scenario.probes} probes"
    )

    # Check dependencies
    deps_available, deps_message = DependencyChecker.check_edit_dependencies(
        scenario.edit
    )
    if not deps_available:
        logger.warning(f"Skipping scenario: {deps_message}")
        return ScenarioResult(config=scenario, skipped=True, skip_reason=deps_message)

    config_manager = ConfigurationManager()
    metrics_aggregator = MetricsAggregator()

    # Scenario-scoped artifact directory
    scenario_slug = f"{scenario.edit}__{scenario.tier}__p{scenario.probes}"
    scenario_dir = output_dir / "scenarios" / scenario_slug
    scenario_dir.mkdir(parents=True, exist_ok=True)

    runtime: dict[str, Any] = {"dataset_name": config.dataset}

    # Run bare configuration
    logger.debug(f"Running bare configuration for {scenario.edit}")
    bare_config = config_manager.create_bare_config(scenario)
    try:
        bare_config.setdefault("dataset", {})["provider"] = config.dataset
    except Exception:
        pass
    bare_result = execute_single_run(
        bare_config, scenario, "bare", scenario_dir, runtime=runtime
    )

    # Run guarded configuration
    logger.debug(f"Running guarded configuration for {scenario.edit}")
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

    # Generate evaluation report artifact when both runs produced reports
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
                    print(summary_line)
            report_path = scenario_dir / "evaluation.report.json"
            report_path.write_text(
                json.dumps(evaluation_report, indent=2), encoding="utf-8"
            )
            artifacts["evaluation_report"] = str(report_path)
    except Exception as exc:
        logger.warning(
            f"Evaluation report generation failed for {scenario_slug}: {exc}"
        )

    # Resolve epsilon from runtime or use config
    epsilon_used = config.epsilon
    if epsilon_used is None and guarded_result.success:
        epsilon_used = resolve_epsilon_from_runtime(guarded_result.report)
    elif epsilon_used is None:
        epsilon_used = 0.10  # Default fallback

    # Compute comparison metrics and validate gates.
    comparison_metrics = metrics_aggregator.compute_comparison_metrics(
        bare_result, guarded_result
    )
    if not (bare_result.success and guarded_result.success):
        # Treat execution failures as a hard FAIL: benchmarks are only meaningful
        # when both paired runs complete.
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

    # Mock probes_used based on scenario.probes (in real implementation, this would come from auto-tuner)
    probes_used = min(
        scenario.probes, scenario.probes
    )  # All requested probes used in mock

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


def run_guard_effect_benchmark(
    edits: list[str],
    tiers: list[str],
    probes: list[int],
    profile: str = "ci",
    output_dir: str | Path = "benchmarks",
    epsilon: float | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Run guard effect benchmark across edit × tier × probes scenarios.

    Args:
        edits: List of edit types to benchmark
        tiers: List of tier configurations
        probes: List of probe counts
        profile: "ci" (50/50 windows) or "release" (100/100 windows)
        output_dir: Directory to save results
        epsilon: Optional epsilon override
        **kwargs: Additional configuration options

    Returns:
        Dictionary with benchmark results and summary

    Raises:
        SystemExit: If any gates fail (non-zero exit code)
    """
    start_time = datetime.now()

    # Create configuration
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

    # Generate scenarios
    scenarios = generate_scenarios(config)
    scenario_results = []

    # Execute each scenario
    for scenario in scenarios:
        result = execute_scenario(scenario, config, config.output_dir)
        scenario_results.append(result)

    # Create summary
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    # Check overall pass/fail - any non-skipped scenario with failed gates = overall fail
    overall_pass = True
    for result in scenario_results:
        if not result.skipped and result.gates:
            if not all(result.gates.values()):
                overall_pass = False
                break

    summary = BenchmarkSummary(
        config=config,
        scenarios=scenario_results,
        overall_pass=overall_pass,
        timestamp=start_time.isoformat(),
        execution_time_seconds=execution_time,
    )

    # Generate outputs
    _generate_outputs(summary)

    logger.info(f"Benchmark completed in {execution_time:.1f}s")
    logger.info(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")

    # Return results as dictionary
    result = {
        "overall_pass": overall_pass,
        "execution_time_seconds": execution_time,
        "timestamp": start_time.isoformat(),
        "scenarios": [_scenario_result_to_dict(result) for result in scenario_results],
        "config": _config_to_dict(config),
    }

    return result


def _generate_outputs(summary: BenchmarkSummary) -> None:
    """Generate JSON and Markdown outputs according to Step 14 specification."""
    results_dir = summary.config.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate JSON artifact
    json_path = results_dir / "guard_effect.json"
    json_data = _summary_to_step14_json(summary)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"JSON artifact saved: {json_path}")

    # Generate Markdown summary
    md_path = results_dir / "guard_effect.md"
    with open(md_path, "w") as f:
        f.write(_generate_step14_markdown(summary))

    logger.info(f"Markdown report saved: {md_path}")


def main():
    """CLI entry point for Step 14 specification."""
    parser = argparse.ArgumentParser(
        description="InvarLock Guard Effect Benchmark - Step 14",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--edits",
        required=True,
        help="Comma-separated list of edit types (quant_rtn)",
    )
    parser.add_argument(
        "--tiers",
        default="balanced",
        help="Comma-separated list of tiers (conservative,balanced,aggressive)",
    )
    parser.add_argument(
        "--probes", default="0", help="Comma-separated list of probe counts (0,2,4)"
    )
    parser.add_argument(
        "--profile",
        default="ci",
        choices=["ci", "release"],
        help="Benchmark profile (ci=50/50 windows, release=100/100 windows)",
    )

    # Optional threshold configuration
    parser.add_argument(
        "--epsilon",
        type=float,
        help="RMT outliers epsilon threshold (default: use resolved RMT deadband)",
    )

    # Model and dataset configuration
    parser.add_argument(
        "--dataset", default="wikitext2", help="Dataset to use for benchmarking"
    )
    parser.add_argument("--model-id", default="gpt2", help="Model identifier")
    parser.add_argument("--adapter", default="hf_causal", help="Model adapter to use")
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto|cuda|mps|cpu)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length for tokenization"
    )
    parser.add_argument(
        "--stride", type=int, default=128, help="Stride for window generation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="benchmarks", help="Output directory")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse lists
    edits = [edit.strip() for edit in args.edits.split(",")]
    tiers = [tier.strip() for tier in args.tiers.split(",")]
    probes = [int(probe.strip()) for probe in args.probes.split(",")]

    # Validate inputs — only quant_rtn is supported
    valid_edits = {"quant_rtn"}
    valid_tiers = {"conservative", "balanced", "aggressive"}

    for edit in edits:
        if edit not in valid_edits:
            print(
                f"❌ Invalid edit type: {edit}. Valid: {', '.join(sorted(valid_edits))}"
            )
            sys.exit(1)

    for tier in tiers:
        if tier not in valid_tiers:
            print(f"❌ Invalid tier: {tier}. Valid: {', '.join(sorted(valid_tiers))}")
            sys.exit(1)

    for probe in probes:
        if probe < 0:
            print(f"❌ Invalid probe count: {probe}. Must be >= 0")
            sys.exit(1)

    # Prepare kwargs
    kwargs = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "adapter": args.adapter,
        "device": args.device,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "seed": args.seed,
    }

    try:
        # Run benchmark
        result = run_guard_effect_benchmark(
            edits=edits,
            tiers=tiers,
            probes=probes,
            profile=args.profile,
            output_dir=args.out,
            epsilon=args.epsilon,
            **kwargs,
        )

        # Exit with appropriate code per Step 14 specification
        if result["overall_pass"]:
            print("✅ All gates passed!")
            sys.exit(0)
        else:
            print("❌ Some gates failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
