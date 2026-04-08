"""
CLI entrypoint for the InvarLock guard-effect benchmark.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from invarlock.core.exceptions import InvarlockError
from invarlock.eval.bench import (
    BenchmarkConfig,
    BenchmarkSummary,
    ConfigurationManager,
    DependencyChecker,
    MetricsAggregator,
    RunResult,
    ScenarioConfig,
    ScenarioResult,
    ValidationGates,
    config_to_dict,
    execute_scenario,
    execute_single_run,
    generate_scenarios,
    generate_step14_markdown,
    resolve_epsilon_from_runtime,
    run_guard_effect_benchmark,
    scenario_result_to_dict,
    summary_to_step14_json,
)

_BENCH_COMMAND_ERRORS = (
    InvarlockError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkSummary",
    "ConfigurationManager",
    "DependencyChecker",
    "MetricsAggregator",
    "RunResult",
    "ScenarioConfig",
    "ScenarioResult",
    "ValidationGates",
    "config_to_dict",
    "generate_step14_markdown",
    "scenario_result_to_dict",
    "summary_to_step14_json",
    "execute_scenario",
    "execute_single_run",
    "generate_scenarios",
    "build_parser",
    "main",
    "resolve_epsilon_from_runtime",
    "run_guard_effect_benchmark",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InvarLock Guard Effect Benchmark - Step 14",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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

    parser.add_argument(
        "--epsilon",
        type=float,
        help="RMT outliers epsilon threshold (default: use resolved RMT deadband)",
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Step 14 specification."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    edits = [edit.strip() for edit in args.edits.split(",")]
    tiers = [tier.strip() for tier in args.tiers.split(",")]
    probes = [int(probe.strip()) for probe in args.probes.split(",")]

    valid_edits = {"quant_rtn"}
    valid_tiers = {"conservative", "balanced", "aggressive"}

    for edit in edits:
        if edit not in valid_edits:
            print(
                f"❌ Invalid edit type: {edit}. Valid: {', '.join(sorted(valid_edits))}"
            )
            return 1

    for tier in tiers:
        if tier not in valid_tiers:
            print(f"❌ Invalid tier: {tier}. Valid: {', '.join(sorted(valid_tiers))}")
            return 1

    for probe in probes:
        if probe < 0:
            print(f"❌ Invalid probe count: {probe}. Must be >= 0")
            return 1

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
        result = run_guard_effect_benchmark(
            edits=edits,
            tiers=tiers,
            probes=probes,
            profile=args.profile,
            output_dir=args.out,
            epsilon=args.epsilon,
            **kwargs,
        )

        if result["overall_pass"]:
            print("✅ All gates passed!")
            return 0
        print("❌ Some gates failed!")
        return 1
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        return 1
    except _BENCH_COMMAND_ERRORS as exc:
        print(f"❌ Benchmark failed: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
