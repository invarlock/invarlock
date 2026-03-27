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
import logging
import sys

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
from .bench_runner import (
    DependencyChecker,
    execute_scenario,
    execute_single_run,
    run_guard_effect_benchmark,
)

logger = logging.getLogger(__name__)

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
    "_config_to_dict",
    "_generate_step14_markdown",
    "_scenario_result_to_dict",
    "_summary_to_step14_json",
    "execute_scenario",
    "execute_single_run",
    "generate_scenarios",
    "main",
    "resolve_epsilon_from_runtime",
    "run_guard_effect_benchmark",
]


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
