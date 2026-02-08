#!/usr/bin/env python3
"""Pilot script to find RMT/Spectral-triggering parameters.

This script tests different parameters to find configurations where:
- For rmt_norm_noise: RMT trips while invariants/spectral stay stable
- For spectral_moderate_scale: Spectral trips while invariants/RMT stay stable

Usage:
    # RMT pilot
    python pilot_rmt_noise.py <baseline_path> --error-type rmt_norm_noise --scales 0.01,0.02,0.05,0.1

    # Spectral pilot
    python pilot_rmt_noise.py <baseline_path> --error-type spectral_moderate_scale --scales 2.0,3.0,5.0

Environment variables:
    CUDA_VISIBLE_DEVICES: GPU to use
    HF_HOME, HF_DATASETS_CACHE: HuggingFace cache directories
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _tail(path: Path, max_bytes: int = 16_384) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    if len(data) <= max_bytes:
        return data.decode(errors="replace")
    return data[-max_bytes:].decode(errors="replace")


def _run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return int(proc.wait())


def run_pilot(
    baseline_path: Path,
    param_value: float,
    output_dir: Path,
    error_type: str = "rmt_norm_noise",
    dataset: str = "hf_text",
    seed: int = 42,
) -> dict | None:
    """Run a single pilot test with given parameter value."""
    subject_dir = output_dir / f"subject_{error_type}_{param_value}"
    report_dir = output_dir / f"report_{error_type}_{param_value}"
    run_dir = output_dir / f"runs_{error_type}_{param_value}"
    log_dir = output_dir / f"logs_{error_type}_{param_value}"

    # Clean up previous runs
    for d in [subject_dir, report_dir, run_dir, log_dir]:
        if d.exists():
            shutil.rmtree(d)

    subject_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create error model
    print(f"\n{'=' * 60}", flush=True)
    print(f"Testing {error_type} param={param_value}, seed={seed}", flush=True)
    print(f"{'=' * 60}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Set appropriate environment variable based on error type
    if error_type == "rmt_norm_noise":
        env["INVARLOCK_RMT_NORM_NOISE_SCALE"] = str(param_value)
        env["INVARLOCK_RMT_NORM_SEED"] = str(seed)
        param_name = "noise_scale"
    elif error_type == "spectral_moderate_scale":
        env["INVARLOCK_SPECTRAL_SCALE_FACTOR"] = str(param_value)
        env["INVARLOCK_SPECTRAL_SEED"] = str(seed)
        param_name = "scale_factor"
    else:
        print(f"Unknown error type: {error_type}", flush=True)
        return None

    create_cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).parent / "create_error_model.py"),
        str(baseline_path),
        str(subject_dir),
        error_type,
    ]

    create_log = log_dir / "create_error_model.log"
    print(f"Creating error model: {param_name}={param_value}", flush=True)
    rc = _run_logged(create_cmd, env=env, log_path=create_log)
    if rc != 0:
        print(f"Error creating model (exit={rc}). Log tail:", flush=True)
        print(_tail(create_log), flush=True)
        return None

    # Read error metadata
    metadata_path = subject_dir / "error_metadata.json"
    if not metadata_path.exists():
        print("Error: error_metadata.json not created", flush=True)
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    if not metadata.get("injected"):
        print("Warning: No injection performed", flush=True)
        return None

    print(f"Modified {metadata.get('modified_count', 0)} modules/params", flush=True)

    # Run evaluation
    eval_env = env.copy()
    eval_env["INVARLOCK_ALLOW_NETWORK"] = "1"
    # Use larger windows for proper evaluation (CI mode too small for real models)
    eval_env.setdefault("INVARLOCK_CI_PREVIEW", "64")
    eval_env.setdefault("INVARLOCK_CI_FINAL", "64")

    if dataset == "hf_text":
        preset = "configs/presets/causal_lm/hf_text_c4_128.yaml"
    else:
        preset = "configs/presets/causal_lm/wikitext2_512.yaml"

    pilot_tier = os.environ.get("INVARLOCK_PILOT_TIER", "aggressive")
    eval_cmd = [
        "invarlock",
        "evaluate",
        "--baseline",
        str(baseline_path),
        "--subject",
        str(subject_dir),
        "--preset",
        preset,
        "--profile",
        "ci",
        "--tier",
        pilot_tier,
        "--report-out",
        str(report_dir),
        "--out",
        str(run_dir),
    ]

    eval_log = log_dir / "evaluate.log"
    print("Running evaluation...", flush=True)
    rc = _run_logged(eval_cmd, env=eval_env, log_path=eval_log)
    if rc != 0:
        print(f"Evaluation exited non-zero (exit={rc}). Log tail:", flush=True)
        print(_tail(eval_log), flush=True)

    # Parse report
    report_path = report_dir / "evaluation.report.json"
    if not report_path.exists():
        print("No report generated. Log tail:", flush=True)
        print(_tail(eval_log), flush=True)
        return None

    with open(report_path) as f:
        report = json.load(f)

    validation = report.get("validation", {})
    results = {
        "error_type": error_type,
        "param_name": param_name,
        "param_value": param_value,
        "seed": seed,
        "modified_count": metadata.get("modified_count", 0),
        "invariants_pass": validation.get("invariants_pass"),
        "spectral_stable": validation.get("spectral_stable"),
        "rmt_stable": validation.get("rmt_stable"),
        "primary_metric_acceptable": validation.get("primary_metric_acceptable"),
        "ppl_ratio": report.get("primary_metric", {}).get("ratio_vs_baseline"),
    }

    # Print summary
    print(f"\nResults for {param_name}={param_value}:", flush=True)
    print(f"  invariants_pass: {results['invariants_pass']}", flush=True)
    print(f"  spectral_stable: {results['spectral_stable']}", flush=True)
    print(f"  rmt_stable: {results['rmt_stable']}", flush=True)
    print(
        f"  primary_metric_acceptable: {results['primary_metric_acceptable']}",
        flush=True,
    )
    print(f"  PPL ratio: {results['ppl_ratio']}", flush=True)

    # Check success criteria based on error type
    if error_type == "rmt_norm_noise":
        # RMT should trip, others should be stable
        success = (
            results["invariants_pass"] is True
            and results["spectral_stable"] is True
            and results["rmt_stable"] is False
        )
        target_desc = "RMT trips, invariants/spectral stable"
    elif error_type == "spectral_moderate_scale":
        # Spectral should trip, others should be stable
        success = (
            results["invariants_pass"] is True
            and results["spectral_stable"] is False
            and results["rmt_stable"] is True
        )
        target_desc = "Spectral trips, invariants/RMT stable"
    else:
        success = False
        target_desc = "unknown"

    results["success"] = success
    results["target_desc"] = target_desc

    if success:
        print(f"  ✓ SUCCESS: {target_desc}!", flush=True)
    else:
        print(f"  ✗ Not ideal configuration (target: {target_desc})", flush=True)

    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # py3.7+
    parser = argparse.ArgumentParser(description="Pilot RMT/Spectral parameters")
    parser.add_argument("baseline_path", type=Path, help="Path to baseline model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/pilot"),
        help="Output directory",
    )
    parser.add_argument(
        "--error-type",
        choices=["rmt_norm_noise", "spectral_moderate_scale"],
        default="rmt_norm_noise",
        help="Error type to pilot",
    )
    parser.add_argument(
        "--dataset",
        choices=["hf_text", "wikitext2"],
        default="hf_text",
        help="Dataset to use",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=None,
        help="Comma-separated parameter values to test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Default scales based on error type
    if args.scales is None:
        if args.error_type == "rmt_norm_noise":
            args.scales = "0.01,0.02,0.05,0.1,0.2,0.3"
        else:
            args.scales = "2.0,3.0,5.0,7.0,10.0"

    scales = [float(s.strip()) for s in args.scales.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    successful_configs = []

    for scale in scales:
        result = run_pilot(
            args.baseline_path,
            scale,
            args.output_dir,
            error_type=args.error_type,
            dataset=args.dataset,
            seed=args.seed,
        )
        if result:
            all_results.append(result)
            if result.get("success"):
                successful_configs.append(result)

    # Save summary
    summary_path = args.output_dir / "pilot_summary.json"
    summary = {
        "baseline": str(args.baseline_path),
        "error_type": args.error_type,
        "dataset": args.dataset,
        "seed": args.seed,
        "scales_tested": scales,
        "results": all_results,
        "successful_configs": successful_configs,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PILOT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Error type: {args.error_type}")
    print(f"Tested {len(scales)} values: {scales}")
    print(f"Successful configs: {len(successful_configs)}")

    if successful_configs:
        print("\nBest configurations:")
        for cfg in successful_configs:
            print(
                f"  {cfg['param_name']}={cfg['param_value']}: "
                f"ppl_ratio={cfg.get('ppl_ratio', 'N/A')}"
            )
        # Recommend the smallest value that works
        best = min(successful_configs, key=lambda x: x["param_value"])
        print(f"\nRecommended: {best['param_name']}={best['param_value']}")
    else:
        print(
            f"\nNo configuration found that achieves: {all_results[0]['target_desc'] if all_results else 'target'}"
        )
        print("Consider trying different parameter ranges or target layers.")

    print(f"\nFull results saved to: {summary_path}")
    return 0 if successful_configs else 1


if __name__ == "__main__":
    raise SystemExit(main())
