#!/usr/bin/env python3
"""Run a small deterministic mutation smoke test for trust-critical checks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    original: str
    mutated: str
    killed_by: tuple[str, ...]


MUTANTS = (
    Mutant(
        name="bootstrap-paired-delta-reverses-subject-and-baseline",
        path="src/invarlock/core/bootstrap.py",
        original="delta = final_arr - base_arr",
        mutated="delta = base_arr - final_arr",
        killed_by=(
            "tests/core/test_bootstrap.py::"
            "test_compute_paired_delta_log_ci_weighted_percentile_matches_seeded_oracle",
        ),
    ),
    Mutant(
        name="bootstrap-ratio-ci-reverses-bound-order",
        path="src/invarlock/core/bootstrap.py",
        original="return math.exp(lo), math.exp(hi)",
        mutated="return math.exp(hi), math.exp(lo)",
        killed_by=(
            "tests/core/test_bootstrap.py::"
            "test_compute_paired_delta_and_ratio_ci_consistency",
        ),
    ),
    Mutant(
        name="rmt-mp-bulk-edge-inverts-aspect-ratio",
        path="src/invarlock/guards/rmt_analysis.py",
        original="q = n / m",
        mutated="q = m / n",
        killed_by=(
            "tests/guards/rmt/test_rmt_activation_helpers.py::"
            "test_rmt_activation_edge_risk_matches_synthetic_mp_oracle",
        ),
    ),
    Mutant(
        name="rmt-deadband-boundary-becomes-exclusive",
        path="src/invarlock/guards/rmt_analysis.py",
        original="return sigma_cur <= (1.0 + deadband) * sigma_base",
        mutated="return sigma_cur < (1.0 + deadband) * sigma_base",
        killed_by=(
            "tests/guards/rmt/test_rmt_branch_small.py::"
            "test_growth_and_deadband_boundaries",
        ),
    ),
    Mutant(
        name="rmt-growth-ratio-inverts-current-and-baseline",
        path="src/invarlock/guards/rmt_analysis.py",
        original="return r_cur / max(r_base, 1e-12)",
        mutated="return r_base / max(r_cur, 1e-12)",
        killed_by=(
            "tests/guards/rmt/test_rmt_utils.py::test_growth_ratio_and_deadband",
        ),
    ),
    Mutant(
        name="spectral-z-score-reverses-sigma-and-mean",
        path="src/invarlock/guards/spectral_detection.py",
        original="return float((sigma - mean) / std)",
        mutated="return float((mean - sigma) / std)",
        killed_by=(
            "tests/guards/spectral/test_spectral_guard_paths.py::"
            "test_compute_z_score_for_value_with_std",
        ),
    ),
    Mutant(
        name="spectral-zero-std-deadband-ignores-scale",
        path="src/invarlock/guards/spectral_detection.py",
        original="scale = deadband if deadband > 0 else 1.0",
        mutated="scale = 1.0 if deadband > 0 else 1.0",
        killed_by=(
            "tests/guards/spectral/test_spectral_guard_paths.py::"
            "test_compute_z_score_for_value_deadband_zero_std",
        ),
    ),
    Mutant(
        name="spectral-family-z-summary-ignores-negative-outliers",
        path="src/invarlock/guards/spectral_detection.py",
        original="np.sum(np.abs(arr) > float(cap))",
        mutated="np.sum(arr > float(cap))",
        killed_by=(
            "tests/guards/spectral/test_spectral_multiple_testing_enforcement.py::"
            "test_spectral_negative_z_decision_and_summary_match_production_oracle",
        ),
    ),
    Mutant(
        name="variance-calibration-truncation-is-disabled",
        path="src/invarlock/guards/variance_batching.py",
        original="return value[..., :max_seq_len].clone(), True, seq_len",
        mutated="return value, False, seq_len",
        killed_by=(
            "tests/guards/variance/test_variance_tensorize_and_ensure_utils.py::"
            "test_tensorize_calibration_batches_honors_max_seq_len",
        ),
    ),
    Mutant(
        name="variance-window-id-fallback-drops-last-batch",
        path="src/invarlock/guards/variance_batching.py",
        original="window_ids = [str(index) for index in range(len(batches))]",
        mutated="window_ids = [str(index) for index in range(len(batches) - 1)]",
        killed_by=(
            "tests/guards/variance/test_variance_extract_window_ids_default.py::"
            "test_extract_window_ids_defaults_to_index_when_missing",
        ),
    ),
    Mutant(
        name="variance-two-sided-ci-zero-boundary-becomes-exclusive",
        path="src/invarlock/guards/variance_policy.py",
        original="if lower <= 0.0 <= upper:",
        mutated="if lower < 0.0 < upper:",
        killed_by=(
            "tests/guards/variance/test_variance_predictive_gate_outcome.py::"
            "test_predictive_gate_outcome_two_sided_zero_in_ci_fails",
        ),
    ),
    Mutant(
        name="variance-finalize-gain-threshold-comparison-inverts",
        path="src/invarlock/guards/variance_results.py",
        original="if ab_gain < required_gain_with_deadband:",
        mutated="if ab_gain > required_gain_with_deadband:",
        killed_by=(
            "tests/guards/variance/test_variance_results.py::"
            "test_evaluate_finalize_state_collects_errors_and_warnings",
        ),
    ),
)


def _apply_mutant(repo: Path, worktree: Path, mutant: Mutant) -> None:
    source = worktree / mutant.path
    text = source.read_text(encoding="utf-8")
    if mutant.original not in text:
        raise RuntimeError(
            f"{mutant.name}: mutation anchor not found in {repo / mutant.path}"
        )
    source.write_text(
        text.replace(mutant.original, mutant.mutated, 1), encoding="utf-8"
    )


def _run_pytest(
    repo: Path, worktree: Path, tests: tuple[str, ...]
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_mutation_smoke(repo: Path) -> int:
    failures: list[str] = []
    for mutant in MUTANTS:
        with tempfile.TemporaryDirectory(prefix="invarlock-mutant-") as tmp:
            worktree = Path(tmp)
            shutil.copytree(repo / "src", worktree / "src")
            _apply_mutant(repo, worktree, mutant)
            result = _run_pytest(repo, worktree, mutant.killed_by)
            if result.returncode == 0:
                failures.append(f"{mutant.name}: survived")
                print(result.stdout)
            else:
                print(f"{mutant.name}: killed")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    args = parser.parse_args(argv)
    return run_mutation_smoke(args.repo.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
