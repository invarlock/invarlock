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
        name="spectral-family-z-summary-ignores-negative-outliers",
        path="src/invarlock/guards/spectral_detection.py",
        original="np.sum(np.abs(arr) > float(cap))",
        mutated="np.sum(arr > float(cap))",
        killed_by=(
            "tests/guards/spectral/test_spectral_multiple_testing_enforcement.py::"
            "test_spectral_negative_z_decision_and_summary_match_production_oracle",
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
