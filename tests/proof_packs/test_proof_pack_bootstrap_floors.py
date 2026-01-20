from __future__ import annotations

import subprocess
from pathlib import Path

from invarlock.core.runner import BOOTSTRAP_COVERAGE_REQUIREMENTS


def _bash(repo_root: Path, script: str) -> str:
    result = subprocess.run(
        ["bash", "-c", f"set -e\n{script}\n"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    return result.stdout.strip()


def test_proof_pack_bootstrap_floor_matches_core_runner() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for tier, requirements in BOOTSTRAP_COVERAGE_REQUIREMENTS.items():
        expected = int(requirements["replicates"])
        observed = int(
            _bash(
                repo_root,
                (
                    "source scripts/proof_packs/lib/task_functions.sh\n"
                    f"_bootstrap_replicates_floor_for_tier {tier}\n"
                ),
            )
        )
        assert observed == expected


def test_proof_pack_bootstrap_resolver_enforces_floor_for_large_models() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    floor = int(BOOTSTRAP_COVERAGE_REQUIREMENTS["balanced"]["replicates"])
    observed_default = int(
        _bash(
            repo_root,
            (
                "source scripts/proof_packs/lib/task_functions.sh\n"
                "unset INVARLOCK_BOOTSTRAP_N\n"
                "_resolve_bootstrap_replicates 30 balanced\n"
            ),
        )
    )
    assert observed_default >= floor

    observed_override = int(
        _bash(
            repo_root,
            (
                "source scripts/proof_packs/lib/task_functions.sh\n"
                "INVARLOCK_BOOTSTRAP_N=1000 _resolve_bootstrap_replicates 30 balanced\n"
            ),
        )
    )
    assert observed_override >= floor
