from __future__ import annotations

import subprocess
import sys


def test_claim_surface_consistency_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/checks/check_claim_surface_consistency.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
