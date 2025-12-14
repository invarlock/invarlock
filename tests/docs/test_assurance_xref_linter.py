from __future__ import annotations

import subprocess
import sys


def test_assurance_cross_reference_linter_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/lint_assurance_xrefs.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
