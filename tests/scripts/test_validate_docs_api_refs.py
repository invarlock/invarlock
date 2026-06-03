from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_docs_api_refs_script_exits_zero(project_root: Path | None = None) -> None:
    """The docs API reference validator should succeed on current docs.

    This acts as a guardrail to keep examples in sync with the public API.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "docs" / "docs_check.py"
    assert script.exists(), "docs check script is missing"
    proc = subprocess.run(
        [sys.executable, str(script), "--api-refs"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
