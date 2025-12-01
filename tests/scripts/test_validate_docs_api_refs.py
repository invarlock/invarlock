from __future__ import annotations

import subprocess
from pathlib import Path


def test_docs_api_refs_script_exits_zero(project_root: Path | None = None) -> None:
    """The docs API reference validator should succeed on current docs.

    This acts as a guardrail to keep examples in sync with the public API.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "validate_docs_api_refs.py"
    assert script.exists(), "validation script is missing"
    proc = subprocess.run(["python", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
