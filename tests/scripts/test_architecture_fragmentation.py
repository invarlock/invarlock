from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_architecture_fragmentation.py"


def test_architecture_fragmentation_metrics_are_machine_readable() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format_version"] == "architecture-fragmentation-v1"
    assert payload["source_python_files"] > 0
    assert payload["run_orchestrator_file_count"] > 0
    assert isinstance(payload["reexport_shims"], list)
