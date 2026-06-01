from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_config_integrity_runs_ci_matrix_without_shell_wrapper(tmp_path: Path) -> None:
    script = Path("scripts/checks/check_config_integrity.py")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    python_link = bin_dir / "python3"
    python_link.symlink_to(sys.executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)

    result = subprocess.run(
        [sys.executable, str(script), "--ci-matrix", "configs"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Required CI preset/edit surfaces are present" in result.stdout
    assert "OK   hf_causal" in result.stdout
