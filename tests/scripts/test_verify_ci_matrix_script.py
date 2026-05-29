from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_verify_ci_matrix_script_falls_back_without_ripgrep(tmp_path: Path) -> None:
    script = Path("scripts/checks/verify_ci_matrix.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    python_link = bin_dir / "python3"
    grep_link = bin_dir / "grep"
    python_link.symlink_to(sys.executable)
    grep_link.symlink_to(Path(shutil.which("grep")).resolve())

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)

    result = subprocess.run(
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "✅ quant_rtn" in result.stdout
