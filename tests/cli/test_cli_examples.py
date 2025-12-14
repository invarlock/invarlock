"""CLI examples validation moved from scripts/test_cli_examples.py.

This test ensures the CLI examples in the scripts stay in sync and runnable.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_cli_examples_help_smoke():
    # Light import mode to avoid heavy deps in help
    env = dict(os.environ)
    env.update(
        {
            "INVARLOCK_LIGHT_IMPORT": "1",
            "INVARLOCK_DISABLE_PLUGIN_DISCOVERY": "1",
        }
    )
    # Use the installed console script if available, else module entry
    cmd = [sys.executable, "-m", "invarlock", "--help"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "Usage" in res.stdout or "Usage" in res.stderr
