from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HEAVY_PREFIXES = ("torch", "transformers", "tensorflow", "accelerate", "xformers")
REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE_SCRIPT = r"""
import json
import sys

HEAVY_PREFIXES = ("torch", "transformers", "tensorflow", "accelerate", "xformers")


def heavy_modules():
    return sorted(
        module
        for module in sys.modules
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in HEAVY_PREFIXES
        )
    )


preloaded = heavy_modules()
if preloaded:
    print(json.dumps({"stage": "fresh_process_startup", "heavy": preloaded[:20]}))
    raise SystemExit(2)

from typer.testing import CliRunner
from invarlock.cli.app import app

category = sys.argv[1]
result = CliRunner().invoke(
    app,
    ["advanced", "plugins", "list", category, "--json"],
)
if result.exit_code != 0:
    print(
        json.dumps(
            {
                "stage": "command",
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "exception": repr(result.exception),
            }
        )
    )
    raise SystemExit(3)

command_payload = json.loads(result.stdout.strip().splitlines()[-1])
loaded = heavy_modules()
print(
    json.dumps(
        {
            "stage": "complete",
            "category": category,
            "command_payload_type": type(command_payload).__name__,
            "heavy": loaded[:20],
        }
    )
)
raise SystemExit(4 if loaded else 0)
"""


@pytest.mark.parametrize("category", ["adapters", "guards", "edits", "plugins"])
def test_plugins_json_does_not_import_heavy_libs_in_fresh_process(
    category: str,
) -> None:
    env = os.environ.copy()
    env["INVARLOCK_LIGHT_IMPORT"] = "1"
    env["INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"] = "0"
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, env.get("PYTHONPATH", "")) if part
    )

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, category],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = proc.stdout.strip().splitlines()
    assert output, proc.stderr
    payload = json.loads(output[-1])
    assert proc.returncode == 0, payload
    assert payload["stage"] == "complete"
    assert payload["category"] == category
    assert payload["heavy"] == []
