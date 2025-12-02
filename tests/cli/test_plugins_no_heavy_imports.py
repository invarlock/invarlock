import json
import sys

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app

HEAVY_PREFIXES = ("torch", "transformers", "tensorflow", "accelerate", "xformers")


@pytest.mark.parametrize("category", ["adapters", "guards", "edits", "plugins"])
def test_plugins_json_does_not_import_heavy_libs(category):
    # Skip if the test environment preloaded heavy libs.
    preloaded = {
        m
        for m in sys.modules
        for h in HEAVY_PREFIXES
        if m == h or m.startswith(h + ".")
    }
    if preloaded:
        pytest.skip(f"heavy libs already loaded: {sorted(preloaded)[:3]}...")

    res = CliRunner().invoke(app, ["plugins", "list", category, "--json"])
    assert res.exit_code == 0
    _ = json.loads(res.stdout.strip().splitlines()[-1])

    newly_loaded = {
        m
        for m in sys.modules
        for h in HEAVY_PREFIXES
        if m == h or m.startswith(h + ".")
    }
    assert not newly_loaded, f"heavy libs were imported: {sorted(newly_loaded)[:3]}..."
