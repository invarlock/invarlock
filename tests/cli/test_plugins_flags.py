import json
import os

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app

runner = CliRunner()


def test_plugins_alias_adapter_singular():
    # Should accept 'adapter' as alias for 'adapters'
    res = runner.invoke(app, ["plugins", "adapter", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload.get("kind") == "adapters"
    assert isinstance(payload.get("items"), list)


def test_plugins_hide_unsupported_filter():
    # Compare JSON with and without --hide-unsupported; the filtered set
    # should not contain any entries with status == 'unsupported'
    res_all = runner.invoke(app, ["plugins", "adapters", "--json"])
    assert res_all.exit_code == 0, res_all.output
    _ = json.loads(res_all.output)

    res_filtered = runner.invoke(
        app, ["plugins", "adapters", "--json", "--hide-unsupported"]
    )
    assert res_filtered.exit_code == 0, res_filtered.output
    filt_payload = json.loads(res_filtered.output)
    for r in filt_payload.get("items", []):
        assert r.get("status") != "unsupported"
