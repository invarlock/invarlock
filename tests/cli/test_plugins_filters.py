from __future__ import annotations

import json
import types
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app


def _prov(name: str):
    # Return a namespace similar to AdapterProvenance for controlled adapters
    name = name.lower()
    if name == "hf_bnb":
        return SimpleNamespace(
            family="bnb",
            library="bitsandbytes",
            version="0.42.0",
            supported=True,
            tested=[],
        )
    if name in {"hf_gptq", "hf_awq"}:
        # Simulate package missing → needs_extra
        lib = "auto-gptq" if name == "hf_gptq" else "autoawq"
        return SimpleNamespace(
            family=name.split("_")[1],
            library=lib,
            version=None,
            supported=False,
            tested=[],
        )
    # Core adapters → transformers present
    return SimpleNamespace(
        family="hf", library="transformers", version="4.40.0", supported=True, tested=[]
    )


def test_plugins_adapters_json_backend_and_filters(monkeypatch):
    # Stub torch to avoid CUDA and speed up import path
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    # Patch provenance to control backend presence
    import invarlock.cli.commands.plugins as plug_mod
    import invarlock.cli.provenance as prov_mod

    monkeypatch.setattr(prov_mod, "extract_adapter_provenance", _prov)
    # Force Linux to avoid Linux-only gating → needs_extra instead of unsupported
    monkeypatch.setattr(plug_mod.platform, "system", lambda: "Linux")

    r = CliRunner().invoke(app, ["plugins", "adapters", "--json", "--show-unsupported"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    items = payload.get("items", [])
    # hf_bnb present without CUDA → unsupported
    bnb = next((x for x in items if x.get("name") == "hf_bnb"), None)
    assert bnb and bnb.get("status") == "unsupported"
    assert bnb.get("backend", {}).get("name") == "bitsandbytes"
    assert bnb.get("backend", {}).get("version") == "0.42.0"
    # gptq/awq missing → needs_extra
    gptq = next((x for x in items if x.get("name") == "hf_gptq"), None)
    awq = next((x for x in items if x.get("name") == "hf_awq"), None)
    assert gptq and gptq.get("status") == "needs_extra"
    assert awq and awq.get("status") == "needs_extra"

    # only=missing filter should return only needs_extra
    r2 = CliRunner().invoke(
        app,
        ["plugins", "adapters", "--json", "--only", "missing", "--show-unsupported"],
    )
    assert r2.exit_code == 0
    payload2 = json.loads(r2.stdout.strip().splitlines()[-1])
    assert all(x.get("status") == "needs_extra" for x in payload2.get("items", []))
