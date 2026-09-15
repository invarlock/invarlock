"""The three-scorer installed-wheel rehearsal uses production CLI and SDK paths."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.test_captured_wheel_smoke import _real_cli_transport

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/captured-results/scorer_wheel_smoke.py"
FIXTURE = ROOT / "examples/judge-measurements"


def _module():
    spec = importlib.util.spec_from_file_location("scorer_wheel_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transport(monkeypatch, module, fault=None):
    calls, roots = _real_cli_transport(monkeypatch, module, fault=fault)
    transport = module.subprocess.run

    def bounded(*arguments, timeout, **kwargs):
        assert timeout == 60
        return transport(*arguments, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", bounded)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--cli", "/candidate/bin/invarlock", "--fixture", str(FIXTURE)],
    )
    monkeypatch.setattr(module, "require_core_only", lambda: None)
    return calls, roots


def test_three_scorers_use_v2_cli_and_public_sdk_with_scoped_receipts(
    monkeypatch, capsys
):
    module = _module()
    calls, roots = _transport(monkeypatch, module)
    module.main()
    output = capsys.readouterr().out
    for kind in ("exact_match", "normalized_nll_per_utf8_byte", "judge"):
        assert (
            f"{kind}: installed SDK, v2 evaluation, independent receipt and reports pass"
            in output
        )
    assert [call.returncode for call in calls if call.args[1] == "verify"] == [0, 0, 7]
    assert not roots[0].exists()


def test_install_smoke_copies_every_fixture_needed_by_three_scorer_consumer(
    tmp_path, monkeypatch, capsys
):
    # Exercise the release helper's actual fixture inventory outside the checkout,
    # so checkout-only files cannot hide a release handoff omission.
    from scripts.release.core_wheel_consumers import FILES

    fixture = tmp_path / "judge"
    fixture.mkdir()
    for source, destination in FILES:
        if Path(destination).parent == Path("judge"):
            shutil.copy2(ROOT / source, tmp_path / destination)
    module = _module()
    calls, _ = _transport(monkeypatch, module)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--cli", "/candidate/bin/invarlock", "--fixture", str(fixture)],
    )
    monkeypatch.chdir(tmp_path)
    module.main()
    assert "judge: installed SDK" in capsys.readouterr().out
    assert [call.returncode for call in calls if call.args[1] == "verify"] == [0, 0, 7]


def test_requires_installed_cli(monkeypatch):
    module = _module()
    _transport(monkeypatch, module)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="Install the candidate wheel"):
        module.main()


def test_failed_commands_preserve_diagnostics_and_cleanup(monkeypatch):
    module = _module()

    def fail(completed, _root):
        completed.returncode = 2
        completed.stdout, completed.stderr = "publication failed", "reason"

    _, roots = _transport(monkeypatch, module, fault=fail)
    with pytest.raises(
        RuntimeError, match="evaluate returned 2, expected 0: publication failedreason"
    ):
        module.main()
    assert not roots[0].exists()


@pytest.mark.parametrize("mode", ["absent", "namespace-absent", "available", "loaded"])
def test_core_only_check_rejects_optional_provider_sdks(monkeypatch, mode):
    module = _module()

    def find(name):
        if mode == "namespace-absent" and name == "invarlock_addins.inspect_judge":
            raise ModuleNotFoundError(name)
        return object() if mode == "available" else None

    monkeypatch.setattr(module.importlib.util, "find_spec", find)
    for name in ("inspect_ai", "openai", "invarlock_addins.inspect_judge"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    if mode == "loaded":
        monkeypatch.setitem(sys.modules, "openai", object())
    if mode in {"available", "loaded"}:
        with pytest.raises(RuntimeError, match="optional module"):
            module.require_core_only()
    else:
        module.require_core_only()
