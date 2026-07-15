from __future__ import annotations

import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_linter_module():
    path = Path("scripts/docs/lint_assurance_xrefs.py")
    spec = importlib.util.spec_from_file_location("lint_assurance_xrefs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assurance_cross_reference_linter_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/docs/lint_assurance_xrefs.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_assurance_cross_reference_samples_do_not_require_runtime_builder(
    monkeypatch,
) -> None:
    module = _load_linter_module()
    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy" or name.startswith("invarlock.reporting.report_make"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    certs = module._sample_reports()

    assert certs
    assert any(
        module._path_exists_in_obj(cert, "dataset.windows.stats.bootstrap.seed")
        for cert in certs
    )
    assert any(
        module._path_exists_in_obj(cert, "primary_metric.ratio_vs_baseline")
        for cert in certs
    )


def test_assurance_cross_reference_regex_accepts_bare_file_and_pytest_node() -> None:
    module = _load_linter_module()
    text = (
        "See tests/core/runner/test_runner_pairing.py and "
        "tests/eval/test_assurance_contracts.py::test_seed_bundle_contract."
    )

    refs = module.TEST_REF_RE.findall(text)

    assert "tests/core/runner/test_runner_pairing.py" in refs
    assert "tests/eval/test_assurance_contracts.py::test_seed_bundle_contract" in refs
