"""Explicit native selection fails clearly without importing a Python fallback."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


@pytest.fixture
def unavailable(monkeypatch):
    from invarlock import native_backend

    original = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "invarlock_native":
            raise ModuleNotFoundError("deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(native_backend.importlib, "import_module", missing)


def test_required_extension_diagnostic(unavailable):
    from invarlock.native_backend import NativeBackendUnavailable, transact

    with pytest.raises(NativeBackendUnavailable, match="companion wheel"):
        transact("report", {})


def test_obsolete_extension_diagnostic(monkeypatch):
    from invarlock import native_backend

    monkeypatch.setattr(
        native_backend.importlib, "import_module", lambda _: SimpleNamespace()
    )
    with pytest.raises(
        native_backend.NativeBackendUnavailable, match="transaction API"
    ):
        native_backend.require_native()


@pytest.mark.parametrize("preflight", [False, True])
def test_public_evaluation_errors(unavailable, preflight):
    from invarlock.evaluation_transaction import (
        EvaluationPreflightError,
        EvaluationTransactionError,
        evaluate_request_file,
        preflight_evaluation_request,
    )

    function = preflight_evaluation_request if preflight else evaluate_request_file
    error = EvaluationPreflightError if preflight else EvaluationTransactionError
    with pytest.raises(error, match="companion wheel"):
        function(Path("unavailable.yaml"), signing_key_path=None, backend="rust")


def test_public_report_error(unavailable):
    from invarlock.evidence_reporting import EvidenceReportError, render_evidence

    with pytest.raises(EvidenceReportError, match="companion wheel") as caught:
        render_evidence(Path("unavailable"), backend="rust")
    assert caught.value.exit_code == 2


@pytest.mark.parametrize("command", ["evaluate", "verify", "report"])
def test_cli_structured_error(unavailable, tmp_path, command):
    from invarlock.cli.app import app

    path = tmp_path / ("request.yaml" if command == "evaluate" else "evidence")
    if command == "evaluate":
        path.write_text("{}\n")
    else:
        path.mkdir()
    result = CliRunner().invoke(
        app, [command, str(path), "--backend", "rust", "--json"]
    )
    assert result.exit_code == 2, (result.output, result.exception)
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "companion wheel" in " ".join(payload["errors"])


def test_default_import_does_not_load_native():
    import subprocess
    import sys

    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'invarlock_native':
        raise AssertionError('default imports attempted to load native extension')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import invarlock.native_backend
import invarlock.evaluation_transaction
import invarlock.evidence_verification
import invarlock.evidence_reporting
"""
    subprocess.run([sys.executable, "-c", script], check=True)
