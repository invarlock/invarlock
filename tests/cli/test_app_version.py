import sys
from types import ModuleType

from invarlock.cli.app import version as version_fn


def test_version_prints_version(capsys, monkeypatch):
    # Inject a dummy invarlock module with __version__
    m = ModuleType("invarlock")
    m.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "invarlock", m)

    # Force package metadata import path to fail so we exercise fallback
    def _raise(*_args, **_kwargs):
        raise Exception("boom")

    monkeypatch.setattr("importlib.metadata.version", _raise)
    version_fn()
    out = capsys.readouterr().out
    assert "InvarLock 9.9.9" in out


def test_version_import_error_path(capsys, monkeypatch):
    # Force ImportError when importing invarlock
    import builtins as _builtins

    real_import = _builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "invarlock":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", fake_import)

    # Also make package metadata resolution fail
    def _raise(*_args, **_kwargs):
        raise Exception("boom")

    monkeypatch.setattr("importlib.metadata.version", _raise)
    version_fn()
    out = capsys.readouterr().out
    assert "version unknown" in out.lower()


def test_version_includes_schema_when_available(capsys, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "1.2.3")

    import invarlock.reporting.certificate as cert_mod

    monkeypatch.setattr(
        cert_mod, "CERTIFICATE_SCHEMA_VERSION", "schema-test", raising=False
    )
    version_fn()
    out = capsys.readouterr().out
    assert "1.2.3" in out
    assert "schema=schema-test" in out


def test_version_handles_schema_import_failure(capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "invarlock.reporting.certificate", raising=False)
    monkeypatch.delitem(sys.modules, "invarlock.reporting", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "invarlock.reporting.certificate":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.metadata.version", lambda _: "4.5.6")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    version_fn()
    out = capsys.readouterr().out
    assert "4.5.6" in out
    assert "schema=" not in out
