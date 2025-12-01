from __future__ import annotations

import importlib


def _mk_err(mod_name: str, cls: str, *args, **kwargs):
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls)(*args, **kwargs)


def test_resolve_exit_code_invarlockerror_profiles():
    run_mod = importlib.import_module("invarlock.cli.commands.run")
    InvarlockError = importlib.import_module("invarlock.cli.errors").InvarlockError
    err = InvarlockError(code="E005", message="boom")
    assert run_mod._resolve_exit_code(err, profile="ci") == 3
    assert run_mod._resolve_exit_code(err, profile="release") == 3
    assert run_mod._resolve_exit_code(err, profile="dev") == 1


def test_resolve_exit_code_schema_validation_types():
    run_mod = importlib.import_module("invarlock.cli.commands.run")
    # These derive from InvarlockError but should map to 2 regardless of profile
    ConfigError = importlib.import_module("invarlock.core.exceptions").ConfigError
    ValidationError = importlib.import_module(
        "invarlock.core.exceptions"
    ).ValidationError
    DataError = importlib.import_module("invarlock.core.exceptions").DataError

    for e in (
        ConfigError(code="E201", message="cfg"),
        ValidationError(code="E202", message="val"),
        DataError(code="E203", message="data"),
    ):
        assert run_mod._resolve_exit_code(e, profile="dev") == 2
        assert run_mod._resolve_exit_code(e, profile="ci") == 2
        assert run_mod._resolve_exit_code(e, profile="release") == 2


def test_resolve_exit_code_invalid_runreport_value_error_special_case():
    run_mod = importlib.import_module("invarlock.cli.commands.run")
    e = ValueError("Invalid RunReport: shape mismatch")
    assert run_mod._resolve_exit_code(e, profile="dev") == 2
