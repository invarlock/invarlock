from __future__ import annotations

from invarlock.core.exceptions import (
    ConfigError,
    DataError,
    InvarlockError,
    ValidationError,
)
from invarlock.core.exit_codes import resolve_command_exit_code


def test_resolve_exit_code_invarlockerror_profiles() -> None:
    err = InvarlockError(code="E005", message="boom")
    assert resolve_command_exit_code(err, profile="ci") == 3
    assert resolve_command_exit_code(err, profile="release") == 3
    assert resolve_command_exit_code(err, profile="dev") == 1


def test_resolve_exit_code_schema_validation_types() -> None:
    for err in (
        ConfigError(code="E201", message="cfg"),
        ValidationError(code="E202", message="val"),
        DataError(code="E203", message="data"),
    ):
        assert resolve_command_exit_code(err, profile="dev") == 2
        assert resolve_command_exit_code(err, profile="ci") == 2
        assert resolve_command_exit_code(err, profile="release") == 2


def test_resolve_exit_code_invalid_runreport_value_error_special_case() -> None:
    err = ValueError("Invalid RunReport: shape mismatch")
    assert resolve_command_exit_code(err, profile="dev") == 2
