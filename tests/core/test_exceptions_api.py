import re


def test_core_exceptions_import_and_str() -> None:
    from invarlock.core.exceptions import (
        AdapterError,
        ConfigError,
        DataError,
        DependencyError,
        EditError,
        GuardError,
        InvarlockError,
        MetricsError,
        ModelLoadError,
        ObservabilityError,
        PolicyViolationError,
        ResourceError,
        TimeoutError,
        ValidationError,
        VersionError,
    )

    # Base class behavior
    err = InvarlockError(code="E000", message="hello", details={"a": 1})
    assert str(err) == "[INVARLOCK:E000] hello"
    assert err.code == "E000"
    assert err.message == "hello"
    assert err.details == {"a": 1}
    assert err.recoverable is False

    # Subclassing shape sanity
    for exc in (
        AdapterError,
        ConfigError,
        DataError,
        DependencyError,
        EditError,
        GuardError,
        MetricsError,
        ModelLoadError,
        ObservabilityError,
        PolicyViolationError,
        ResourceError,
        TimeoutError,
        ValidationError,
        VersionError,
    ):
        e = exc(code="E123", message="m")
        assert isinstance(e, InvarlockError)
        assert re.match(r"^E\d{3}$", e.code)


def test_core_exceptions_reexport_from_core() -> None:
    # invarlock.core should re-export the base type for convenience
    from invarlock.core import InvarlockError as CE1
    from invarlock.core.exceptions import InvarlockError as CE2

    assert CE1 is CE2
