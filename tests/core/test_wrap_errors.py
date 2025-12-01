def test_wrap_errors_context_manager() -> None:
    from invarlock.core.error_utils import wrap_errors
    from invarlock.core.exceptions import AdapterError

    try:
        with wrap_errors(AdapterError, code="E202", message="ADAPTER-LOAD-FAILED"):
            raise ValueError("boom")
    except AdapterError as e:
        assert e.code == "E202"
        assert "ADAPTER-LOAD-FAILED" in str(e)
    else:  # pragma: no cover - sanity
        raise AssertionError("AdapterError not raised")


def test_wrap_errors_decorator() -> None:
    from invarlock.core.error_utils import wrap_errors
    from invarlock.core.exceptions import ValidationError

    @wrap_errors(ValidationError, code="E301", message="VALIDATION-FAILED")
    def risky(x: int) -> int:
        if x < 0:
            raise KeyError("neg")
        return x

    try:
        _ = risky(-1)
    except ValidationError as e:
        assert e.code == "E301"
        assert e.details is None
    else:  # pragma: no cover - sanity
        raise AssertionError("ValidationError not raised")


def test_wrap_errors_context_does_not_double_wrap_invarlockerror() -> None:
    from invarlock.core.error_utils import wrap_errors
    from invarlock.core.exceptions import InvarlockError, ValidationError

    try:
        with wrap_errors(ValidationError, code="E301", message="VALIDATION-FAILED"):
            raise InvarlockError(code="E999", message="existing")
    except InvarlockError as e:
        # Should propagate the original InvarlockError without re-wrapping
        assert e.code == "E999"
        assert "existing" in str(e)
    else:  # pragma: no cover
        raise AssertionError("InvarlockError not propagated")
