from invarlock.plugins.hello_guard import HelloGuard


def test_hello_guard_passes_below_threshold():
    guard = HelloGuard(threshold=1.5)
    result = guard.validate(model=None, adapter=None, context={"hello_score": 1.0})
    assert result["passed"] is True
    assert result["action"] == "warn"


def test_hello_guard_blocks_above_threshold():
    guard = HelloGuard(threshold=0.5)
    result = guard.validate(model=None, adapter=None, context={"hello_score": 0.8})
    assert result["passed"] is False
    assert result["action"] == "abort"
