from invarlock.core.run_policy import should_measure_overhead


def test_should_measure_overhead_respects_config():
    measure, skip, source = should_measure_overhead(
        "ci", {"context": {"run": {"skip_overhead_check": True}}}
    )
    assert skip is True
    assert measure is False
    assert source == "config:context.run.skip_overhead_check"

    measure_default, skip_default, source_default = should_measure_overhead(
        "release", {}
    )
    assert skip_default is False
    assert measure_default is True
    assert source_default is None


def test_should_measure_overhead_non_ci_profile() -> None:
    measure, skip, source = should_measure_overhead(
        "dev", {"context": {"run": {"skip_overhead_check": True}}}
    )
    assert skip is False
    assert measure is False
    assert source is None
