import pytest

from invarlock.core.exceptions import ConfigError
from invarlock.core.run_policy import should_measure_metric_impact


def test_should_measure_metric_impact_respects_config():
    measure, skip, source = should_measure_metric_impact(
        "ci", {"context": {"run": {"skip_guard_metric_impact_check": True}}}
    )
    assert skip is True
    assert measure is False
    assert source == "config:context.run.skip_guard_metric_impact_check"

    measure_default, skip_default, source_default = should_measure_metric_impact(
        "release", {}
    )
    assert skip_default is False
    assert measure_default is True
    assert source_default is None


def test_should_measure_metric_impact_rejects_release_skip_config() -> None:
    with pytest.raises(
        ConfigError, match="Release runs require measured guard metric impact"
    ):
        should_measure_metric_impact(
            "release",
            {"context": {"run": {"skip_guard_metric_impact_check": True}}},
        )


def test_should_measure_metric_impact_non_ci_profile() -> None:
    measure, skip, source = should_measure_metric_impact(
        "dev", {"context": {"run": {"skip_guard_metric_impact_check": True}}}
    )
    assert skip is False
    assert measure is False
    assert source is None
