from invarlock.cli.commands import run


def test_should_measure_overhead_respects_env(monkeypatch):
    monkeypatch.setenv("INVARLOCK_SKIP_OVERHEAD_CHECK", "1")
    measure, skip = run._should_measure_overhead("ci")
    assert skip is True
    assert measure is False

    monkeypatch.delenv("INVARLOCK_SKIP_OVERHEAD_CHECK", raising=False)
    measure_default, skip_default = run._should_measure_overhead("release")
    assert skip_default is False
    assert measure_default is True


def test_should_measure_overhead_non_ci_profile(monkeypatch):
    monkeypatch.delenv("INVARLOCK_SKIP_OVERHEAD_CHECK", raising=False)
    measure, skip = run._should_measure_overhead("dev")
    assert skip is False
    assert measure is False
