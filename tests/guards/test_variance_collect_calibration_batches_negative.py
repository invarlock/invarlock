from invarlock.guards.variance import VarianceGuard


def test_collect_calibration_batches_negative_windows_returns_empty():
    g = VarianceGuard()
    data = [object(), object()]
    out = g._collect_calibration_batches(iter(data), windows=-5)
    assert out == []
