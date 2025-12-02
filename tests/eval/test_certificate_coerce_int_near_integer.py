from invarlock.reporting.certificate import _coerce_int


def test_coerce_int_near_integer_boundary():
    # Very near but not exactly integer should return None
    assert _coerce_int(5.000000001) is None
    # Exact rounding still OK
    assert _coerce_int(round(5.0)) == 5
