from invarlock.guards.variance import VarianceGuard


def test_normalize_module_name_non_string_returns_empty():
    g = VarianceGuard()
    assert g._normalize_module_name(123) == ""
