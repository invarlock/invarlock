from invarlock.guards.variance import VarianceGuard


def test_ensure_tensor_value_list_of_strings_returns_original():
    g = VarianceGuard()
    val = ["a", "b", "c"]
    out = g._ensure_tensor_value(val)
    assert out is val
