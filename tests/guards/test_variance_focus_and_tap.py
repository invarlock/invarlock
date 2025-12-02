from invarlock.guards.variance import VarianceGuard


def test_focus_and_tap_matching():
    g = VarianceGuard(policy={"tap": ["transformer.h.*.mlp.c_proj"]})
    # Tap matches mlp but not attn
    assert g._matches_tap("block0.mlp") is True
    assert g._matches_tap("block0.attn") is False

    # Focus list filters strictly
    g._focus_modules = {"transformer.h.0.attn.c_proj"}
    assert g._is_focus_match("block0.attn") is True
    assert g._is_focus_match("block1.attn") is False
