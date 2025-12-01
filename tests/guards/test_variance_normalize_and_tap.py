from invarlock.guards.variance import VarianceGuard


def test_normalize_pairing_ids_and_module_names_and_tap():
    g = VarianceGuard(policy={"tap": ["transformer.h.*.mlp.c_proj"]})

    # Pairing IDs: add prefix when not present, keep when present
    out = g._normalize_pairing_ids("preview", ["1", "preview::2"])
    assert out == ["preview::1", "preview::2"]

    # Module name normalization from block form
    assert g._normalize_module_name("block0.mlp") == "transformer.h.0.mlp.c_proj"
    # Already canonical stays canonical
    assert (
        g._normalize_module_name("transformer.h.1.attn.c_proj")
        == "transformer.h.1.attn.c_proj"
    )
    # Canonicalization when missing .c_proj suffix
    assert (
        g._normalize_module_name("transformer.h.2.mlp") == "transformer.h.2.mlp.c_proj"
    )
    assert (
        g._normalize_module_name("transformer.h.3.attn")
        == "transformer.h.3.attn.c_proj"
    )

    # Tap match on normalized name
    assert g._matches_tap("block3.mlp") is True
    # Focus list set causes non‑match
    g._focus_modules = {"transformer.h.0.mlp.c_proj"}
    assert g._is_focus_match("block5.mlp") is False
