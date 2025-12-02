from invarlock.guards.variance import VarianceGuard


def test_normalize_module_names_block_and_missing_cproj():
    g = VarianceGuard()
    # block prefix should convert to transformer.h.<idx>.<branch>.c_proj
    assert g._normalize_module_name("block3.attn") == "transformer.h.3.attn.c_proj"
    # Append .c_proj when missing for transformer paths
    assert (
        g._normalize_module_name("transformer.h.4.mlp") == "transformer.h.4.mlp.c_proj"
    )
