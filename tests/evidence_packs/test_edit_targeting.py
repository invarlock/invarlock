from __future__ import annotations

from scripts.evidence_packs.python.editing import implementations as edits
from scripts.evidence_packs.python.editing import tensor_ops, transformation_contract


def test_legacy_substring_targeting_api_is_absent() -> None:
    for module in (tensor_ops, edits):
        for name in (
            "_matches_scope",
            "matches_edit_scope",
            "parse_scope_layers",
            "extract_layer_index",
        ):
            assert not hasattr(module, name), f"{module.__name__}.{name} survived"


def test_architecture_aware_transformation_contract_owns_targeting() -> None:
    assert callable(transformation_contract.checkpoint_transformation_contract)
    assert callable(transformation_contract.is_transformation_target)
    assert transformation_contract.validate_transformation_scope(
        "ffn@layers=2,layer=1"
    ) == ("ffn@layers=2,layer=1")
