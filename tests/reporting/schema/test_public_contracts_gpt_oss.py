from __future__ import annotations

from invarlock import public_contracts as contracts


def test_gpt_oss_catalog_row_is_published_with_public_evidence() -> None:
    catalog = contracts.load_model_family_catalog()
    implemented = {
        item["display_name"]: item for item in catalog["implemented_coverage"]
    }

    gpt_oss = implemented["GPT-OSS"]
    assert gpt_oss["state"] == "published_basis"
    assert (
        "public_evidence/published_basis/gpt_oss_20b/evidence_pack"
        in gpt_oss["repo_evidence"]
    )
    assert "alternate-seed/window robustness claim" in gpt_oss["notes"]
