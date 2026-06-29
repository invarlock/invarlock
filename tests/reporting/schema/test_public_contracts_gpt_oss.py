from __future__ import annotations

from invarlock import public_contracts as contracts


def _declared_by_name() -> dict[str, dict]:
    catalog = contracts.load_model_family_catalog()
    return {item["display_name"]: item for item in catalog["declared_support"]}


def test_gpt_oss_catalog_row_is_published_with_public_evidence() -> None:
    declared = _declared_by_name()
    gpt_oss = declared["GPT-OSS 20B causal LM"]

    assert gpt_oss["state"] == "published_basis"
    assert (
        "public_evidence/published_basis/gpt_oss_20b/evidence_pack"
        in gpt_oss["repo_evidence"]
    )
    assert "alternate-seed/window robustness claim" in gpt_oss["notes"]


def test_followon_public_evidence_rows_are_declared_support() -> None:
    declared = _declared_by_name()

    expected = {
        "Qwen3.5 27B image-text LM (scoped)": (
            "public_evidence/published_basis/qwen3_5_27b_scoped/evidence_pack",
            "Linear-attention module coverage remains",
        ),
        "Qwen3.6 27B image-text LM (scoped)": (
            "public_evidence/published_basis/qwen3_6_27b_scoped/evidence_pack",
            "Linear-attention module coverage remains",
        ),
        "Gemma 4 31B image-text LM": (
            "public_evidence/published_basis/gemma4_31b/evidence_pack",
            "0.610 final accuracy over 400 examples",
        ),
        "Gemma 4 26B-A4B MoE image-text LM": (
            "public_evidence/published_basis/gemma4_26b_a4b/evidence_pack",
            "0.555 final accuracy over 400 examples",
        ),
        "Mixtral 8x7B MoE causal LM": (
            "public_evidence/published_basis/mixtral_8x7b/evidence_pack",
            "Mixtral MoE causal LM published basis",
        ),
        "Qwen3 30B-A3B MoE causal LM": (
            "public_evidence/published_basis/qwen3_30b_a3b/evidence_pack",
            "scoped attention/router/shared-expert guard scans",
        ),
    }

    for display_name, (evidence_path, note_fragment) in expected.items():
        row = declared[display_name]
        assert row["state"] == "published_basis"
        assert evidence_path in row["repo_evidence"]
        assert note_fragment in row["notes"]
