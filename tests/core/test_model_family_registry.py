from __future__ import annotations

import pytest

from invarlock.model_family_registry import (
    CatalogRouteUnavailable,
    ModelFamilyRecord,
    ModelFamilyRouteIndex,
    catalog_lane_defaults,
    catalog_routed_model_ids,
    catalog_slug,
    is_ambiguous_model_id,
    records_by_model_id,
)


def _record(
    *,
    model_id: str = "demo/model",
    display_name: str = "Demo family",
    modalities: tuple[str, ...] = ("text",),
    task_role: str = "causal_lm",
    repo_evidence: tuple[str, ...] = (),
    support_groups: tuple[str, ...] = (),
) -> ModelFamilyRecord:
    return ModelFamilyRecord(
        section="declared_support",
        family_id=display_name.lower().replace(" ", "-"),
        display_name=display_name,
        representative_model=model_id,
        representative_index=0,
        modalities=modalities,
        task_role=task_role,
        state="published_basis",
        repo_evidence=repo_evidence,
        support_groups=support_groups,
    )


def test_catalog_slug_normalizes_hub_model_ids() -> None:
    assert catalog_slug("Qwen/Qwen3.5-4B") == "qwen_qwen3_5_4b"


def test_records_by_model_id_preserves_ambiguous_catalog_records() -> None:
    catalog = {
        "declared_support": [
            {
                "family_id": "demo-text",
                "display_name": "Demo text",
                "modalities": ["text"],
                "task_role": "causal_lm",
                "state": "published_basis",
                "representative_models": ["demo/model"],
            },
            {
                "family_id": "demo-image",
                "display_name": "Demo image",
                "modalities": ["text", "image"],
                "task_role": "image_text",
                "state": "published_basis",
                "representative_models": ["demo/model"],
            },
        ]
    }

    records = records_by_model_id(catalog=catalog)["demo/model"]

    assert len(records) == 2
    assert is_ambiguous_model_id(records) is True


def test_catalog_lane_defaults_uses_record_context_for_same_model_id() -> None:
    support_matrix = {
        "lanes": [
            {
                "family": "Demo text",
                "adapter": "hf_causal",
                "support_groups": ["text"],
                "representative_models": ["demo/model"],
            },
            {
                "family": "Demo image",
                "adapter": "hf_multimodal",
                "support_groups": ["vision"],
                "representative_models": ["demo/model"],
            },
        ]
    }
    text_record = _record(
        display_name="Demo text",
        repo_evidence=("configs/presets/causal_lm/demo_512.yaml",),
        support_groups=("text",),
    )
    image_record = _record(
        display_name="Demo image",
        modalities=("text", "image"),
        task_role="image_text",
        repo_evidence=("configs/presets/multimodal/demo_vqa_256.yaml",),
        support_groups=("vision",),
    )

    text_defaults = catalog_lane_defaults(
        text_record,
        support_matrix=support_matrix,
    )
    image_defaults = catalog_lane_defaults(
        image_record,
        support_matrix=support_matrix,
    )

    assert text_defaults.adapter == "hf_causal"
    assert text_defaults.preset_relpath == "configs/presets/causal_lm/demo_512.yaml"
    assert image_defaults.adapter == "hf_multimodal"
    assert image_defaults.preset_relpath == (
        "configs/presets/multimodal/demo_vqa_256.yaml"
    )


def test_model_family_route_index_preserves_ambiguous_context() -> None:
    catalog = {
        "declared_support": [
            {
                "family_id": "demo-text",
                "display_name": "Demo text",
                "modalities": ["text"],
                "task_role": "causal_lm",
                "state": "published_basis",
                "representative_models": ["demo/model"],
                "repo_evidence": ["configs/presets/causal_lm/demo_512.yaml"],
                "support_groups": ["text"],
            },
            {
                "family_id": "demo-image",
                "display_name": "Demo image",
                "modalities": ["text", "image"],
                "task_role": "image_text",
                "state": "published_basis",
                "representative_models": ["demo/model"],
                "repo_evidence": ["configs/presets/multimodal/demo_vqa_256.yaml"],
                "support_groups": ["vision"],
            },
        ]
    }
    support_matrix = {
        "lanes": [
            {
                "family": "Demo text",
                "adapter": "hf_causal",
                "support_groups": ["text"],
                "representative_models": ["demo/model"],
            },
            {
                "family": "Demo image",
                "adapter": "hf_multimodal",
                "support_groups": ["vision"],
                "representative_models": ["demo/model"],
            },
        ]
    }

    index = ModelFamilyRouteIndex.from_contracts(
        catalog=catalog,
        support_matrix=support_matrix,
    )
    text_record, image_record = index.records_for_model("demo/model")

    assert index.lane_defaults(text_record).adapter == "hf_causal"
    assert index.lane_defaults(image_record).adapter == "hf_multimodal"
    assert index.routed_model_ids() == {"demo/model"}


def test_catalog_lane_defaults_requires_explicit_multimodal_preset() -> None:
    record = _record(
        modalities=("text", "image"),
        task_role="image_text",
        repo_evidence=("src/invarlock/adapters/hf_multimodal.py",),
    )

    with pytest.raises(CatalogRouteUnavailable):
        catalog_lane_defaults(record, support_matrix={"lanes": []})


def test_catalog_lane_defaults_keeps_causal_role_text_default() -> None:
    record = _record(
        modalities=("text", "image"),
        task_role="causal_lm",
        repo_evidence=("src/invarlock/model_profile.py",),
    )

    defaults = catalog_lane_defaults(record, support_matrix={"lanes": []})

    assert defaults.adapter == "hf_causal"
    assert defaults.preset_relpath == "configs/presets/causal_lm/wikitext2_512.yaml"


def test_catalog_routed_model_ids_skips_records_without_runnable_route() -> None:
    catalog = {
        "declared_support": [
            {
                "family_id": "demo-causal",
                "display_name": "Demo causal",
                "modalities": ["text"],
                "task_role": "causal_lm",
                "state": "published_basis",
                "representative_models": ["demo/causal"],
            },
            {
                "family_id": "demo-image",
                "display_name": "Demo image",
                "modalities": ["text", "image"],
                "task_role": "image_text",
                "state": "planned",
                "representative_models": ["demo/image"],
                "repo_evidence": ["docs/reference/model-adapters.md"],
            },
        ]
    }

    assert catalog_routed_model_ids(catalog=catalog, support_matrix={"lanes": []}) == {
        "demo/causal"
    }
