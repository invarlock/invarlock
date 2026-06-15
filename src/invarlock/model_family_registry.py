"""Data-driven model-family registry helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from invarlock.public_contracts import load_model_family_catalog, load_support_matrix

CATALOG_MODEL_SECTIONS: tuple[str, ...] = (
    "declared_support",
    "implemented_coverage",
    "usage_only",
    "recommended_additions",
)
_PRESET_PREFIX = "configs/presets/"
_PRESET_SUFFIX = ".yaml"
_PRESET_ADAPTER_SEGMENTS = {
    "causal_lm": "hf_causal",
    "masked_lm": "hf_mlm",
    "multimodal": "hf_multimodal",
    "seq2seq": "hf_seq2seq",
}


class CatalogRouteUnavailable(ValueError):
    """Raised when a catalog record does not describe a runnable evidence lane."""


@dataclass(frozen=True, slots=True)
class ModelFamilyRecord:
    section: str
    family_id: str
    display_name: str
    representative_model: str
    representative_index: int
    modalities: tuple[str, ...]
    task_role: str
    state: str
    repo_evidence: tuple[str, ...]
    support_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogLaneDefaults:
    preset_relpath: str
    adapter: str
    source: str


def catalog_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "_")
    for old, new in ((".", "_"), ("-", "_"), ("+", "_")):
        slug = slug.replace(old, new)
    return slug


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _support_rows_by_model_id(
    support_matrix: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    rows_by_model: dict[str, list[Mapping[str, Any]]] = {}
    lanes = support_matrix.get("lanes")
    if not isinstance(lanes, list):
        return {}
    for row in lanes:
        if not isinstance(row, Mapping):
            continue
        for model_id in _as_str_tuple(row.get("representative_models")):
            rows_by_model.setdefault(model_id, []).append(row)
    return {key: tuple(value) for key, value in rows_by_model.items()}


def iter_model_family_records(
    *,
    catalog: Mapping[str, Any] | None = None,
    sections: Iterable[str] = CATALOG_MODEL_SECTIONS,
) -> tuple[ModelFamilyRecord, ...]:
    payload = catalog or load_model_family_catalog()
    records: list[ModelFamilyRecord] = []
    for section in sections:
        families = payload.get(section) or []
        if not isinstance(families, list):
            raise ValueError(f"model_family_catalog.{section} must be a list")
        for family in families:
            if not isinstance(family, Mapping):
                continue
            models = family.get("representative_models") or []
            if not isinstance(models, list):
                continue
            family_id = str(family.get("family_id") or "")
            display_name = str(family.get("display_name") or family_id or section)
            modalities = _as_str_tuple(family.get("modalities"))
            support_groups = _as_str_tuple(family.get("support_groups"))
            repo_evidence = _as_str_tuple(family.get("repo_evidence"))
            task_role = str(family.get("task_role") or "")
            state = str(family.get("state") or "")
            for index, model_id in enumerate(models):
                if not isinstance(model_id, str) or not model_id:
                    continue
                records.append(
                    ModelFamilyRecord(
                        section=section,
                        family_id=family_id,
                        display_name=display_name,
                        representative_model=model_id,
                        representative_index=index,
                        modalities=modalities,
                        task_role=task_role,
                        state=state,
                        repo_evidence=repo_evidence,
                        support_groups=support_groups,
                    )
                )
    return tuple(records)


def records_by_model_id(
    *,
    catalog: Mapping[str, Any] | None = None,
    sections: Iterable[str] = CATALOG_MODEL_SECTIONS,
) -> dict[str, tuple[ModelFamilyRecord, ...]]:
    grouped: dict[str, list[ModelFamilyRecord]] = {}
    for record in iter_model_family_records(catalog=catalog, sections=sections):
        grouped.setdefault(record.representative_model, []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def is_ambiguous_model_id(records: Iterable[ModelFamilyRecord]) -> bool:
    signatures = {
        (record.task_role, record.modalities)
        for record in records
        if record.task_role or record.modalities
    }
    return len(signatures) > 1


def _adapter_from_record(record: ModelFamilyRecord) -> str:
    role = record.task_role.lower()
    modalities = {item.lower() for item in record.modalities}
    if role in {"masked_lm", "mlm"}:
        return "hf_mlm"
    if "seq2seq" in role or role in {"translation", "summarization"}:
        return "hf_seq2seq"
    if role in {"any_to_any", "image_text"}:
        return "hf_multimodal"
    if "causal" in role or role.endswith("_lm"):
        return "hf_causal"
    if "image" in modalities or "audio" in modalities or "multimodal" in role:
        return "hf_multimodal"
    return "auto"


def _preset_adapter(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) < 3 or parts[:2] != ["configs", "presets"]:
        return None
    return _PRESET_ADAPTER_SEGMENTS.get(parts[2])


def _preset_paths(
    record: ModelFamilyRecord,
    *,
    adapter: str | None = None,
) -> tuple[str, ...]:
    paths = [
        path
        for path in record.repo_evidence
        if path.startswith(_PRESET_PREFIX) and path.endswith(_PRESET_SUFFIX)
    ]
    if adapter in {None, "auto"}:
        return tuple(paths)
    return tuple(path for path in paths if _preset_adapter(path) == adapter)


def _model_tokens(model_id: str) -> set[str]:
    normalized = catalog_slug(model_id)
    return {
        token
        for token in re.split(r"_+", normalized)
        if token and token not in {"the", "instruct", "it", "hf"}
    }


def _score_preset_path(model_id: str, path: str) -> int:
    path_slug = catalog_slug(path)
    return sum(1 for token in _model_tokens(model_id) if token in path_slug)


def _select_preset_path(
    record: ModelFamilyRecord,
    *,
    adapter: str | None = None,
) -> str | None:
    paths = _preset_paths(record, adapter=adapter)
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    scored = sorted(
        (
            (-_score_preset_path(record.representative_model, path), index, path)
            for index, path in enumerate(paths)
        ),
    )
    return scored[0][2]


def _default_preset_for_adapter(adapter: str) -> str:
    if adapter == "hf_mlm":
        return "configs/presets/masked_lm/wikitext2_128.yaml"
    if adapter == "hf_seq2seq":
        return "configs/presets/seq2seq/synth_128.yaml"
    if adapter == "hf_multimodal":
        raise CatalogRouteUnavailable(
            "multimodal catalog records must name a concrete preset"
        )
    return "configs/presets/causal_lm/wikitext2_512.yaml"


def _support_row_score(
    record: ModelFamilyRecord,
    row: Mapping[str, Any],
    inferred_adapter: str,
) -> tuple[int, str]:
    score = 0
    row_adapter = row.get("adapter")
    if row_adapter == inferred_adapter:
        score += 4
    if row.get("family") == record.display_name:
        score += 3
    row_groups = set(_as_str_tuple(row.get("support_groups")))
    if row_groups.intersection(record.support_groups):
        score += 1
    return score, str(row_adapter or "")


def _adapter_from_support_rows(
    record: ModelFamilyRecord,
    rows: tuple[Mapping[str, Any], ...],
    inferred_adapter: str,
) -> str:
    if not rows:
        return inferred_adapter
    ranked = sorted(
        (
            (-_support_row_score(record, row, inferred_adapter)[0], index, row)
            for index, row in enumerate(rows)
        )
    )
    best = ranked[0][2]
    adapter = best.get("adapter")
    if isinstance(adapter, str) and adapter:
        return adapter
    return inferred_adapter


def catalog_lane_defaults(
    record: ModelFamilyRecord,
    *,
    support_matrix: Mapping[str, Any] | None = None,
) -> CatalogLaneDefaults:
    support_payload = support_matrix or load_support_matrix()
    support_rows = _support_rows_by_model_id(support_payload).get(
        record.representative_model,
        (),
    )
    inferred_adapter = _adapter_from_record(record)
    adapter = _adapter_from_support_rows(record, support_rows, inferred_adapter)
    preset = _select_preset_path(record, adapter=adapter)
    if preset:
        return CatalogLaneDefaults(
            preset_relpath=preset,
            adapter=adapter,
            source="model_family_catalog.repo_evidence",
        )
    return CatalogLaneDefaults(
        preset_relpath=_default_preset_for_adapter(adapter),
        adapter=adapter,
        source="task_role_default",
    )


def catalog_lane_defaults_for_model(
    model_id: str,
    *,
    catalog: Mapping[str, Any] | None = None,
    support_matrix: Mapping[str, Any] | None = None,
) -> CatalogLaneDefaults:
    records = records_by_model_id(catalog=catalog).get(model_id, ())
    if records:
        return catalog_lane_defaults(records[0], support_matrix=support_matrix)
    record = ModelFamilyRecord(
        section="synthetic",
        family_id=model_id,
        display_name=model_id,
        representative_model=model_id,
        representative_index=0,
        modalities=("text",),
        task_role="causal_lm",
        state="unknown",
        repo_evidence=(),
        support_groups=(),
    )
    return catalog_lane_defaults(record, support_matrix=support_matrix)


def catalog_routed_model_ids(
    *,
    catalog: Mapping[str, Any] | None = None,
    support_matrix: Mapping[str, Any] | None = None,
) -> set[str]:
    support_payload = support_matrix or load_support_matrix()
    routed: set[str] = set()
    for record in iter_model_family_records(catalog=catalog):
        try:
            defaults = catalog_lane_defaults(record, support_matrix=support_payload)
        except CatalogRouteUnavailable:
            continue
        if defaults.preset_relpath and defaults.adapter:
            routed.add(record.representative_model)
    return routed


__all__ = [
    "CATALOG_MODEL_SECTIONS",
    "CatalogLaneDefaults",
    "CatalogRouteUnavailable",
    "ModelFamilyRecord",
    "catalog_lane_defaults",
    "catalog_lane_defaults_for_model",
    "catalog_routed_model_ids",
    "catalog_slug",
    "is_ambiguous_model_id",
    "iter_model_family_records",
    "records_by_model_id",
]
