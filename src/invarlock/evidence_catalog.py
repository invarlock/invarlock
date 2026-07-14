"""Public evidence-catalog bindings and exact pack-set verification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invarlock.evidence_catalog_binding import validate_catalog_evidence_binding
from invarlock.evidence_catalog_contracts.execution import execution_policy_errors
from invarlock.evidence_catalog_contracts.primitives import (
    EvidenceCatalogError,
    canonical_json_bytes,
    entry_digest,
    input_digest,
)
from invarlock.evidence_catalog_contracts.primitives import (
    require_text as _require_text,
)
from invarlock.evidence_catalog_contracts.primitives import (
    safe_artifact_path as _safe_artifact_path,
)
from invarlock.evidence_catalog_contracts.primitives import (
    safe_preset_path as _safe_preset_path,
)
from invarlock.evidence_catalog_contracts.primitives import (
    sha256_bytes as _sha256_bytes,
)
from invarlock.evidence_catalog_contracts.primitives import (
    unexpected_keys as _unexpected_keys,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)
from invarlock.strict_yaml import StrictYamlError, parse_yaml_bytes

EVIDENCE_CATALOG_FORMAT = "invarlock/evidence-catalog-v1"
EVIDENCE_PACK_SET_RECEIPT_FORMAT = "invarlock/evidence-pack-set-receipt-v1"
RESOLVED_INPUTS_FORMAT = "invarlock/resolved-inputs-v1"
_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_COMMIT_RE = re.compile(r"[a-f0-9]{40}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_CATALOG_PATH = "metadata/catalog.json"
_CATALOG_KEYS = frozenset({"format_version", "entry_count", "entries"})
_ENTRY_KEYS = frozenset(
    {
        "lane_id",
        "slug",
        "model",
        "preset",
        "inputs",
        "execution",
        "required_artifacts",
    }
)
_MODEL_KEYS = frozenset({"id", "adapter"})
_PRESET_KEYS = frozenset({"path", "sha256"})
_INPUT_KEYS = frozenset({"kind", "digest", "source", "materialization"})
_INPUT_SOURCE_KEYS = frozenset({"provider", "dataset_id", "config_name", "split"})
_MATERIALIZATION_KEYS = frozenset(
    {
        "dataset",
        "revision",
        "config_name",
        "split",
        "max_samples",
        "min_usable_samples",
        "seed",
        "shuffle",
        "image_field",
        "prompt_field",
        "answer_field",
        "answers_field",
        "id_field",
        "prompt_template",
        "image_format",
    }
)
_REQUIRED_ARTIFACT_KEYS = frozenset({"role", "path"})
_BASE_REQUIRED_ARTIFACTS = {
    "report": "evaluation.report.json",
    "runtime_manifest": "runtime.manifest.json",
    "final_verdict": "final_verdict.json",
    "source_provenance": "source_repo.json",
    "resolved_inputs": "resolved-inputs.json",
    "runtime_config": "resolved-config.yaml",
    "preset": "preset.yaml",
    "independent_baseline": "baseline.report.json",
    "policy_pack": "policy-pack.json",
}
_VISION_REQUIRED_ARTIFACTS = {
    **_BASE_REQUIRED_ARTIFACTS,
    "input_materialization": "dataset/dataset_evidence.json",
}


def catalog_digest(path: Path) -> str:
    try:
        return _sha256_bytes(read_regular_file_bytes(path, label="evidence catalog"))
    except StrictJsonError as exc:
        raise EvidenceCatalogError(f"catalog cannot be loaded: {exc}") from exc


def _entry_errors(entry: object, *, index: int) -> list[str]:
    label = f"entries[{index}]"
    if not isinstance(entry, Mapping):
        return [f"{label} must be an object"]
    errors: list[str] = []
    errors.extend(_unexpected_keys(entry, allowed=_ENTRY_KEYS, label=label))
    _require_text(
        entry.get("lane_id"),
        label=f"{label}.lane_id",
        errors=errors,
        pattern=_IDENTIFIER_RE,
    )
    _require_text(
        entry.get("slug"), label=f"{label}.slug", errors=errors, pattern=_IDENTIFIER_RE
    )
    required_artifacts = entry.get("required_artifacts")
    declared_artifacts: dict[str, str] = {}
    if not isinstance(required_artifacts, list) or not required_artifacts:
        errors.append(f"{label}.required_artifacts must be a non-empty list")
    else:
        roles: set[str] = set()
        for artifact_index, artifact in enumerate(required_artifacts):
            artifact_label = f"{label}.required_artifacts[{artifact_index}]"
            if not isinstance(artifact, Mapping):
                errors.append(f"{artifact_label} must be an object")
                continue
            errors.extend(
                _unexpected_keys(
                    artifact, allowed=_REQUIRED_ARTIFACT_KEYS, label=artifact_label
                )
            )
            role = artifact.get("role")
            _require_text(
                role,
                label=f"{artifact_label}.role",
                errors=errors,
                pattern=_IDENTIFIER_RE,
            )
            if isinstance(role, str):
                if role in roles:
                    errors.append(
                        f"{label}.required_artifacts duplicates role {role!r}"
                    )
                roles.add(role)
            path = artifact.get("path")
            if not _safe_artifact_path(path):
                errors.append(f"{artifact_label}.path must be a safe relative path")
            elif isinstance(role, str) and isinstance(path, str):
                declared_artifacts[role] = path
    model = entry.get("model")
    if not isinstance(model, Mapping):
        errors.append(f"{label}.model must be an object")
    else:
        errors.extend(
            _unexpected_keys(model, allowed=_MODEL_KEYS, label=f"{label}.model")
        )
        _require_text(model.get("id"), label=f"{label}.model.id", errors=errors)
        _require_text(
            model.get("adapter"), label=f"{label}.model.adapter", errors=errors
        )
    preset = entry.get("preset")
    if not isinstance(preset, Mapping):
        errors.append(f"{label}.preset must be an object")
    else:
        errors.extend(
            _unexpected_keys(preset, allowed=_PRESET_KEYS, label=f"{label}.preset")
        )
        if not _safe_preset_path(preset.get("path")):
            errors.append(
                f"{label}.preset.path must be a safe YAML path below configs/"
            )
        _require_text(
            preset.get("sha256"),
            label=f"{label}.preset.sha256",
            errors=errors,
            pattern=_SHA256_RE,
        )
    inputs = entry.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append(f"{label}.inputs must be an object")
    else:
        errors.extend(
            _unexpected_keys(inputs, allowed=_INPUT_KEYS, label=f"{label}.inputs")
        )
        _require_text(inputs.get("kind"), label=f"{label}.inputs.kind", errors=errors)
        _require_text(
            inputs.get("digest"),
            label=f"{label}.inputs.digest",
            errors=errors,
            pattern=_SHA256_RE,
        )
        if isinstance(inputs.get("digest"), str) and inputs.get(
            "digest"
        ) != input_digest(inputs):
            errors.append(f"{label}.inputs.digest does not match declared inputs")
        source = inputs.get("source")
        if not isinstance(source, Mapping):
            errors.append(f"{label}.inputs.source must be an object")
        else:
            errors.extend(
                _unexpected_keys(
                    source, allowed=_INPUT_SOURCE_KEYS, label=f"{label}.inputs.source"
                )
            )
            _require_text(
                source.get("provider"),
                label=f"{label}.inputs.source.provider",
                errors=errors,
                pattern=_IDENTIFIER_RE,
            )
            _require_text(
                source.get("dataset_id"),
                label=f"{label}.inputs.source.dataset_id",
                errors=errors,
            )
            _require_text(
                source.get("split"),
                label=f"{label}.inputs.source.split",
                errors=errors,
            )
        if inputs.get("kind") == "vision_text":
            materialization = inputs.get("materialization")
            if not isinstance(materialization, Mapping):
                errors.append(f"{label}.inputs.materialization must be an object")
            else:
                errors.extend(
                    _unexpected_keys(
                        materialization,
                        allowed=_MATERIALIZATION_KEYS,
                        label=f"{label}.inputs.materialization",
                    )
                )
                for field in (
                    "dataset",
                    "revision",
                    "split",
                    "image_field",
                    "prompt_field",
                ):
                    _require_text(
                        materialization.get(field),
                        label=f"{label}.inputs.materialization.{field}",
                        errors=errors,
                    )
                if not isinstance(materialization.get("max_samples"), int) or (
                    materialization.get("max_samples", 0) <= 0
                ):
                    errors.append(
                        f"{label}.inputs.materialization.max_samples must be positive"
                    )
                revision = materialization.get("revision")
                if (
                    not isinstance(revision, str)
                    or _REVISION_RE.fullmatch(revision) is None
                ):
                    errors.append(
                        f"{label}.inputs.materialization.revision must be an immutable revision"
                    )
                if isinstance(source, Mapping) and source.get(
                    "dataset_id"
                ) != materialization.get("dataset"):
                    errors.append(
                        f"{label}.inputs.source.dataset_id must match materialization dataset"
                    )
                if (
                    isinstance(source, Mapping)
                    and source.get("provider") != "vision_text"
                ):
                    errors.append(
                        f"{label}.inputs.source.provider must be 'vision_text'"
                    )
        expected_artifacts = (
            _VISION_REQUIRED_ARTIFACTS
            if inputs.get("kind") == "vision_text"
            else _BASE_REQUIRED_ARTIFACTS
        )
        if declared_artifacts != expected_artifacts:
            errors.append(
                f"{label}.required_artifacts must declare the exact v1 role/path set"
            )
        errors.extend(
            execution_policy_errors(
                entry.get("execution"),
                label=f"{label}.execution",
            )
        )
    return errors


@dataclass(frozen=True)
class EvidenceCatalog:
    """A checked catalog plus digest-addressable entries."""

    path: Path
    digest: str
    payload: dict[str, object]
    entries: dict[str, dict[str, object]]

    def binding_for(self, lane_id: str, *, path: str) -> dict[str, str]:
        entry = self.entries.get(lane_id)
        if entry is None:
            raise EvidenceCatalogError(f"catalog has no entry for lane_id {lane_id!r}")
        return {
            "path": path,
            "digest": self.digest,
            "entry_id": lane_id,
            "entry_digest": entry_digest(entry),
        }


def load_evidence_catalog(path: Path) -> EvidenceCatalog:
    try:
        raw, payload = read_json_object_snapshot(path, label="evidence catalog")
    except StrictJsonError as exc:
        raise EvidenceCatalogError(f"catalog cannot be loaded: {exc}") from exc
    errors: list[str] = []
    errors.extend(_unexpected_keys(payload, allowed=_CATALOG_KEYS, label="catalog"))
    if payload.get("format_version") != EVIDENCE_CATALOG_FORMAT:
        errors.append(f"catalog format_version must be {EVIDENCE_CATALOG_FORMAT!r}")
    entries_value = payload.get("entries")
    if not isinstance(entries_value, list) or not entries_value:
        errors.append("catalog entries must be a non-empty list")
        entries_value = []
    count = payload.get("entry_count")
    if not isinstance(count, int) or count != len(entries_value):
        errors.append("catalog entry_count must exactly match entries")
    entries: dict[str, dict[str, object]] = {}
    slugs: set[str] = set()
    for index, candidate in enumerate(entries_value):
        errors.extend(_entry_errors(candidate, index=index))
        if not isinstance(candidate, dict):
            continue
        lane_id = candidate.get("lane_id")
        slug = candidate.get("slug")
        if not isinstance(lane_id, str) or not lane_id:
            continue
        if lane_id in entries:
            errors.append(f"catalog duplicates lane_id {lane_id!r}")
            continue
        if not isinstance(slug, str) or not slug:
            continue
        if slug in slugs:
            errors.append(f"catalog duplicates slug {slug!r}")
            continue
        entries[lane_id] = candidate
        slugs.add(slug)
    if errors:
        raise EvidenceCatalogError("; ".join(errors))
    return EvidenceCatalog(
        path=path,
        digest=_sha256_bytes(raw),
        payload=payload,
        entries=entries,
    )


def load_resolved_inputs(
    path: Path, *, entry: Mapping[str, object] | None = None
) -> tuple[dict[str, object], str]:
    """Load immutable per-run inputs and, when supplied, bind them to one entry."""

    try:
        raw, payload = read_json_object_snapshot(path, label="resolved inputs")
    except StrictJsonError as exc:
        raise EvidenceCatalogError(f"resolved inputs cannot be loaded: {exc}") from exc
    errors: list[str] = []
    allowed = {
        "format_version",
        "lane_id",
        "model",
        "dataset",
        "preset",
    }
    errors.extend(
        _unexpected_keys(payload, allowed=frozenset(allowed), label="resolved inputs")
    )
    if payload.get("format_version") != RESOLVED_INPUTS_FORMAT:
        errors.append(
            f"resolved inputs format_version must be {RESOLVED_INPUTS_FORMAT!r}"
        )
    _require_text(
        payload.get("lane_id"),
        label="resolved inputs.lane_id",
        errors=errors,
        pattern=_IDENTIFIER_RE,
    )
    for field in ("model", "dataset", "preset"):
        value = payload.get(field)
        if not isinstance(value, Mapping):
            errors.append(f"resolved inputs.{field} must be an object")
    model = payload.get("model")
    if isinstance(model, Mapping):
        errors.extend(
            _unexpected_keys(
                model,
                allowed=frozenset({"id", "adapter", "revision"}),
                label="resolved inputs.model",
            )
        )
        _require_text(model.get("id"), label="resolved inputs.model.id", errors=errors)
        _require_text(
            model.get("adapter"),
            label="resolved inputs.model.adapter",
            errors=errors,
        )
        _require_text(
            model.get("revision"),
            label="resolved inputs.model.revision",
            errors=errors,
            pattern=_REVISION_RE,
        )
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping):
        errors.extend(
            _unexpected_keys(
                dataset,
                allowed=frozenset(
                    {"provider", "id", "revision", "config_name", "split"}
                ),
                label="resolved inputs.dataset",
            )
        )
        _require_text(
            dataset.get("provider"),
            label="resolved inputs.dataset.provider",
            errors=errors,
            pattern=_IDENTIFIER_RE,
        )
        _require_text(
            dataset.get("id"), label="resolved inputs.dataset.id", errors=errors
        )
        _require_text(
            dataset.get("revision"),
            label="resolved inputs.dataset.revision",
            errors=errors,
            pattern=_REVISION_RE,
        )
        _require_text(
            dataset.get("split"), label="resolved inputs.dataset.split", errors=errors
        )
    preset = payload.get("preset")
    if isinstance(preset, Mapping):
        errors.extend(
            _unexpected_keys(
                preset,
                allowed=frozenset({"path", "sha256"}),
                label="resolved inputs.preset",
            )
        )
        if not _safe_preset_path(preset.get("path")):
            errors.append(
                "resolved inputs.preset.path must be a safe YAML path below configs/"
            )
        _require_text(
            preset.get("sha256"),
            label="resolved inputs.preset.sha256",
            errors=errors,
            pattern=_SHA256_RE,
        )
    if entry is not None:
        entry_model = entry.get("model")
        entry_preset = entry.get("preset")
        entry_inputs = entry.get("inputs")
        if (
            not isinstance(entry_model, Mapping)
            or not isinstance(entry_preset, Mapping)
            or not isinstance(entry_inputs, Mapping)
        ):
            errors.append("catalog entry is invalid")
        else:
            if payload.get("lane_id") != entry.get("lane_id"):
                errors.append("resolved inputs lane_id does not match catalog entry")
            if not isinstance(model, Mapping) or model.get("id") != entry_model.get(
                "id"
            ):
                errors.append("resolved inputs model id does not match catalog entry")
            if not isinstance(model, Mapping) or model.get(
                "adapter"
            ) != entry_model.get("adapter"):
                errors.append(
                    "resolved inputs model adapter does not match catalog entry"
                )
            if not isinstance(preset, Mapping) or preset.get(
                "path"
            ) != entry_preset.get("path"):
                errors.append(
                    "resolved inputs preset path does not match catalog entry"
                )
            if not isinstance(preset, Mapping) or preset.get(
                "sha256"
            ) != entry_preset.get("sha256"):
                errors.append(
                    "resolved inputs preset digest does not match catalog entry"
                )
            source = entry_inputs.get("source")
            if isinstance(source, Mapping) and isinstance(dataset, Mapping):
                if dataset.get("provider") != source.get("provider"):
                    errors.append(
                        "resolved inputs dataset provider does not match catalog entry"
                    )
                if dataset.get("id") != source.get("dataset_id"):
                    errors.append(
                        "resolved inputs dataset id does not match catalog entry"
                    )
                if dataset.get("split") != source.get("split"):
                    errors.append(
                        "resolved inputs dataset split does not match catalog entry"
                    )
                if dataset.get("config_name") != source.get("config_name"):
                    errors.append(
                        "resolved inputs dataset config does not match catalog entry"
                    )
            materialization = entry_inputs.get("materialization")
            if (
                isinstance(materialization, Mapping)
                and isinstance(dataset, Mapping)
                and (dataset.get("revision") != materialization.get("revision"))
            ):
                errors.append(
                    "resolved inputs dataset revision does not match catalog entry"
                )
    if errors:
        raise EvidenceCatalogError("; ".join(errors))
    return payload, _sha256_bytes(raw)


def _declared_artifact_roles(entry: Mapping[str, object]) -> dict[str, str]:
    required_artifacts = entry.get("required_artifacts")
    if not isinstance(required_artifacts, list):
        return {}
    return {
        str(item.get("role")): str(item.get("path"))
        for item in required_artifacts
        if isinstance(item, Mapping)
    }


def validate_embedded_catalog_binding(
    pack_dir: Path,
    manifest: Mapping[str, object],
    *,
    expected_catalog_digest: str | None = None,
) -> list[str]:
    """Reconstruct a pack's catalog evidence contract from authenticated bytes."""

    binding = manifest.get("catalog")
    if binding is None:
        return []
    if not isinstance(binding, Mapping):
        return ["catalog binding must be an object"]
    if expected_catalog_digest is None:
        return ["catalog-bound verification requires an independent catalog digest"]
    if _SHA256_RE.fullmatch(expected_catalog_digest) is None:
        return ["expected catalog digest has an invalid format"]
    if binding.get("path") != _CATALOG_PATH:
        return ["catalog path must be metadata/catalog.json"]
    try:
        catalog = load_evidence_catalog(pack_dir / _CATALOG_PATH)
    except EvidenceCatalogError as exc:
        return [f"embedded catalog is invalid: {exc}"]
    lane_id = binding.get("entry_id")
    entry = catalog.entries.get(lane_id) if isinstance(lane_id, str) else None
    errors: list[str] = []
    if binding.get("digest") != catalog.digest:
        errors.append("catalog digest does not match embedded catalog")
    if catalog.digest != expected_catalog_digest:
        errors.append("catalog digest does not match the independent catalog digest")
    if entry is None:
        errors.append("catalog entry id is not in embedded catalog")
    elif binding.get("entry_digest") != entry_digest(entry):
        errors.append("catalog entry digest does not match embedded catalog")
    if entry is None:
        return errors

    materials_value = manifest.get("materials")
    materials: dict[str, str] = {}
    if not isinstance(materials_value, list):
        return errors + ["catalog-bound pack materials are missing"]
    for material in materials_value:
        if not isinstance(material, Mapping):
            errors.append("catalog-bound pack material declaration is invalid")
            continue
        name = material.get("name")
        path = material.get("path")
        if not isinstance(name, str) or not isinstance(path, str) or name in materials:
            errors.append("catalog-bound pack material names must be unique")
            continue
        materials[name] = path

    material_paths = {
        "catalog": "metadata/catalog.json",
        "resolved-inputs": "metadata/resolved-inputs.json",
        "runtime-config": "metadata/runtime-config.yaml",
        "preset": "metadata/preset.yaml",
        "input-materialization": "metadata/input-materialization.json",
    }
    for name in ("catalog", "resolved-inputs", "runtime-config", "preset"):
        if materials.get(name) != material_paths[name]:
            errors.append(f"catalog-bound pack {name} material path is invalid")
    entry_inputs = entry.get("inputs")
    vision = (
        isinstance(entry_inputs, Mapping) and entry_inputs.get("kind") == "vision_text"
    )
    if vision:
        if (
            materials.get("input-materialization")
            != material_paths["input-materialization"]
        ):
            errors.append("catalog-bound pack input materialization path is invalid")
    elif "input-materialization" in materials:
        errors.append("non-vision catalog pack has input materialization")

    expected_role_paths = {
        "report": "reports/report-001/evaluation.report.json",
        "runtime_manifest": "reports/report-001/runtime.manifest.json",
        "final_verdict": "results/final_verdict.json",
        "source_provenance": "metadata/source_repo.json",
        "resolved_inputs": material_paths["resolved-inputs"],
        "runtime_config": material_paths["runtime-config"],
        "preset": material_paths["preset"],
        "independent_baseline": "baselines/baseline-001/evaluation.report.json",
        "policy_pack": "policy/policy-pack.json",
    }
    if vision:
        expected_role_paths["input_materialization"] = material_paths[
            "input-materialization"
        ]
    declared_roles = _declared_artifact_roles(entry)
    expected_declared = (
        _VISION_REQUIRED_ARTIFACTS if vision else _BASE_REQUIRED_ARTIFACTS
    )
    if declared_roles != expected_declared:
        errors.append("catalog entry required artifacts are not canonical")
    report_paths = sorted((pack_dir / "reports").glob("**/evaluation.report.json"))
    runtime_manifest_paths = sorted(
        (pack_dir / "reports").glob("**/runtime.manifest.json")
    )
    if report_paths != [pack_dir / expected_role_paths["report"]]:
        errors.append("catalog-bound pack must contain exactly one canonical report")
    if runtime_manifest_paths != [pack_dir / expected_role_paths["runtime_manifest"]]:
        errors.append(
            "catalog-bound pack must contain exactly one canonical runtime manifest"
        )
    for role, pack_path in expected_role_paths.items():
        if not (pack_dir / pack_path).is_file():
            errors.append(f"catalog required artifact is missing: {role}")

    subject = manifest.get("subject")
    if (
        not isinstance(subject, Mapping)
        or subject.get("path") != expected_role_paths["final_verdict"]
    ):
        errors.append("catalog final verdict manifest reference is invalid")
    invocation = manifest.get("invocation")
    source_ref = (
        invocation.get("config_source") if isinstance(invocation, Mapping) else None
    )
    if (
        not isinstance(source_ref, Mapping)
        or source_ref.get("path") != expected_role_paths["source_provenance"]
    ):
        errors.append("catalog source provenance manifest reference is invalid")
    baselines = manifest.get("verification_baselines")
    if (
        not isinstance(baselines, list)
        or len(baselines) != 1
        or not isinstance(baselines[0], Mapping)
        or baselines[0].get("path") != expected_role_paths["independent_baseline"]
    ):
        errors.append("catalog independent baseline manifest reference is invalid")
    policy = manifest.get("verification_policy_pack")
    if (
        not isinstance(policy, Mapping)
        or policy.get("path") != expected_role_paths["policy_pack"]
    ):
        errors.append("catalog policy-pack manifest reference is invalid")
    if errors:
        return errors

    try:
        resolved, resolved_digest = load_resolved_inputs(
            pack_dir / material_paths["resolved-inputs"], entry=entry
        )
        runtime_config_bytes = read_regular_file_bytes(
            pack_dir / material_paths["runtime-config"], label="resolved runtime config"
        )
        runtime_config = parse_yaml_bytes(
            runtime_config_bytes, label="resolved runtime config"
        )
        preset_bytes = read_regular_file_bytes(
            pack_dir / material_paths["preset"], label="catalog preset"
        )
        preset = parse_yaml_bytes(preset_bytes, label="catalog preset")
        _report_raw, report = read_json_object_snapshot(
            pack_dir / expected_role_paths["report"], label="catalog report"
        )
        _runtime_raw, runtime = read_json_object_snapshot(
            pack_dir / expected_role_paths["runtime_manifest"],
            label="catalog runtime manifest",
        )
        _baseline_raw, baseline = read_json_object_snapshot(
            pack_dir / expected_role_paths["independent_baseline"],
            label="catalog independent baseline",
        )
        _source_raw, source = read_json_object_snapshot(
            pack_dir / expected_role_paths["source_provenance"],
            label="catalog source provenance",
        )
        materialization: dict[str, object] | None = None
        if vision:
            _materialization_raw, materialization = read_json_object_snapshot(
                pack_dir / expected_role_paths["input_materialization"],
                label="catalog input materialization",
            )
    except (EvidenceCatalogError, OSError, StrictJsonError, StrictYamlError) as exc:
        return [f"catalog evidence material cannot be loaded: {exc}"]
    if not isinstance(runtime_config, Mapping) or not isinstance(preset, Mapping):
        return ["catalog evidence YAML materials must be objects"]
    errors.extend(
        validate_catalog_evidence_binding(
            entry=entry,
            catalog_digest=catalog.digest,
            catalog_entry_id=str(lane_id),
            catalog_entry_digest=entry_digest(entry),
            resolved_inputs=resolved,
            resolved_inputs_digest=resolved_digest,
            reports=[report],
            runtime_manifests=[runtime],
            runtime_config=runtime_config,
            runtime_config_digest=sha256_prefixed(runtime_config_bytes),
            preset=preset,
            preset_digest=sha256_prefixed(preset_bytes),
            baseline_report=baseline,
            input_materialization=materialization,
            source_provenance=source,
        )
    )
    return errors


__all__ = [
    "EVIDENCE_CATALOG_FORMAT",
    "EVIDENCE_PACK_SET_RECEIPT_FORMAT",
    "EvidenceCatalog",
    "EvidenceCatalogError",
    "canonical_json_bytes",
    "catalog_digest",
    "entry_digest",
    "input_digest",
    "load_evidence_catalog",
    "load_resolved_inputs",
    "validate_embedded_catalog_binding",
]
