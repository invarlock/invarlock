"""Semantic binding between one catalog lane and its evaluated evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import PurePath, PurePosixPath

from invarlock.core.config_runtime import InvarLockConfig
from invarlock.core.evaluate_plan import (
    build_subject_noop_run_config,
    resolve_guards_order,
    sanitize_preset_data_for_evaluate,
)
from invarlock.evidence_catalog_contracts.execution import (
    load_catalog_profile_overrides,
)
from invarlock.evidence_catalog_contracts.primitives import EvidenceCatalogError
from invarlock.vision_dataset_evidence import (
    validate_dataset_evidence,
    validate_evaluation_materialization_binding,
)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
SOURCE_PROVENANCE_FORMAT = "invarlock/source-provenance-v1"
EVALUATION_INPUT_BINDING_FORMAT = "invarlock/evaluation-input-binding-v1"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def evaluation_input_binding_errors(payload: object) -> list[str]:
    """Validate the closed, pre-run binding propagated into report provenance."""

    if not isinstance(payload, Mapping):
        return ["evaluation input binding must be an object"]
    base_fields = {
        "format_version",
        "catalog_digest",
        "catalog_entry_id",
        "catalog_entry_digest",
        "resolved_inputs_digest",
        "preset_digest",
    }
    vision_fields = {"materialization_digest", "materialization_manifest_digest"}
    keys = frozenset(payload)
    if keys not in {frozenset(base_fields), frozenset(base_fields | vision_fields)}:
        return ["evaluation input binding has non-canonical fields"]
    errors: list[str] = []
    if payload.get("format_version") != EVALUATION_INPUT_BINDING_FORMAT:
        errors.append("evaluation input binding format is invalid")
    for field in (
        "catalog_digest",
        "catalog_entry_digest",
        "resolved_inputs_digest",
        "preset_digest",
        "materialization_digest",
        "materialization_manifest_digest",
    ):
        if field in payload:
            value = payload.get(field)
            if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
                errors.append(f"evaluation input binding {field} is invalid")
    entry_id = payload.get("catalog_entry_id")
    if (
        not isinstance(entry_id, str)
        or not entry_id
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", entry_id) is None
    ):
        errors.append("evaluation input binding catalog_entry_id is invalid")
    return errors


def _field_error(
    errors: list[str], *, observed: object, expected: object, label: str
) -> None:
    if observed != expected:
        errors.append(f"catalog evidence {label} mismatch")


def _model_identity_errors(
    payload: Mapping[str, object],
    *,
    expected_model_id: object,
    expected_adapter: object,
    expected_revision: object,
    label: str,
) -> list[str]:
    errors: list[str] = []
    _field_error(
        errors,
        observed=payload.get("model_id"),
        expected=expected_model_id,
        label=f"{label}.model_id",
    )
    _field_error(
        errors,
        observed=payload.get("adapter"),
        expected=expected_adapter,
        label=f"{label}.adapter",
    )
    identity = _mapping(payload.get("model_identity"))
    _field_error(
        errors,
        observed=identity.get("kind"),
        expected="remote_revision",
        label=f"{label}.model_identity.kind",
    )
    _field_error(
        errors,
        observed=identity.get("revision"),
        expected=expected_revision,
        label=f"{label}.model_identity.revision",
    )
    return errors


def _runtime_provider(config: Mapping[str, object]) -> Mapping[str, object]:
    dataset = _mapping(config.get("dataset"))
    provider = dataset.get("provider")
    if isinstance(provider, str):
        return {"kind": provider}
    return _mapping(provider)


def _deep_merge(
    base: Mapping[str, object], overlay: Mapping[str, object]
) -> dict[str, object]:
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _yaml_material(value: object) -> object:
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _yaml_material(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_material(item) for item in value]
    if isinstance(value, tuple):
        return [_yaml_material(item) for item in value]
    return value


def _portable_runtime_path(
    value: object,
    *,
    label: str,
    errors: list[str],
    required_leaf: str | None = None,
) -> str | None:
    path = PurePosixPath(value) if isinstance(value, str) else None
    if (
        path is None
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        errors.append(f"catalog evidence {label} is not portable")
        return None
    if required_leaf is not None and path.name != required_leaf:
        errors.append(f"catalog evidence {label} must end in {required_leaf!r}")
        return None
    return value


def _normalize_loaded_config(payload: Mapping[str, object]) -> dict[str, object]:
    """Mirror the deterministic normalization performed while loading run YAML."""

    normalized = deepcopy(dict(payload))
    guards = normalized.get("guards")
    if isinstance(guards, dict):
        variance = guards.get("variance")
        if isinstance(variance, dict) and variance.get("mode") is None:
            variance["mode"] = "ci"
    return InvarLockConfig(normalized).model_dump()


def _apply_catalog_profile(
    payload: Mapping[str, object], execution: Mapping[str, object]
) -> dict[str, object]:
    overrides = load_catalog_profile_overrides(execution)
    base = deepcopy(dict(payload))
    merged = _deep_merge(base, overrides)
    base_primary_metric = base.get("primary_metric")
    merged_primary_metric = merged.get("primary_metric")
    if isinstance(base_primary_metric, Mapping) and isinstance(
        merged_primary_metric, Mapping
    ):
        merged["primary_metric"] = _deep_merge(
            merged_primary_metric, base_primary_metric
        )
    return InvarLockConfig(merged).model_dump()


def _expected_runtime_config(
    *,
    observed_config: Mapping[str, object],
    resolved_inputs: Mapping[str, object],
    preset: Mapping[str, object],
    execution: Mapping[str, object],
    expected_input_binding: Mapping[str, object],
    errors: list[str],
) -> dict[str, object] | None:
    resolved_model = _mapping(resolved_inputs.get("model"))
    resolved_dataset = _mapping(resolved_inputs.get("dataset"))
    adapter = resolved_model.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        errors.append("catalog evidence resolved model adapter is invalid")
        return None

    prepared = deepcopy(dict(preset))
    prepared_model = prepared.get("model")
    if not isinstance(prepared_model, dict):
        prepared_model = {}
        prepared["model"] = prepared_model
    prepared_model.update(
        {
            "id": resolved_model.get("id"),
            "adapter": adapter,
            "model_identity": {
                "kind": "remote_revision",
                "revision": resolved_model.get("revision"),
            },
        }
    )

    prepared_dataset = prepared.get("dataset")
    if not isinstance(prepared_dataset, dict):
        prepared_dataset = {}
        prepared["dataset"] = prepared_dataset
    provider_value = prepared_dataset.get("provider")
    prepared_provider = (
        dict(provider_value)
        if isinstance(provider_value, Mapping)
        else {"kind": provider_value}
    )
    expected_provider_kind = resolved_dataset.get("provider")
    prepared_provider["kind"] = expected_provider_kind
    if expected_provider_kind == "vision_text":
        observed_provider = _runtime_provider(observed_config)
        observed_manifest = _portable_runtime_path(
            observed_provider.get("path"),
            label="vision runtime provider path",
            errors=errors,
            required_leaf="manifest.jsonl",
        )
        if observed_manifest is None:
            return None
        for field in ("dataset_name", "config_name", "revision"):
            prepared_provider.pop(field, None)
        prepared_provider["path"] = observed_manifest
    else:
        prepared_provider.pop("path", None)
        prepared_provider["dataset_name"] = resolved_dataset.get("id")
        prepared_provider["revision"] = resolved_dataset.get("revision")
        config_name = resolved_dataset.get("config_name")
        if config_name is None:
            prepared_provider.pop("config_name", None)
        else:
            prepared_provider["config_name"] = config_name
    prepared_dataset["provider"] = prepared_provider
    prepared_dataset["split"] = resolved_dataset.get("split")

    sanitized = sanitize_preset_data_for_evaluate(
        prepared,
        adapter_name=adapter,
    )
    context = sanitized.setdefault("context", {})
    if not isinstance(context, dict):
        errors.append("catalog evidence preset context is invalid")
        return None
    context["evaluation_inputs"] = deepcopy(dict(expected_input_binding))
    try:
        guards_order = resolve_guards_order(sanitized, require_canonical=True)
    except ValueError as exc:
        errors.append(f"catalog evidence preset guard order is invalid: {exc}")
        return None

    output = _mapping(observed_config.get("output"))
    output_dir = _portable_runtime_path(
        output.get("dir"),
        label="runtime config output.dir",
        errors=errors,
        required_leaf="edited",
    )
    if output_dir is None:
        return None
    profile = execution.get("profile")
    tier = execution.get("tier")
    assurance_mode = execution.get("assurance_mode")
    execution_mode = execution.get("execution_mode")
    if not all(
        isinstance(value, str) and value
        for value in (profile, tier, assurance_mode, execution_mode)
    ):
        errors.append("catalog evidence execution policy is invalid")
        return None

    unprofiled = build_subject_noop_run_config(
        sanitized,
        model_id=str(resolved_model.get("id") or ""),
        adapter_name=adapter,
        model_identity={
            "kind": "remote_revision",
            "revision": str(resolved_model.get("revision") or ""),
        },
        output_dir=output_dir,
        profile=profile,
        tier=tier,
        guards_order=guards_order,
        assurance_mode=assurance_mode,
        execution_mode=execution_mode,
    )
    try:
        effective = _apply_catalog_profile(
            _normalize_loaded_config(unprofiled), execution
        )
    except (EvidenceCatalogError, TypeError, ValueError) as exc:
        errors.append(f"catalog evidence effective config cannot be derived: {exc}")
        return None
    auto = effective.get("auto")
    if not isinstance(auto, dict):
        auto = {}
        effective["auto"] = auto
    auto["tier"] = tier
    material = _yaml_material(InvarLockConfig(effective).model_dump())
    return material if isinstance(material, dict) else None


def _runtime_config_errors(
    config: Mapping[str, object],
    *,
    resolved_inputs: Mapping[str, object],
    preset: Mapping[str, object],
    execution: Mapping[str, object],
    expected_input_binding: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    expected_model = _mapping(resolved_inputs.get("model"))
    model = _mapping(config.get("model"))
    _field_error(
        errors,
        observed=model.get("id"),
        expected=expected_model.get("id"),
        label="runtime config model.id",
    )
    _field_error(
        errors,
        observed=model.get("adapter"),
        expected=expected_model.get("adapter"),
        label="runtime config model.adapter",
    )
    identity = _mapping(model.get("model_identity"))
    _field_error(
        errors,
        observed=identity.get("kind"),
        expected="remote_revision",
        label="runtime config model.model_identity.kind",
    )
    _field_error(
        errors,
        observed=identity.get("revision"),
        expected=expected_model.get("revision"),
        label="runtime config model.model_identity.revision",
    )

    expected_dataset = _mapping(resolved_inputs.get("dataset"))
    dataset = _mapping(config.get("dataset"))
    provider = _runtime_provider(config)
    expected_provider = expected_dataset.get("provider")
    _field_error(
        errors,
        observed=provider.get("kind"),
        expected=expected_provider,
        label="runtime config dataset.provider.kind",
    )
    _field_error(
        errors,
        observed=dataset.get("split"),
        expected=expected_dataset.get("split"),
        label="runtime config dataset.split",
    )
    if expected_provider != "vision_text":
        _field_error(
            errors,
            observed=provider.get("dataset_name"),
            expected=expected_dataset.get("id"),
            label="runtime config dataset.provider.dataset_name",
        )
        _field_error(
            errors,
            observed=provider.get("config_name"),
            expected=expected_dataset.get("config_name"),
            label="runtime config dataset.provider.config_name",
        )
        _field_error(
            errors,
            observed=provider.get("revision"),
            expected=expected_dataset.get("revision"),
            label="runtime config dataset.provider.revision",
        )
    context = _mapping(config.get("context"))
    context_input_binding = _mapping(context.get("evaluation_inputs"))
    if dict(context_input_binding) != dict(expected_input_binding):
        errors.append(
            "catalog evidence runtime config evaluation-input binding mismatch"
        )
    expected_config = _expected_runtime_config(
        observed_config=config,
        resolved_inputs=resolved_inputs,
        preset=preset,
        execution=execution,
        expected_input_binding=expected_input_binding,
        errors=errors,
    )
    if expected_config is not None:
        expected_provider_config = _runtime_provider(expected_config)
        if dict(provider) != dict(expected_provider_config):
            errors.append("catalog evidence runtime config dataset.provider mismatch")
        if config.get("dataset") != expected_config.get("dataset"):
            errors.append("catalog evidence runtime config dataset fields mismatch")
        if config.get("model") != expected_config.get("model"):
            errors.append("catalog evidence runtime config model fields mismatch")
        if config.get("context") != expected_config.get("context"):
            errors.append("catalog evidence runtime config context mismatch")
        if dict(config) != expected_config:
            errors.append(
                "catalog evidence runtime config does not match the effective execution config"
            )
    return errors


def _vision_errors(
    report: Mapping[str, object],
    *,
    resolved_dataset: Mapping[str, object],
    materialization: Mapping[str, object] | None,
) -> list[str]:
    if materialization is None:
        return ["catalog evidence input materialization is missing"]
    errors = validate_dataset_evidence(
        dict(materialization),
        strict_counts=True,
        require_runtime_identity=False,
    )
    materialized_dataset = _mapping(materialization.get("dataset"))
    for field, resolved_field in (
        ("id", "id"),
        ("revision", "revision"),
        ("config_name", "config_name"),
        ("split", "split"),
    ):
        _field_error(
            errors,
            observed=materialized_dataset.get(field),
            expected=resolved_dataset.get(resolved_field),
            label=f"input materialization dataset.{field}",
        )
    report_evidence = report.get("dataset_evidence")
    if not isinstance(report_evidence, Mapping):
        return errors + ["catalog evidence report dataset_evidence is missing"]
    errors.extend(
        validate_evaluation_materialization_binding(
            dict(materialization),
            dict(report_evidence),
            strict_counts=True,
        )
    )
    return errors


def validate_catalog_evidence_binding(
    *,
    entry: Mapping[str, object],
    catalog_digest: str,
    catalog_entry_id: str,
    catalog_entry_digest: str,
    resolved_inputs: Mapping[str, object],
    resolved_inputs_digest: str,
    reports: Sequence[Mapping[str, object]],
    runtime_manifests: Sequence[Mapping[str, object]],
    runtime_config: Mapping[str, object],
    preset: Mapping[str, object],
    runtime_config_digest: str,
    preset_digest: str,
    baseline_report: Mapping[str, object] | None = None,
    input_materialization: Mapping[str, object] | None = None,
    source_provenance: Mapping[str, object] | None = None,
) -> list[str]:
    """Reject any lane label that is not borne out by its authenticated evidence."""

    errors: list[str] = []
    if not isinstance(source_provenance, Mapping):
        errors.append("catalog evidence source provenance is missing")
        source_provenance = {}
    if set(source_provenance) != {
        "format_version",
        "commit",
        "source_bundle_sha256",
        "dirty",
    }:
        errors.append("catalog evidence source provenance has non-canonical fields")
    if source_provenance.get("format_version") != SOURCE_PROVENANCE_FORMAT:
        errors.append("catalog evidence source provenance format is invalid")
    source_commit = source_provenance.get("commit")
    if (
        not isinstance(source_commit, str)
        or _COMMIT_RE.fullmatch(source_commit) is None
    ):
        errors.append("catalog evidence source provenance commit is invalid")
    source_bundle = source_provenance.get("source_bundle_sha256")
    if (
        not isinstance(source_bundle, str)
        or _DIGEST_RE.fullmatch(source_bundle) is None
    ):
        errors.append("catalog evidence source bundle digest is invalid")
    if source_provenance.get("dirty") is not False:
        errors.append("catalog evidence source provenance must declare dirty=false")
    entry_model = _mapping(entry.get("model"))
    resolved_model = _mapping(resolved_inputs.get("model"))
    resolved_dataset = _mapping(resolved_inputs.get("dataset"))
    entry_preset = _mapping(entry.get("preset"))
    _field_error(
        errors,
        observed=preset_digest,
        expected=entry_preset.get("sha256"),
        label="preset digest",
    )
    expected_input_binding: dict[str, object] = {
        "format_version": EVALUATION_INPUT_BINDING_FORMAT,
        "catalog_digest": catalog_digest,
        "catalog_entry_id": catalog_entry_id,
        "catalog_entry_digest": catalog_entry_digest,
        "resolved_inputs_digest": resolved_inputs_digest,
        "preset_digest": preset_digest,
    }
    if input_materialization is not None:
        expected_input_binding["materialization_digest"] = input_materialization.get(
            "semantic_digest"
        )
        expected_input_binding["materialization_manifest_digest"] = (
            input_materialization.get("manifest_sha256")
        )
    errors.extend(
        _runtime_config_errors(
            runtime_config,
            resolved_inputs=resolved_inputs,
            preset=preset,
            execution=_mapping(entry.get("execution")),
            expected_input_binding=expected_input_binding,
        )
    )
    if len(reports) != 1 or len(runtime_manifests) != 1:
        errors.append(
            "catalog evidence requires exactly one report and runtime manifest"
        )
        return errors

    report = reports[0]
    for field in ("meta", "subject_ref"):
        errors.extend(
            _model_identity_errors(
                _mapping(report.get(field)),
                expected_model_id=entry_model.get("id"),
                expected_adapter=entry_model.get("adapter"),
                expected_revision=resolved_model.get("revision"),
                label=f"report.{field}",
            )
        )
    report_dataset = _mapping(report.get("dataset"))
    _field_error(
        errors,
        observed=_mapping(report.get("meta")).get("commit"),
        expected=source_commit,
        label="report.meta.commit",
    )
    _field_error(
        errors,
        observed=report_dataset.get("provider"),
        expected=resolved_dataset.get("provider"),
        label="report.dataset.provider",
    )
    _field_error(
        errors,
        observed=report_dataset.get("split"),
        expected=resolved_dataset.get("split"),
        label="report.dataset.split",
    )
    if resolved_dataset.get("provider") == "vision_text":
        errors.extend(
            _vision_errors(
                report,
                resolved_dataset=resolved_dataset,
                materialization=input_materialization,
            )
        )
    else:
        for field, resolved_field in (
            ("dataset_name", "id"),
            ("config_name", "config_name"),
            ("revision", "revision"),
        ):
            _field_error(
                errors,
                observed=report_dataset.get(field),
                expected=resolved_dataset.get(resolved_field),
                label=f"report.dataset.{field}",
            )

    report_input_binding = _mapping(
        _mapping(report.get("context")).get("evaluation_inputs")
    )
    if dict(report_input_binding) != expected_input_binding:
        errors.append("catalog evidence report evaluation-input binding mismatch")

    if not isinstance(baseline_report, Mapping):
        errors.append("catalog evidence independent baseline is missing")
    else:
        errors.extend(
            _model_identity_errors(
                _mapping(baseline_report.get("meta")),
                expected_model_id=entry_model.get("id"),
                expected_adapter=entry_model.get("adapter"),
                expected_revision=resolved_model.get("revision"),
                label="baseline.meta",
            )
        )
        baseline_dataset = _mapping(baseline_report.get("data"))
        _field_error(
            errors,
            observed=_mapping(baseline_report.get("meta")).get("commit"),
            expected=source_commit,
            label="baseline.meta.commit",
        )
        _field_error(
            errors,
            observed=(
                baseline_dataset.get("provider") or baseline_dataset.get("dataset")
            ),
            expected=resolved_dataset.get("provider"),
            label="baseline.data.provider",
        )
        _field_error(
            errors,
            observed=baseline_dataset.get("split"),
            expected=resolved_dataset.get("split"),
            label="baseline.data.split",
        )
        if resolved_dataset.get("provider") != "vision_text":
            for field, resolved_field in (
                ("dataset_name", "id"),
                ("config_name", "config_name"),
                ("revision", "revision"),
            ):
                _field_error(
                    errors,
                    observed=baseline_dataset.get(field),
                    expected=resolved_dataset.get(resolved_field),
                    label=f"baseline.data.{field}",
                )

    manifest_config = _mapping(runtime_manifests[0].get("config"))
    _field_error(
        errors,
        observed=manifest_config.get("path"),
        expected="resolved-config.yaml",
        label="runtime manifest config.path",
    )
    expected_hex = (
        runtime_config_digest.removeprefix("sha256:")
        if isinstance(runtime_config_digest, str)
        else None
    )
    _field_error(
        errors,
        observed=manifest_config.get("sha256"),
        expected=expected_hex,
        label="runtime manifest config.sha256",
    )
    source_binding = _mapping(
        _mapping(runtime_manifests[0].get("context")).get("source_bundle")
    )
    _field_error(
        errors,
        observed=source_binding.get("read_only"),
        expected=True,
        label="runtime manifest source bundle read_only",
    )
    _field_error(
        errors,
        observed=source_binding.get("sha256"),
        expected=source_bundle,
        label="runtime manifest source bundle sha256",
    )
    input_binding = _mapping(
        _mapping(runtime_manifests[0].get("context")).get("evaluation_inputs")
    )
    if dict(input_binding) != expected_input_binding:
        errors.append("catalog evidence runtime evaluation-input binding mismatch")
    return errors


__all__ = [
    "EVALUATION_INPUT_BINDING_FORMAT",
    "SOURCE_PROVENANCE_FORMAT",
    "evaluation_input_binding_errors",
    "validate_catalog_evidence_binding",
]
