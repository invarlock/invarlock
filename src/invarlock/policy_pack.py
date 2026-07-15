from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from invarlock.core.dataset_identity import DATASET_IDENTITY_FIELDS
from invarlock.core.runtime_provider.claims import RUNTIME_BEHAVIORAL_CLAIM_SET
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.guards.authority import (
    DEFAULT_GUARD_AUTHORITY,
    guard_authority_errors,
)
from invarlock.public_contracts import load_policy_pack_schema

try:  # pragma: no cover - exercised in integration/tests
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover
    jsonschema = None

if jsonschema is None:  # pragma: no cover - defensive import fallback
    _JSONSCHEMA_VALIDATE_ERRORS: tuple[type[BaseException], ...] = ()
else:  # pragma: no cover - exercised when jsonschema is installed
    _JSONSCHEMA_VALIDATE_ERRORS = (
        RuntimeError,
        jsonschema.SchemaError,
        TypeError,
        ValueError,
        jsonschema.ValidationError,
    )

POLICY_PACK_FORMAT = "policy-pack-v2"
LEGACY_POLICY_PACK_FORMAT = "policy-pack-v1"
BEHAVIORAL_POLICY_PACK_FORMAT = "policy-pack-v3"
POLICY_PACK_FORMATS = frozenset(
    {LEGACY_POLICY_PACK_FORMAT, POLICY_PACK_FORMAT, BEHAVIORAL_POLICY_PACK_FORMAT}
)
POLICY_PACK_DIGEST_PREFIX = "sha256:"

POLICY_PACK_TIERS = frozenset({"aggressive", "balanced", "conservative"})
POLICY_PACK_SUPPORT_TIER_ORDER = (
    "maintained_catalog",
    "supported_experimental",
    "community_experimental",
)
POLICY_PACK_SUPPORT_TIERS = frozenset(POLICY_PACK_SUPPORT_TIER_ORDER)
LEGACY_POLICY_PACK_SUPPORT_TIER_ORDER = (
    "published_basis",
    "supported_experimental",
    "community_experimental",
)
LEGACY_POLICY_PACK_SUPPORT_TIERS = frozenset(LEGACY_POLICY_PACK_SUPPORT_TIER_ORDER)
POLICY_PACK_REQUIRED_FIELDS = frozenset(
    {"format", "tier", "resolved_policy", "overrides", "policy_digest", "compatibility"}
)
POLICY_PACK_OPTIONAL_FIELDS = frozenset({"approval", "metadata", "behavioral_claim"})
POLICY_PACK_COMPATIBILITY_FIELDS = frozenset(
    {"support_tiers", "adapter_families", "runtime_lanes", "dataset_identity"}
)
POLICY_PACK_APPROVAL_FIELDS = frozenset(
    {"owner", "change_ticket", "rationale", "effective_date", "signature"}
)
POLICY_PACK_BEHAVIORAL_CLAIM_FIELDS = frozenset(
    {
        "claim_set",
        "schedule_sha256",
        "baseline",
        "subject",
        "required_capabilities",
        "metric_policy",
    }
)
POLICY_PACK_BEHAVIORAL_BINDING_FIELDS = frozenset(
    {
        "provider_name",
        "artifact_format",
        "artifact_identity_sha256",
        "outer_image_digest",
        "execution_settings_sha256",
    }
)
POLICY_PACK_BEHAVIORAL_CAPABILITY_FIELDS = frozenset(
    {"tasks", "metrics", "evidence_surfaces"}
)
POLICY_PACK_BEHAVIORAL_METRIC_FIELDS = frozenset(
    {"kind", "minimum_subject_score", "maximum_regression"}
)
RUNTIME_BEHAVIORAL_METRICS = frozenset({"exact_match"})
RUNTIME_ARTIFACT_FORMATS = frozenset({"hf_snapshot", "gguf", "tensorrt_llm_engine"})

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OVERRIDE_PATH_RE = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
_JSON_INTEGER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_JSON_FLOAT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


_STRUCTURED_TEXT_LOAD_ERRORS = (
    json.JSONDecodeError,
    OverflowError,
    RecursionError,
    TypeError,
    ValueError,
    yaml.YAMLError,
)
_STRUCTURED_FUZZ_SUFFIXES = (".json", ".yaml", ".yml")


def _choose_structured_fuzz_suffix(data: bytes) -> str:
    if not data:
        return ".json"
    return _STRUCTURED_FUZZ_SUFFIXES[data[0] % len(_STRUCTURED_FUZZ_SUFFIXES)]


class _StrictPolicyYamlLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate and merge-key mappings."""


def _construct_strict_yaml_mapping(
    loader: _StrictPolicyYamlLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge" or key_node.value == "<<":
            raise ValueError("policy pack YAML merge keys are not allowed")
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or not key:
            raise ValueError("policy pack YAML object keys must be non-empty strings")
        if key in result:
            raise ValueError(f"policy pack YAML object has duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictPolicyYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_strict_yaml_mapping,
)


def _construct_strict_yaml_bool(
    _loader: _StrictPolicyYamlLoader, node: yaml.ScalarNode
) -> bool:
    if node.value not in {"true", "false"}:
        raise ValueError("policy pack YAML booleans must use lowercase true or false")
    return str(node.value) == "true"


def _construct_strict_yaml_int(
    _loader: _StrictPolicyYamlLoader, node: yaml.ScalarNode
) -> int:
    if not _JSON_INTEGER_RE.fullmatch(node.value):
        raise ValueError("policy pack YAML integers must use canonical JSON syntax")
    return int(node.value)


def _construct_strict_yaml_float(
    _loader: _StrictPolicyYamlLoader, node: yaml.ScalarNode
) -> float:
    if not _JSON_FLOAT_RE.fullmatch(node.value):
        raise ValueError("policy pack YAML numbers must use finite JSON syntax")
    return float(node.value)


def _construct_strict_yaml_null(
    _loader: _StrictPolicyYamlLoader, node: yaml.ScalarNode
) -> None:
    if node.value != "null":
        raise ValueError("policy pack YAML nulls must use lowercase null")
    return None


for _tag, _constructor in (
    ("tag:yaml.org,2002:bool", _construct_strict_yaml_bool),
    ("tag:yaml.org,2002:int", _construct_strict_yaml_int),
    ("tag:yaml.org,2002:float", _construct_strict_yaml_float),
    ("tag:yaml.org,2002:null", _construct_strict_yaml_null),
):
    _StrictPolicyYamlLoader.add_constructor(_tag, _constructor)


def _load_structured_text(text: str, *, suffix: str) -> Any:
    try:
        if suffix.lower() in {".yaml", ".yml"}:
            if any(
                isinstance(
                    token,
                    (
                        yaml.tokens.AliasToken,
                        yaml.tokens.AnchorToken,
                        yaml.tokens.TagToken,
                    ),
                )
                for token in yaml.scan(text)
            ):
                raise ValueError(
                    "policy pack YAML aliases and explicit tags are not allowed"
                )
            return yaml.load(text, Loader=_StrictPolicyYamlLoader)
        return parse_json_bytes(text.encode("utf-8"), label="policy pack")
    except _STRUCTURED_TEXT_LOAD_ERRORS as exc:
        raise ValueError(
            f"policy pack could not be decoded as JSON/YAML: {exc}"
        ) from exc


def _load_structured_file_snapshot(
    path: Path, *, max_bytes: int | None = None
) -> tuple[bytes, Any]:
    try:
        payload = read_regular_file_bytes(
            path,
            label="policy pack",
            max_bytes=max_bytes,
        )
        text = payload.decode("utf-8")
    except StrictJsonError as exc:
        raise ValueError(str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise ValueError("policy pack could not be decoded as JSON/YAML") from exc
    return payload, _load_structured_text(text, suffix=path.suffix)


def _load_structured_file(path: Path) -> Any:
    return _load_structured_file_snapshot(path)[1]


def _normalize_overrides(overrides: Any) -> list[dict[str, Any]]:
    if overrides is None:
        return []
    if not isinstance(overrides, list) or any(
        not isinstance(item, dict) or set(item) != {"path", "value"}
        for item in overrides
    ):
        raise ValueError(
            "overrides must be an ordered list of exact path/value objects"
        )
    return [dict(item) for item in overrides]


def _compute_policy_pack_digest(policy: dict[str, Any]) -> str:
    """Return the full digest of a canonical policy authorization payload."""

    canonical = json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{POLICY_PACK_DIGEST_PREFIX}{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _json_value_errors(value: Any, *, path: str) -> list[str]:
    if value is None or isinstance(value, bool | int | str):
        return []
    if isinstance(value, float):
        return (
            []
            if math.isfinite(value)
            else [f"{path} must not contain non-finite numbers"]
        )
    if isinstance(value, list):
        return [
            error
            for index, item in enumerate(value)
            for error in _json_value_errors(item, path=f"{path}[{index}]")
        ]
    if isinstance(value, dict):
        errors: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                errors.append(f"{path} object keys must be non-empty strings")
                continue
            errors.extend(_json_value_errors(item, path=f"{path}.{key}"))
        return errors
    return [f"{path} contains unsupported value type {type(value).__name__}"]


def _ordered_string_list_errors(
    value: object,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
    canonical_order: tuple[str, ...] | None = None,
) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{path} must be a non-empty ordered list"]
    if any(not isinstance(item, str) or not item for item in value):
        return [f"{path} entries must be non-empty strings"]
    strings = list(value)
    errors: list[str] = []
    if len(strings) != len(set(strings)):
        errors.append(f"{path} entries must be unique")
    expected_order = (
        sorted(strings, key=canonical_order.index)
        if canonical_order is not None
        and all(item in canonical_order for item in strings)
        else sorted(strings)
    )
    if strings != expected_order:
        errors.append(f"{path} entries must use canonical sorted order")
    if allowed is not None and any(item not in allowed for item in strings):
        errors.append(f"{path} contains an unsupported value")
    return errors


def _unit_interval_errors(value: object, *, path: str) -> list[str]:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return [f"{path} must be a finite number in [0, 1]"]
    return []


def _behavioral_claim_errors(value: object) -> list[str]:
    path = "behavioral_claim"
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    fields = set(value)
    if fields != POLICY_PACK_BEHAVIORAL_CLAIM_FIELDS:
        return [
            f"{path} must contain exactly "
            + ", ".join(sorted(POLICY_PACK_BEHAVIORAL_CLAIM_FIELDS))
        ]

    errors: list[str] = []
    if value.get("claim_set") != RUNTIME_BEHAVIORAL_CLAIM_SET:
        errors.append(f"{path}.claim_set must be {RUNTIME_BEHAVIORAL_CLAIM_SET}")

    schedule_sha256 = value.get("schedule_sha256")
    if (
        not isinstance(schedule_sha256, str)
        or _SHA256_RE.fullmatch(schedule_sha256) is None
    ):
        errors.append(f"{path}.schedule_sha256 must be a lowercase sha256 digest")

    for role in ("baseline", "subject"):
        binding = value.get(role)
        binding_path = f"{path}.{role}"
        if not isinstance(binding, dict) or set(binding) != (
            POLICY_PACK_BEHAVIORAL_BINDING_FIELDS
        ):
            errors.append(
                f"{binding_path} must contain exactly "
                + ", ".join(sorted(POLICY_PACK_BEHAVIORAL_BINDING_FIELDS))
            )
            continue
        provider_name = binding.get("provider_name")
        if (
            not isinstance(provider_name, str)
            or _PROVIDER_NAME_RE.fullmatch(provider_name) is None
        ):
            errors.append(f"{binding_path}.provider_name must be canonical")
        if binding.get("artifact_format") not in RUNTIME_ARTIFACT_FORMATS:
            errors.append(f"{binding_path}.artifact_format is unsupported")
        for field in ("artifact_identity_sha256", "execution_settings_sha256"):
            digest = binding.get(field)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                errors.append(
                    f"{binding_path}.{field} must be a lowercase sha256 digest"
                )
        outer_image_digest = binding.get("outer_image_digest")
        if (
            not isinstance(outer_image_digest, str)
            or _DIGEST_RE.fullmatch(outer_image_digest) is None
        ):
            errors.append(
                f"{binding_path}.outer_image_digest must be a sha256 image digest"
            )

    capabilities = value.get("required_capabilities")
    required_metrics: object = None
    if not isinstance(capabilities, dict) or set(capabilities) != (
        POLICY_PACK_BEHAVIORAL_CAPABILITY_FIELDS
    ):
        errors.append(
            f"{path}.required_capabilities must contain exactly "
            + ", ".join(sorted(POLICY_PACK_BEHAVIORAL_CAPABILITY_FIELDS))
        )
    else:
        if capabilities.get("tasks") != ["text_causal"]:
            errors.append(
                f"{path}.required_capabilities.tasks must equal ['text_causal']"
            )
        required_metrics = capabilities.get("metrics")
        errors.extend(
            _ordered_string_list_errors(
                required_metrics,
                path=f"{path}.required_capabilities.metrics",
                allowed=RUNTIME_BEHAVIORAL_METRICS,
            )
        )
        surfaces = capabilities.get("evidence_surfaces")
        errors.extend(
            _ordered_string_list_errors(
                surfaces,
                path=f"{path}.required_capabilities.evidence_surfaces",
                allowed=frozenset({"behavior", "tokenizer", "build"}),
            )
        )
        if isinstance(surfaces, list) and not {
            "behavior",
            "tokenizer",
        }.issubset(surfaces):
            errors.append(
                f"{path}.required_capabilities.evidence_surfaces must require "
                "behavior and tokenizer"
            )

    metric_policy = value.get("metric_policy")
    if not isinstance(metric_policy, dict) or set(metric_policy) != (
        POLICY_PACK_BEHAVIORAL_METRIC_FIELDS
    ):
        errors.append(
            f"{path}.metric_policy must contain exactly "
            + ", ".join(sorted(POLICY_PACK_BEHAVIORAL_METRIC_FIELDS))
        )
    else:
        metric_kind = metric_policy.get("kind")
        if metric_kind not in RUNTIME_BEHAVIORAL_METRICS:
            errors.append(f"{path}.metric_policy.kind must be exact_match")
        elif isinstance(required_metrics, list) and metric_kind not in required_metrics:
            errors.append(
                f"{path}.metric_policy.kind must be listed in required metrics"
            )
        for field in ("minimum_subject_score", "maximum_regression"):
            errors.extend(
                _unit_interval_errors(
                    metric_policy.get(field), path=f"{path}.metric_policy.{field}"
                )
            )
    return errors


def _policy_pack_shape_errors(pack: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fields = set(pack)
    missing = POLICY_PACK_REQUIRED_FIELDS - fields
    unknown = fields - POLICY_PACK_REQUIRED_FIELDS - POLICY_PACK_OPTIONAL_FIELDS
    if missing:
        errors.append(
            "policy pack is missing required fields: " + ", ".join(sorted(missing))
        )
    if unknown:
        errors.append(
            "policy pack contains unknown fields: " + ", ".join(sorted(unknown))
        )
    pack_format = pack.get("format")
    if pack_format not in POLICY_PACK_FORMATS:
        errors.append(
            f"policy pack format must be {POLICY_PACK_FORMAT}, "
            f"{LEGACY_POLICY_PACK_FORMAT}, or {BEHAVIORAL_POLICY_PACK_FORMAT} "
            f"(found {pack_format!r})"
        )
    support_tiers = (
        LEGACY_POLICY_PACK_SUPPORT_TIERS
        if pack_format == LEGACY_POLICY_PACK_FORMAT
        else POLICY_PACK_SUPPORT_TIERS
    )
    support_tier_order = (
        LEGACY_POLICY_PACK_SUPPORT_TIER_ORDER
        if pack_format == LEGACY_POLICY_PACK_FORMAT
        else POLICY_PACK_SUPPORT_TIER_ORDER
    )
    tier = pack.get("tier")
    if tier not in POLICY_PACK_TIERS:
        errors.append("tier must be aggressive, balanced, or conservative")

    resolved_policy = pack.get("resolved_policy")
    if not isinstance(resolved_policy, dict):
        errors.append("resolved_policy must be an object")
    else:
        errors.extend(_json_value_errors(resolved_policy, path="resolved_policy"))
        raw_authority = resolved_policy.get("guard_authority")
        if pack_format == POLICY_PACK_FORMAT:
            errors.extend(
                guard_authority_errors(
                    raw_authority,
                    path="resolved_policy.guard_authority",
                )
            )
        elif "guard_authority" in resolved_policy:
            errors.append(
                f"{pack_format or 'policy pack'} cannot declare "
                "resolved_policy.guard_authority"
            )

    behavioral_claim = pack.get("behavioral_claim")
    if pack_format == BEHAVIORAL_POLICY_PACK_FORMAT:
        errors.extend(_behavioral_claim_errors(behavioral_claim))
    elif behavioral_claim is not None:
        errors.append("behavioral_claim is allowed only for policy-pack-v3")

    overrides = pack.get("overrides")
    if not isinstance(overrides, list):
        errors.append("overrides must be an ordered list")
    else:
        seen_paths: set[str] = set()
        for index, override in enumerate(overrides):
            path = f"overrides[{index}]"
            if not isinstance(override, dict) or set(override) != {"path", "value"}:
                errors.append(f"{path} must contain exactly path and value")
                continue
            override_path = override.get("path")
            if not isinstance(override_path, str) or not _OVERRIDE_PATH_RE.fullmatch(
                override_path
            ):
                errors.append(f"{path}.path must be a canonical dotted path")
            elif override_path in seen_paths:
                errors.append(f"{path}.path duplicates an earlier override")
            else:
                seen_paths.add(override_path)
            errors.extend(
                _json_value_errors(override.get("value"), path=f"{path}.value")
            )

    compatibility = pack.get("compatibility")
    if not isinstance(compatibility, dict):
        errors.append("compatibility must be an object")
    else:
        compatibility_fields = set(compatibility)
        unknown_compatibility = compatibility_fields - POLICY_PACK_COMPATIBILITY_FIELDS
        if unknown_compatibility:
            errors.append(
                "compatibility contains unknown fields: "
                + ", ".join(sorted(unknown_compatibility))
            )
        if "support_tiers" not in compatibility:
            errors.append("compatibility.support_tiers is required")
        else:
            errors.extend(
                _ordered_string_list_errors(
                    compatibility["support_tiers"],
                    path="compatibility.support_tiers",
                    allowed=support_tiers,
                    canonical_order=support_tier_order,
                )
            )
        for field in ("adapter_families", "runtime_lanes"):
            if field in compatibility:
                errors.extend(
                    _ordered_string_list_errors(
                        compatibility[field], path=f"compatibility.{field}"
                    )
                )
        if "dataset_identity" in compatibility:
            identity = compatibility["dataset_identity"]
            if not isinstance(identity, dict) or set(identity) != set(
                DATASET_IDENTITY_FIELDS
            ):
                errors.append(
                    "compatibility.dataset_identity must contain exactly "
                    + ", ".join(DATASET_IDENTITY_FIELDS)
                )
            else:
                for field in ("provider", "split"):
                    if not isinstance(identity[field], str) or not identity[field]:
                        errors.append(
                            f"compatibility.dataset_identity.{field} must be non-empty"
                        )
                for field in ("dataset_name", "config_name", "revision"):
                    value = identity[field]
                    if value is not None and (not isinstance(value, str) or not value):
                        errors.append(
                            f"compatibility.dataset_identity.{field} must be null or non-empty"
                        )
        elif pack_format == BEHAVIORAL_POLICY_PACK_FORMAT:
            errors.append(
                "compatibility.dataset_identity is required for policy-pack-v3"
            )

    approval = pack.get("approval")
    if approval is not None:
        if not isinstance(approval, dict) or not approval:
            errors.append("approval must be a non-empty object when present")
        else:
            unknown_approval = set(approval) - POLICY_PACK_APPROVAL_FIELDS
            if unknown_approval:
                errors.append(
                    "approval contains unknown fields: "
                    + ", ".join(sorted(unknown_approval))
                )
            for key, value in approval.items():
                if not isinstance(value, str) or not value:
                    errors.append(f"approval.{key} must be a non-empty string")

    metadata = pack.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            errors.append("metadata must be an object when present")
        else:
            errors.extend(_json_value_errors(metadata, path="metadata"))

    digest = pack.get("policy_digest")
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        errors.append("policy_digest must be a canonical sha256 digest")
    return errors


def compute_policy_pack_digest(
    *,
    tier: str,
    resolved_policy: dict[str, Any],
    overrides: list[dict[str, Any]],
    compatibility: dict[str, Any] | None = None,
    approval: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    if compatibility is not None and not isinstance(compatibility, dict):
        raise ValueError("compatibility must be an object")
    if approval is not None and not isinstance(approval, dict):
        raise ValueError("approval must be an object")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    compatibility_obj = (
        dict(compatibility)
        if isinstance(compatibility, dict)
        else {"support_tiers": ["maintained_catalog"]}
    )
    normalized_overrides = _normalize_overrides(overrides)
    resolved = copy.deepcopy(resolved_policy)
    resolved.setdefault("guard_authority", dict(DEFAULT_GUARD_AUTHORITY))
    digest_payload = {
        "format": POLICY_PACK_FORMAT,
        "tier": tier,
        "resolved_policy": resolved,
        "overrides": normalized_overrides,
        "compatibility": compatibility_obj,
    }
    if approval:
        digest_payload["approval"] = dict(approval)
    if metadata:
        digest_payload["metadata"] = dict(metadata)
    candidate = {**digest_payload, "policy_digest": "sha256:" + "0" * 64}
    errors = _policy_pack_shape_errors(candidate)
    if errors:
        raise ValueError("invalid policy pack digest input: " + "; ".join(errors))
    return _compute_policy_pack_digest(digest_payload)


def build_policy_pack(
    *,
    tier: str,
    resolved_policy: dict[str, Any],
    overrides: list[dict[str, Any]] | None = None,
    compatibility: dict[str, Any] | None = None,
    approval: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if compatibility is not None and not isinstance(compatibility, dict):
        raise ValueError("compatibility must be an object")
    if approval is not None and not isinstance(approval, dict):
        raise ValueError("approval must be an object")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    normalized_overrides = _normalize_overrides(overrides)
    compatibility_obj = dict(compatibility) if isinstance(compatibility, dict) else {}
    compatibility_obj.setdefault("support_tiers", ["maintained_catalog"])
    resolved = copy.deepcopy(resolved_policy)
    resolved.setdefault("guard_authority", dict(DEFAULT_GUARD_AUTHORITY))
    pack: dict[str, Any] = {
        "format": POLICY_PACK_FORMAT,
        "tier": tier,
        "resolved_policy": resolved,
        "overrides": normalized_overrides,
        "compatibility": compatibility_obj,
    }
    if isinstance(approval, dict) and approval:
        pack["approval"] = approval
    if isinstance(metadata, dict) and metadata:
        pack["metadata"] = metadata
    pack["policy_digest"] = _compute_policy_pack_digest(pack)
    errors = _policy_pack_shape_errors(pack)
    if errors:
        raise ValueError("invalid policy pack: " + "; ".join(errors))
    return pack


def build_behavioral_policy_pack(
    *,
    tier: str,
    schedule_sha256: str,
    baseline: dict[str, Any],
    subject: dict[str, Any],
    metric_kind: str,
    minimum_subject_score: float,
    maximum_regression: float,
    dataset_identity: dict[str, Any],
    required_evidence_surfaces: list[str] | None = None,
    approval: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v3 authorization for the narrow runtime-behavioral claim."""

    if approval is not None and not isinstance(approval, dict):
        raise ValueError("approval must be an object")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    pack: dict[str, Any] = {
        "format": BEHAVIORAL_POLICY_PACK_FORMAT,
        "tier": tier,
        "resolved_policy": {},
        "overrides": [],
        "compatibility": {
            "support_tiers": ["maintained_catalog"],
            "dataset_identity": copy.deepcopy(dataset_identity),
        },
        "behavioral_claim": {
            "claim_set": RUNTIME_BEHAVIORAL_CLAIM_SET,
            "schedule_sha256": schedule_sha256,
            "baseline": copy.deepcopy(baseline),
            "subject": copy.deepcopy(subject),
            "required_capabilities": {
                "tasks": ["text_causal"],
                "metrics": [metric_kind],
                "evidence_surfaces": list(
                    required_evidence_surfaces or ["behavior", "tokenizer"]
                ),
            },
            "metric_policy": {
                "kind": metric_kind,
                "minimum_subject_score": minimum_subject_score,
                "maximum_regression": maximum_regression,
            },
        },
    }
    if isinstance(approval, dict) and approval:
        pack["approval"] = copy.deepcopy(approval)
    if isinstance(metadata, dict) and metadata:
        pack["metadata"] = copy.deepcopy(metadata)
    pack["policy_digest"] = _compute_policy_pack_digest(pack)
    errors = _policy_pack_shape_errors(pack)
    if errors:
        raise ValueError("invalid behavioral policy pack: " + "; ".join(errors))
    return pack


def load_policy_pack(path: Path) -> dict[str, Any]:
    return read_policy_pack_snapshot(path)[1]


def read_policy_pack_snapshot(
    path: Path, *, max_bytes: int | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Read one finite policy-pack object from one regular-file snapshot."""

    raw, payload = _load_structured_file_snapshot(path, max_bytes=max_bytes)
    errors = _json_value_errors(payload, path="policy input")
    if errors:
        raise ValueError("; ".join(errors))
    if not isinstance(payload, dict):
        raise ValueError("policy pack must decode to a JSON/YAML object")
    return raw, payload


def load_policy_input(path: Path) -> Any:
    """Load one strict finite JSON/YAML policy input from a regular snapshot."""

    payload = _load_structured_file(path)
    errors = _json_value_errors(payload, path="policy input")
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def write_policy_pack(path: Path, pack: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pack, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_policy_pack(pack: object) -> list[str]:
    if not isinstance(pack, dict):
        return ["policy pack must be a mapping"]

    errors = _policy_pack_shape_errors(pack)

    schema = load_policy_pack_schema()
    if schema and jsonschema is not None:
        try:
            jsonschema.validate(instance=pack, schema=schema)
        except _JSONSCHEMA_VALIDATE_ERRORS as exc:
            errors.append(f"schema validation failed: {exc}")

    digest_payload = {
        key: value for key, value in pack.items() if key != "policy_digest"
    }
    try:
        expected_digest = _compute_policy_pack_digest(digest_payload)
    except (TypeError, ValueError):
        return errors
    observed_digest = pack.get("policy_digest")
    if observed_digest != expected_digest:
        errors.append(
            f"policy digest mismatch: observed={observed_digest!r} expected={expected_digest!r}"
        )
    return errors


def exercise_policy_pack_bytes(data: bytes) -> None:
    text = data.decode("utf-8", errors="ignore")
    try:
        payload = _load_structured_text(
            text, suffix=_choose_structured_fuzz_suffix(data)
        )
    except (
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ):
        return

    verify_policy_pack(payload)


__all__ = [
    "BEHAVIORAL_POLICY_PACK_FORMAT",
    "LEGACY_POLICY_PACK_FORMAT",
    "POLICY_PACK_FORMAT",
    "POLICY_PACK_DIGEST_PREFIX",
    "build_behavioral_policy_pack",
    "build_policy_pack",
    "compute_policy_pack_digest",
    "exercise_policy_pack_bytes",
    "load_policy_pack",
    "load_policy_input",
    "read_policy_pack_snapshot",
    "verify_policy_pack",
    "write_policy_pack",
]
