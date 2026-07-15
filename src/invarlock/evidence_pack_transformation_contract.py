"""Generated-transformation canonicalization and runtime-proof contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_edit_common import (
    _MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS,
    _MAX_SYNTHETIC_LOWRANK_RANK,
    _RUNTIME_LOAD_DIAGNOSTIC_FIELDS,
    _RUNTIME_LOAD_DIAGNOSTICS_FIELDS,
    _RUNTIME_RELOAD_PROOF_FIELDS,
    _RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS,
    _RUNTIME_STORAGE_KEY_AUDIT_FIELDS,
    _SHA256_RE,
    _TRANSFORMATION_ALGORITHMS,
    _VERIFIER_GRADE_TRANSFORMATION_EDIT_TYPES,
    RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
    RUNTIME_RELOAD_PROOF_SCHEMA,
    RUNTIME_RELOAD_PROOF_SIDECAR,
    RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_PARAMETERS_SCHEMA,
    _load_json_sidecar,
)


def _canonical_json_sha256(payload: object) -> str | None:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_exact_json_value(actual: object, expected: object) -> bool:
    """Compare JSON values without accepting a non-canonical numeric spelling."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _is_exact_json_value(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _is_exact_json_value(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _canonical_transformation_parameters(
    edit_type: object,
    parameters: object,
) -> tuple[dict[str, int | float] | None, str | None]:
    """Return the one accepted parameter representation for a generated edit."""

    if not isinstance(edit_type, str) or edit_type not in _TRANSFORMATION_ALGORITHMS:
        return None, "has no verifier-grade generated-lane contract"
    if not isinstance(parameters, dict):
        return None, "parameters must be a JSON object"

    def positive_int(value: object, *, field: str) -> tuple[int | None, str | None]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None, f"{field} must be a positive integer"
        return value, None

    def positive_float(value: object, *, field: str) -> tuple[float | None, str | None]:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None, f"{field} must be a finite positive number"
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0.0:
            return None, f"{field} must be a finite positive number"
        return normalized, None

    if edit_type == "quant_rtn":
        if set(parameters) != {"bits", "group_size"}:
            return (
                None,
                "quant_rtn parameters must contain exactly ['bits', 'group_size']",
            )
        bits, error = positive_int(parameters.get("bits"), field="quant_rtn.bits")
        if error is not None:
            return None, error
        assert bits is not None
        if not 2 <= bits <= 8:
            return None, "quant_rtn.bits must be in [2, 8]"
        group_size, error = positive_int(
            parameters.get("group_size"), field="quant_rtn.group_size"
        )
        if error is not None:
            return None, error
        assert group_size is not None
        return {"bits": bits, "group_size": group_size}, None

    if edit_type == "synthetic_lowrank_delta":
        if set(parameters) != {"rank", "scale"}:
            return None, (
                "synthetic_lowrank_delta parameters must contain exactly ['rank', 'scale']"
            )
        rank, error = positive_int(
            parameters.get("rank"), field="synthetic_lowrank_delta.rank"
        )
        if error is not None:
            return None, error
        if rank is not None and rank > _MAX_SYNTHETIC_LOWRANK_RANK:
            return (
                None,
                "synthetic_lowrank_delta.rank must not exceed "
                f"{_MAX_SYNTHETIC_LOWRANK_RANK}",
            )
        scale, error = positive_float(
            parameters.get("scale"), field="synthetic_lowrank_delta.scale"
        )
        if error is not None:
            return None, error
        assert rank is not None and scale is not None
        return {"rank": rank, "scale": scale}, None

    if edit_type == "synthetic_dense_update":
        if set(parameters) != {"step_size", "iterations"}:
            return None, (
                "synthetic_dense_update parameters must contain exactly "
                "['iterations', 'step_size']"
            )
        step_size, error = positive_float(
            parameters.get("step_size"), field="synthetic_dense_update.step_size"
        )
        if error is not None:
            return None, error
        iterations, error = positive_int(
            parameters.get("iterations"),
            field="synthetic_dense_update.iterations",
        )
        if error is not None:
            return None, error
        if (
            iterations is not None
            and iterations > _MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS
        ):
            return (
                None,
                "synthetic_dense_update.iterations must not exceed "
                f"{_MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS}",
            )
        assert step_size is not None and iterations is not None
        return {"step_size": step_size, "iterations": iterations}, None

    raise AssertionError(f"unhandled generated transformation: {edit_type}")


def _canonical_transformation_spec(
    edit_type: object,
    parameters: object,
) -> tuple[dict[str, object] | None, str | None]:
    canonical_parameters, error = _canonical_transformation_parameters(
        edit_type, parameters
    )
    if error is not None or canonical_parameters is None:
        return None, error or "transformation parameters are invalid"
    assert isinstance(edit_type, str)
    return {
        "schema": TRANSFORMATION_PARAMETERS_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": edit_type,
        "algorithm": _TRANSFORMATION_ALGORITHMS[edit_type],
        "parameters": canonical_parameters,
    }, None


def _canonical_transformation_scope(value: object) -> tuple[str | None, str | None]:
    """Parse the public scope grammar without a permissive compatibility path."""

    if not isinstance(value, str):
        return None, "transformation scope must be a string"
    text = value.strip()
    if not text or text.count("@") > 1:
        return None, "transformation scope syntax is invalid"
    raw_base, raw_qualifiers = text.split("@", 1) if "@" in text else (text, "")
    base = raw_base.strip().lower()
    if base not in {"ffn", "attn", "all"}:
        return None, "transformation scope base is invalid"
    if "@" not in text:
        return base, None
    if not raw_qualifiers.strip():
        return None, "transformation scope qualifier is invalid"
    values: dict[str, int] = {}
    for raw_item in raw_qualifiers.split(","):
        item = raw_item.strip()
        if not item or item.count("=") != 1:
            return None, "transformation scope qualifier is invalid"
        raw_name, raw_number = (part.strip() for part in item.split("=", 1))
        name = raw_name.lower()
        if name not in {"layers", "layer"} or name in values:
            return None, "transformation scope qualifier is invalid"
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_number) is None:
            return None, "transformation scope qualifier is invalid"
        number = int(raw_number)
        if name == "layers" and number == 0:
            return None, "layers qualifier must be greater than zero"
        values[name] = number
    layer_limit = values.get("layers")
    layer = values.get("layer")
    if layer_limit is not None and layer is not None and layer >= layer_limit:
        return None, "layer qualifier must be smaller than the layers qualifier"
    qualifiers: list[str] = []
    if layer_limit is not None:
        qualifiers.append(f"layers={layer_limit}")
    if layer is not None:
        qualifiers.append(f"layer={layer}")
    return base + "@" + ",".join(qualifiers), None


def _expected_literal_transformation(
    spec: dict[str, Any],
) -> tuple[dict[str, object] | None, str | None, str | None]:
    """Decode a non-clean scenario edit_spec into exact public parameters."""

    generation = spec.get("generation")
    edit_spec = generation.get("edit_spec") if isinstance(generation, dict) else ""
    if not isinstance(edit_spec, str) or not edit_spec:
        return None, None, "generated transformation scenario edit_spec is missing"
    parts = edit_spec.split(":")
    edit_type = parts[0]
    if edit_type not in _VERIFIER_GRADE_TRANSFORMATION_EDIT_TYPES:
        return None, None, "generated transformation scenario is unsupported"
    if len(parts) >= 2 and parts[1] == "clean":
        if len(parts) != 2:
            return None, None, "clean generated transformation edit_spec is invalid"
        return None, None, None
    if len(parts) != 4:
        return None, None, "generated transformation edit_spec has the wrong arity"

    try:
        if edit_type == "quant_rtn":
            parameters: dict[str, object] = {
                "bits": int(parts[1]),
                "group_size": int(parts[2]),
            }
        elif edit_type == "synthetic_lowrank_delta":
            parameters = {"rank": int(parts[1]), "scale": float(parts[2])}
        elif edit_type == "synthetic_dense_update":
            parameters = {
                "step_size": float(parts[1]),
                "iterations": int(parts[2]),
            }
        else:  # pragma: no cover - guarded by the supported set above
            raise AssertionError(f"unhandled generated transformation: {edit_type}")
    except (TypeError, ValueError, OverflowError):
        return None, None, "generated transformation edit_spec has invalid parameters"
    canonical_spec, parameter_error = _canonical_transformation_spec(
        edit_type, parameters
    )
    if canonical_spec is None:
        return (
            None,
            None,
            parameter_error or "generated transformation parameters invalid",
        )
    canonical_scope, scope_error = _canonical_transformation_scope(parts[3])
    if canonical_scope is None or parts[3] != canonical_scope:
        return (
            None,
            None,
            scope_error or "generated transformation scenario scope is not canonical",
        )
    return canonical_spec, canonical_scope, None


def _is_clean_transformation_scenario(spec: dict[str, Any] | None) -> bool:
    if not isinstance(spec, dict):
        return False
    generation = spec.get("generation")
    edit_spec = generation.get("edit_spec") if isinstance(generation, dict) else ""
    if not isinstance(edit_spec, str):
        return False
    parts = edit_spec.split(":")
    return (
        len(parts) >= 2
        and parts[0] in _VERIFIER_GRADE_TRANSFORMATION_EDIT_TYPES
        and parts[1] == "clean"
    )


def _transformation_identity_errors(
    *,
    prefix: str,
    label: str,
    value: object,
) -> list[str]:
    if not isinstance(value, dict):
        return [prefix + f"transformation replay {label} must be an object"]
    kind = value.get("kind")
    digest = value.get("sha256")
    if not isinstance(kind, str) or not kind:
        return [
            prefix + f"transformation replay {label}.kind must be a non-empty string"
        ]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        return [
            prefix + f"transformation replay {label}.sha256 must be a sha256 digest"
        ]
    return []


def _runtime_reload_identity_errors(
    *,
    prefix: str,
    label: str,
    value: object,
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"kind", "sha256"}:
        return [prefix + f"runtime reload proof {label} must be a local identity"]
    if value.get("kind") != "local_checkpoint_tree":
        return [
            prefix + f"runtime reload proof {label}.kind must be local_checkpoint_tree"
        ]
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        return [prefix + f"runtime reload proof {label}.sha256 must be a sha256 digest"]
    return []


def _runtime_load_diagnostics_errors(*, prefix: str, value: object) -> list[str]:
    if (
        not isinstance(value, dict)
        or set(value) != _RUNTIME_LOAD_DIAGNOSTICS_FIELDS
        or value.get("schema") != RUNTIME_LOAD_DIAGNOSTICS_SCHEMA
    ):
        return [prefix + "runtime reload proof load diagnostics are invalid"]
    reloads = value.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        return [
            prefix
            + "runtime reload proof load diagnostics must bind exactly two reloads"
        ]
    errors: list[str] = []
    for index, diagnostic in enumerate(reloads):
        label = f"runtime reload proof load diagnostics reload {index}"
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != _RUNTIME_LOAD_DIAGNOSTIC_FIELDS
        ):
            errors.append(prefix + label + " is invalid")
            continue
        for field in _RUNTIME_LOAD_DIAGNOSTIC_FIELDS:
            entries = diagnostic.get(field)
            if not isinstance(entries, list) or entries:
                errors.append(prefix + label + f" reports {field}")
    return errors


def _runtime_storage_key_audit_errors(*, prefix: str, value: object) -> list[str]:
    if (
        not isinstance(value, dict)
        or set(value) != _RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS
        or value.get("schema") != RUNTIME_STORAGE_KEY_AUDIT_SCHEMA
    ):
        return [prefix + "runtime reload proof storage-key audit is invalid"]
    reloads = value.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        return [
            prefix
            + "runtime reload proof storage-key audit must bind exactly two reloads"
        ]
    errors: list[str] = []
    canonical: dict[str, object] | None = None
    for index, audit in enumerate(reloads):
        label = f"runtime reload proof storage-key audit reload {index}"
        if (
            not isinstance(audit, dict)
            or set(audit) != _RUNTIME_STORAGE_KEY_AUDIT_FIELDS
        ):
            errors.append(prefix + label + " is invalid")
            continue
        artifact_storage_key_count = audit.get("artifact_storage_key_count")
        model_state_key_count = audit.get("model_state_key_count")
        for field, count in (
            ("artifact_storage_key_count", artifact_storage_key_count),
            ("model_state_key_count", model_state_key_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                errors.append(prefix + label + f" {field} is invalid")
        if (
            isinstance(artifact_storage_key_count, int)
            and not isinstance(artifact_storage_key_count, bool)
            and isinstance(model_state_key_count, int)
            and not isinstance(model_state_key_count, bool)
            and artifact_storage_key_count > model_state_key_count
        ):
            errors.append(
                prefix + label + " has more artifact storage keys than model state keys"
            )
        for field in (
            "artifact_storage_keys_sha256",
            "model_state_keys_sha256",
        ):
            digest = audit.get(field)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                errors.append(prefix + label + f" {field} is invalid")
        if audit.get("unexpected_storage_keys") != []:
            errors.append(prefix + label + " has unexpected storage keys")
        if canonical is None:
            canonical = dict(audit)
        elif audit != canonical:
            errors.append(prefix + "runtime reload proof storage-key audits disagree")
    return errors


def _runtime_reload_proof_errors(
    *,
    scenario_id: str,
    report: dict[str, Any],
    replay: dict[str, Any],
    proof: dict[str, Any],
    expected_edit_type: str,
) -> list[str]:
    """Bind a two-reload runtime proof to the replay and evaluated subject."""

    prefix = f"{scenario_id}: "
    errors: list[str] = []
    if set(proof) != _RUNTIME_RELOAD_PROOF_FIELDS:
        errors.append(prefix + "runtime reload proof has unbound fields")
    if proof.get("schema") != RUNTIME_RELOAD_PROOF_SCHEMA:
        errors.append(prefix + "runtime reload proof has unrecognized schema")
    if proof.get("ok") is not True:
        errors.append(prefix + "runtime reload proof did not pass")
    replay_schema = replay.get("schema")
    if proof.get("replay_schema") != replay_schema:
        errors.append(prefix + "runtime reload proof replay schema mismatch")
    if proof.get("edit_type") != replay.get("edit_type"):
        errors.append(prefix + "runtime reload proof replay edit type mismatch")
    if proof.get("edit_type") != expected_edit_type:
        errors.append(prefix + "runtime reload proof scenario edit type mismatch")

    replay_identity = replay.get("artifact_identity")
    errors.extend(
        _runtime_reload_identity_errors(
            prefix=prefix, label="replay artifact identity", value=replay_identity
        )
    )
    for label, identity in (
        ("artifact_identity", proof.get("artifact_identity")),
        ("replay_artifact_identity", proof.get("replay_artifact_identity")),
    ):
        errors.extend(
            _runtime_reload_identity_errors(prefix=prefix, label=label, value=identity)
        )
        if identity != replay_identity:
            errors.append(prefix + f"runtime reload proof {label} mismatch")

    report_identity = (
        (report.get("meta") or {}).get("model_identity")
        if isinstance(report.get("meta"), dict)
        else None
    )
    errors.extend(
        _runtime_reload_identity_errors(
            prefix=prefix, label="evaluation subject identity", value=report_identity
        )
    )
    if report_identity != replay_identity:
        errors.append(
            prefix + "runtime reload proof evaluation subject identity mismatch"
        )

    for field in ("prompt_sha256", "token_ids_sha256", "logits_sha256"):
        value = proof.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            errors.append(
                prefix + f"runtime reload proof {field} must be a sha256 digest"
            )
    device = proof.get("device")
    if device not in {"cpu", "cuda"}:
        errors.append(prefix + "runtime reload proof device is invalid")
    input_device = proof.get("input_device")
    if (
        not isinstance(input_device, str)
        or re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", input_device) is None
    ):
        errors.append(prefix + "runtime reload proof input device is invalid")
    elif device == "cpu" and input_device != "cpu":
        errors.append(prefix + "runtime reload proof input device mismatches CPU run")
    elif device == "cuda" and not input_device.startswith("cuda"):
        errors.append(prefix + "runtime reload proof input device mismatches CUDA run")
    if proof.get("reload_runs") != 2:
        errors.append(prefix + "runtime reload proof must record exactly two reloads")
    for field in ("token_ids_shape", "logits_shape"):
        shape = proof.get(field)
        if (
            not isinstance(shape, list)
            or not shape
            or not all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in shape
            )
        ):
            errors.append(prefix + f"runtime reload proof {field} is invalid")
    if proof.get("all_logits_finite") is not True:
        errors.append(prefix + "runtime reload proof finite logits evidence missing")
    if proof.get("repeat_deterministic") is not True:
        errors.append(prefix + "runtime reload proof determinism evidence missing")
    errors.extend(
        _runtime_load_diagnostics_errors(
            prefix=prefix, value=proof.get("load_diagnostics")
        )
    )
    errors.extend(
        _runtime_storage_key_audit_errors(
            prefix=prefix, value=proof.get("storage_key_audit")
        )
    )
    return errors


def _require_runtime_reload_proof(
    *,
    scenario_id: str,
    report_dir: Path,
    report: dict[str, Any],
    replay: dict[str, Any],
    expected_edit_type: str,
) -> list[str]:
    proof_path = report_dir / RUNTIME_RELOAD_PROOF_SIDECAR
    prefix = f"{scenario_id}: "
    if not proof_path.is_file() or proof_path.is_symlink():
        return [prefix + "runtime reload proof sidecar missing"]
    proof, proof_error = _load_json_sidecar(proof_path)
    if proof_error is not None or proof is None:
        return [prefix + "runtime reload proof sidecar is invalid"]
    return _runtime_reload_proof_errors(
        scenario_id=scenario_id,
        report=report,
        replay=replay,
        proof=proof,
        expected_edit_type=expected_edit_type,
    )
