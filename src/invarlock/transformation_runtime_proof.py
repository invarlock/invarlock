"""Shared contract and storage checks for transformation runtime proofs.

The script-side runtime runner produces observations, while this package-owned
module owns the portable proof schema and the fail-closed checkpoint storage
audit.  Keeping these checks package-owned prevents evidence producers from
silently defining a weaker acceptance contract than downstream verifiers.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core.checkpoint_identity import (
    canonical_checkpoint_tree_digest,
    validated_model_identity,
)
from .evidence_pack_json import StrictJsonError, read_json_object_snapshot
from .pruning_contract import PRUNING_REPLAY_SCHEMA

RUNTIME_RELOAD_PROOF_SCHEMA = "invarlock/transformation-runtime-reload-proof-v1"
RUNTIME_LOAD_DIAGNOSTICS_SCHEMA = "invarlock/pretrained-load-diagnostics-v1"
RUNTIME_STORAGE_KEY_AUDIT_SCHEMA = "invarlock/safetensors-storage-key-audit-v1"
REPLAY_SCHEMAS: dict[str, frozenset[str]] = {
    "invarlock/generated-transformation-replay-v1": frozenset(
        {"quant_rtn", "synthetic_lowrank_delta", "synthetic_dense_update"}
    ),
    PRUNING_REPLAY_SCHEMA: frozenset({"magnitude_prune"}),
}
PROOF_KEYS = frozenset(
    {
        "schema",
        "ok",
        "replay_schema",
        "edit_type",
        "artifact_identity",
        "replay_artifact_identity",
        "prompt_sha256",
        "device",
        "input_device",
        "reload_runs",
        "token_ids_sha256",
        "token_ids_shape",
        "logits_sha256",
        "logits_shape",
        "all_logits_finite",
        "repeat_deterministic",
        "load_diagnostics",
        "storage_key_audit",
    }
)
_LOAD_DIAGNOSTIC_FIELDS = frozenset(
    {"unexpected_keys", "missing_keys", "mismatched_keys", "error_msgs"}
)
_LOAD_DIAGNOSTICS_FIELDS = frozenset({"schema", "reloads"})
_STORAGE_KEY_AUDIT_FIELDS = frozenset(
    {
        "artifact_storage_key_count",
        "artifact_storage_keys_sha256",
        "model_state_key_count",
        "model_state_keys_sha256",
        "unexpected_storage_keys",
    }
)
_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS = frozenset({"schema", "reloads"})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RuntimeReloadProofError(RuntimeError):
    """Raised when runtime evidence cannot be tied to one verified artifact."""


def strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Read one regular-file JSON object from a single strict snapshot."""

    try:
        _, payload = read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        raise RuntimeReloadProofError(f"{label} is not strict UTF-8 JSON") from exc
    return payload


def local_checkpoint_identity(value: object, *, label: str) -> dict[str, str]:
    identity = validated_model_identity(value)
    if identity is None or identity.get("kind") != "local_checkpoint_tree":
        raise RuntimeReloadProofError(f"{label} is invalid")
    return identity


def require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise RuntimeReloadProofError(f"{label} is missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeReloadProofError(f"{label} must be a regular file")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def storage_key_set_sha256(keys: set[str]) -> str:
    return sha256_bytes(
        json.dumps(sorted(keys), separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )


def artifact_storage_keys(artifact_dir: Path) -> set[str]:
    """Return the exact physical safetensors key set for a canonical layout."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency environment specific
        raise RuntimeReloadProofError(
            "runtime proof requires safetensors for storage-key auditing"
        ) from exc
    try:
        candidates = sorted(
            (
                path
                for path in artifact_dir.iterdir()
                if path.name.endswith(".safetensors")
            ),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise RuntimeReloadProofError(
            "artifact safetensors files are unavailable for storage-key auditing"
        ) from exc
    if not candidates:
        raise RuntimeReloadProofError(
            "artifact has no safetensors files for storage-key auditing"
        )
    index_path = artifact_dir / "model.safetensors.index.json"
    indexed_key_to_shard: dict[str, str] | None = None
    if index_path.exists() or index_path.is_symlink():
        index = strict_json_object(index_path, label="artifact safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise RuntimeReloadProofError(
                "artifact safetensors index has no weight map"
            )
        indexed_key_to_shard = {}
        for key, shard in weight_map.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or not isinstance(shard, str)
                or not shard
                or shard != Path(shard).name
                or not shard.endswith(".safetensors")
            ):
                raise RuntimeReloadProofError(
                    "artifact safetensors index has invalid weight-map entries"
                )
            indexed_key_to_shard[key] = shard
        if {path.name for path in candidates} != set(indexed_key_to_shard.values()):
            raise RuntimeReloadProofError(
                "artifact safetensors files do not exactly match the index"
            )
    elif [path.name for path in candidates] != ["model.safetensors"]:
        raise RuntimeReloadProofError(
            "unindexed artifact safetensors layout is not canonical"
        )

    keys: set[str] = set()
    key_to_shard: dict[str, str] = {}
    for path in candidates:
        require_regular_file(path, label="artifact safetensors shard")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                shard_keys = list(handle.keys())
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeReloadProofError(
                "artifact safetensors shard cannot be audited"
            ) from exc
        if not shard_keys or any(
            not isinstance(key, str) or not key or key != key.strip()
            for key in shard_keys
        ):
            raise RuntimeReloadProofError(
                "artifact safetensors shard has invalid storage keys"
            )
        duplicate = keys.intersection(shard_keys)
        if duplicate:
            raise RuntimeReloadProofError(
                "artifact safetensors shards contain duplicate storage keys"
            )
        keys.update(shard_keys)
        key_to_shard.update(dict.fromkeys(shard_keys, path.name))
    if indexed_key_to_shard is not None and key_to_shard != indexed_key_to_shard:
        raise RuntimeReloadProofError(
            "artifact safetensors keys do not exactly match the index"
        )
    return keys


def storage_key_audit(artifact_dir: Path, *, model: Any) -> dict[str, object]:
    """Reject physical checkpoint keys hidden by a model loader's ignore policy."""

    storage_keys = artifact_storage_keys(artifact_dir)
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeReloadProofError("loaded model does not expose state_dict")
    try:
        model_state = state_dict()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeReloadProofError(
            "loaded model state keys are unavailable for storage-key auditing"
        ) from exc
    if not isinstance(model_state, Mapping):
        raise RuntimeReloadProofError(
            "loaded model state keys are unavailable for storage-key auditing"
        )
    model_keys = set(model_state)
    if not model_keys or any(
        not isinstance(key, str) or not key or key != key.strip() for key in model_keys
    ):
        raise RuntimeReloadProofError("loaded model state keys are invalid")
    unexpected = sorted(storage_keys - model_keys)
    if unexpected:
        raise RuntimeReloadProofError(
            "artifact safetensors contains keys absent from loaded model state"
        )
    return {
        "artifact_storage_key_count": len(storage_keys),
        "artifact_storage_keys_sha256": storage_key_set_sha256(storage_keys),
        "model_state_key_count": len(model_keys),
        "model_state_keys_sha256": storage_key_set_sha256(model_keys),
        "unexpected_storage_keys": [],
    }


def _is_sha256(value: object) -> bool:
    return canonical_checkpoint_tree_digest(value) is not None


def _valid_shape(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
    )


def _valid_runtime_device(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", value) is not None
    )


def _validate_clean_load_diagnostics(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _LOAD_DIAGNOSTICS_FIELDS:
        raise RuntimeReloadProofError("runtime proof load diagnostics are invalid")
    if value.get("schema") != RUNTIME_LOAD_DIAGNOSTICS_SCHEMA:
        raise RuntimeReloadProofError(
            "runtime proof load diagnostics schema is invalid"
        )
    reloads = value.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        raise RuntimeReloadProofError(
            "runtime proof load diagnostics must bind both reloads"
        )
    for index, diagnostic in enumerate(reloads):
        if (
            not isinstance(diagnostic, Mapping)
            or set(diagnostic) != _LOAD_DIAGNOSTIC_FIELDS
        ):
            raise RuntimeReloadProofError(
                f"runtime proof load diagnostics reload {index} is invalid"
            )
        for field in _LOAD_DIAGNOSTIC_FIELDS:
            entries = diagnostic.get(field)
            if not isinstance(entries, list) or entries:
                raise RuntimeReloadProofError(
                    f"runtime proof load diagnostics reload {index} reports {field}"
                )


def _validate_storage_key_audit(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _STORAGE_KEY_AUDIT_ENVELOPE_FIELDS
        or value.get("schema") != RUNTIME_STORAGE_KEY_AUDIT_SCHEMA
    ):
        raise RuntimeReloadProofError("runtime proof storage-key audit is invalid")
    reloads = value.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        raise RuntimeReloadProofError(
            "runtime proof storage-key audit must bind both reloads"
        )
    expected: dict[str, object] | None = None
    for index, audit in enumerate(reloads):
        if not isinstance(audit, Mapping) or set(audit) != _STORAGE_KEY_AUDIT_FIELDS:
            raise RuntimeReloadProofError(
                f"runtime proof storage-key audit reload {index} is invalid"
            )
        artifact_count = audit.get("artifact_storage_key_count")
        model_count = audit.get("model_state_key_count")
        for field, count in (
            ("artifact_storage_key_count", artifact_count),
            ("model_state_key_count", model_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RuntimeReloadProofError(
                    f"runtime proof storage-key audit reload {index} {field} is invalid"
                )
        assert isinstance(artifact_count, int)
        assert isinstance(model_count, int)
        if artifact_count > model_count:
            raise RuntimeReloadProofError(
                "runtime proof storage-key audit "
                f"reload {index} has more artifact storage keys than model state keys"
            )
        for field in ("artifact_storage_keys_sha256", "model_state_keys_sha256"):
            digest = audit.get(field)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise RuntimeReloadProofError(
                    f"runtime proof storage-key audit reload {index} {field} is invalid"
                )
        if audit.get("unexpected_storage_keys") != []:
            raise RuntimeReloadProofError(
                f"runtime proof storage-key audit reload {index} has unexpected storage keys"
            )
        normalized = dict(audit)
        if expected is None:
            expected = normalized
        elif normalized != expected:
            raise RuntimeReloadProofError(
                "runtime proof storage-key audits do not agree across reloads"
            )


def validate_proof_payload(proof: Mapping[str, object]) -> None:
    """Validate the exact portable runtime proof envelope."""

    if set(proof) != PROOF_KEYS:
        raise RuntimeReloadProofError("runtime proof has an unexpected schema")
    if (
        proof.get("schema") != RUNTIME_RELOAD_PROOF_SCHEMA
        or proof.get("ok") is not True
    ):
        raise RuntimeReloadProofError("runtime proof is not successful")
    replay_schema = proof.get("replay_schema")
    edit_type = proof.get("edit_type")
    if (
        not isinstance(replay_schema, str)
        or not isinstance(edit_type, str)
        or edit_type not in REPLAY_SCHEMAS.get(replay_schema, frozenset())
    ):
        raise RuntimeReloadProofError("runtime proof edit type is invalid")
    artifact_identity = local_checkpoint_identity(
        proof.get("artifact_identity"), label="runtime proof artifact identity"
    )
    replay_identity = local_checkpoint_identity(
        proof.get("replay_artifact_identity"),
        label="runtime proof replay artifact identity",
    )
    if artifact_identity != replay_identity:
        raise RuntimeReloadProofError("runtime proof identities do not match")
    if not all(
        _is_sha256(proof.get(field))
        for field in ("prompt_sha256", "token_ids_sha256", "logits_sha256")
    ):
        raise RuntimeReloadProofError("runtime proof digest is invalid")
    if not _valid_runtime_device(proof.get("device")) or not _valid_runtime_device(
        proof.get("input_device")
    ):
        raise RuntimeReloadProofError("runtime proof device is invalid")
    if proof.get("reload_runs") != 2:
        raise RuntimeReloadProofError("runtime proof reload count is invalid")
    if not _valid_shape(proof.get("token_ids_shape")) or not _valid_shape(
        proof.get("logits_shape")
    ):
        raise RuntimeReloadProofError("runtime proof tensor shape is invalid")
    if (
        proof.get("all_logits_finite") is not True
        or proof.get("repeat_deterministic") is not True
    ):
        raise RuntimeReloadProofError("runtime proof result is invalid")
    _validate_clean_load_diagnostics(proof.get("load_diagnostics"))
    _validate_storage_key_audit(proof.get("storage_key_audit"))


__all__ = [
    "PROOF_KEYS",
    "REPLAY_SCHEMAS",
    "RUNTIME_LOAD_DIAGNOSTICS_SCHEMA",
    "RUNTIME_RELOAD_PROOF_SCHEMA",
    "RUNTIME_STORAGE_KEY_AUDIT_SCHEMA",
    "RuntimeReloadProofError",
    "artifact_storage_keys",
    "local_checkpoint_identity",
    "require_regular_file",
    "sha256_bytes",
    "storage_key_audit",
    "strict_json_object",
    "validate_proof_payload",
]
