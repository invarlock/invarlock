"""Immutable training-profile contracts for real evidence-pack edits."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.training_protocol import TRAINING_PROFILES_SCHEMA

DEFAULT_TRAINING_PROFILES_PATH = (
    Path(__file__).resolve().parents[2] / "training_profiles.json"
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_REVISION_RE = re.compile(r"^[a-f0-9]{40}$")
_EDIT_TYPES = {"fine_tune", "lora_merge"}
_OPTIMIZERS = {"adamw"}
_DEVICES = {"cpu", "cuda", "mps"}
_DTYPES = {"bfloat16", "float16", "float32"}
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:\+[A-Za-z0-9][A-Za-z0-9._-]*)?$")

_PROFILE_KEYS = set(
    "profile_sha256 edit_type model_id model_revision training_data optimizer "
    "steps micro_batch_size gradient_accumulation_steps max_sequence_length "
    "seed deterministic_algorithms device dtype toolchain model_load lora".split()
)
_DATA_KEYS = {"path", "sha256", "rows", "text_field"}
_OPTIMIZER_KEYS = {"name", "learning_rate", "betas", "eps", "weight_decay"}
_LORA_KEYS = set(
    "rank alpha dropout target_modules bias task_type fan_in_fan_out".split()
)
_TOOLCHAIN_KEYS = {"python", "torch", "transformers", "peft"}
_MODEL_LOAD_KEYS = {"loss_function", "expected_unexpected_keys"}


class TrainingProfileError(ValueError):
    """Raised when an immutable training profile is invalid or tampered."""


@dataclass(frozen=True)
class TrainingDataSpec:
    path: str
    sha256: str
    rows: int
    text_field: str

    def resolve(self, repo_root: Path = _REPO_ROOT) -> Path:
        return _resolve_repo_path(self.path, repo_root=repo_root)


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str
    task_type: str
    fan_in_fan_out: bool


@dataclass(frozen=True)
class TrainingToolchainSpec:
    python: str
    torch: str
    transformers: str
    peft: str | None = None


@dataclass(frozen=True)
class ModelLoadSpec:
    """Pinned baseline-load migration and labeled-forward semantics."""

    loss_function: str
    expected_unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class BaseTrainingProfile:
    profile_id: str
    profile_sha256: str
    edit_type: str
    model_id: str
    model_revision: str
    training_data: TrainingDataSpec
    optimizer: OptimizerSpec
    steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int
    seed: int
    deterministic_algorithms: bool
    device: str
    dtype: str
    toolchain: TrainingToolchainSpec
    model_load: ModelLoadSpec


@dataclass(frozen=True)
class FineTuneTrainingProfile(BaseTrainingProfile):
    """Typed full-parameter fine-tuning profile."""


@dataclass(frozen=True)
class LoraTrainingProfile(BaseTrainingProfile):
    """Typed PEFT LoRA train-and-merge profile."""

    lora: LoraSpec


type TrainingProfile = FineTuneTrainingProfile | LoraTrainingProfile


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON encoding used by profile and receipt hashes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def canonical_profile_digest(profile: Mapping[str, Any]) -> str:
    payload = dict(profile)
    payload.pop("profile_sha256", None)
    return canonical_sha256(payload)


def lora_config_digest(lora: LoraSpec | Mapping[str, Any]) -> str:
    payload = asdict(lora) if isinstance(lora, LoraSpec) else dict(lora)
    return canonical_sha256(payload)


def _resolve_repo_path(value: str, *, repo_root: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise TrainingProfileError("training_data.path must be repository-relative")
    root = repo_root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TrainingProfileError(
            "training_data.path resolves outside the repository"
        ) from exc
    return resolved


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _unknown_keys(
    errors: list[str], value: Mapping[str, Any], *, allowed: set[str], path: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{path} contains unsupported field(s): {', '.join(unknown)}")


def _validate_training_data(
    value: Any,
    *,
    repo_root: Path,
    verify_file: bool,
) -> list[str]:
    path = "training_data"
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    errors: list[str] = []
    _unknown_keys(errors, value, allowed=_DATA_KEYS, path=path)

    relative_path = value.get("path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        errors.append(f"{path}.path must be a non-empty repository-relative path")
        resolved = None
    else:
        try:
            resolved = _resolve_repo_path(relative_path, repo_root=repo_root)
        except TrainingProfileError as exc:
            errors.append(str(exc))
            resolved = None

    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        errors.append(f"{path}.sha256 must be a canonical sha256 digest")

    rows = value.get("rows")
    if not _is_int(rows, minimum=1):
        errors.append(f"{path}.rows must be a positive integer")

    text_field = value.get("text_field")
    if not isinstance(text_field, str) or not text_field.strip():
        errors.append(f"{path}.text_field must be a non-empty string")

    if verify_file and resolved is not None:
        if not resolved.is_file():
            errors.append(f"{path}.path does not exist: {relative_path}")
        else:
            observed_rows = 0
            try:
                raw = read_regular_file_bytes(resolved, label=f"{path}.path")
                observed_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                if isinstance(digest, str) and observed_digest != digest:
                    errors.append(
                        f"{path}.sha256 does not match vendored training data"
                    )
                for line_number, line in enumerate(raw.splitlines(), start=1):
                    if not line.strip():
                        errors.append(
                            f"{path}.path contains a blank row at line {line_number}"
                        )
                        continue
                    row = parse_json_bytes(
                        line, label=f"{path}.path line {line_number}"
                    )
                    if not isinstance(row, dict):
                        errors.append(
                            f"{path}.path line {line_number} must be a JSON object"
                        )
                    elif isinstance(text_field, str) and (
                        not isinstance(row.get(text_field), str)
                        or not str(row.get(text_field)).strip()
                    ):
                        errors.append(
                            f"{path}.path line {line_number} lacks non-empty "
                            f"{text_field!r} text"
                        )
                    observed_rows += 1
            except StrictJsonError as exc:
                errors.append(f"{path}.path is not valid UTF-8 JSONL: {exc}")
            if _is_int(rows, minimum=1) and observed_rows != rows:
                errors.append(
                    f"{path}.rows={rows} does not match observed rows={observed_rows}"
                )
    return errors


def _validate_optimizer(value: Any) -> list[str]:
    path = "optimizer"
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    errors: list[str] = []
    _unknown_keys(errors, value, allowed=_OPTIMIZER_KEYS, path=path)
    if value.get("name") not in _OPTIMIZERS:
        errors.append(f"{path}.name must be one of: {', '.join(sorted(_OPTIMIZERS))}")
    learning_rate = _finite_float(value.get("learning_rate"))
    if learning_rate is None or learning_rate <= 0.0:
        errors.append(f"{path}.learning_rate must be finite and positive")
    betas = value.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        errors.append(f"{path}.betas must contain exactly two values")
    else:
        parsed_betas = [_finite_float(item) for item in betas]
        if any(item is None or item < 0.0 or item >= 1.0 for item in parsed_betas):
            errors.append(f"{path}.betas values must be finite in [0, 1)")
    eps = _finite_float(value.get("eps"))
    if eps is None or eps <= 0.0:
        errors.append(f"{path}.eps must be finite and positive")
    weight_decay = _finite_float(value.get("weight_decay"))
    if weight_decay is None or weight_decay < 0.0:
        errors.append(f"{path}.weight_decay must be finite and non-negative")
    return errors


def _validate_lora(value: Any) -> list[str]:
    path = "lora"
    if not isinstance(value, dict):
        return [f"{path} must be an object for lora_merge profiles"]
    errors: list[str] = []
    _unknown_keys(errors, value, allowed=_LORA_KEYS, path=path)
    for key in ("rank", "alpha"):
        if not _is_int(value.get(key), minimum=1):
            errors.append(f"{path}.{key} must be a positive integer")
    dropout = _finite_float(value.get("dropout"))
    if dropout is None or dropout < 0.0 or dropout >= 1.0:
        errors.append(f"{path}.dropout must be finite in [0, 1)")
    targets = value.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(item, str) or not item.strip() for item in targets)
    ):
        errors.append(f"{path}.target_modules must be a non-empty string array")
    elif len(set(targets)) != len(targets):
        errors.append(f"{path}.target_modules must be unique")
    if value.get("bias") != "none":
        errors.append(
            f"{path}.bias must be none in the v1 adapter-only training contract"
        )
    if value.get("task_type") != "CAUSAL_LM":
        errors.append(f"{path}.task_type must be CAUSAL_LM")
    if not isinstance(value.get("fan_in_fan_out"), bool):
        errors.append(f"{path}.fan_in_fan_out must be a boolean")
    return errors


def _validate_toolchain(value: Any, *, edit_type: Any) -> list[str]:
    path = "toolchain"
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    errors: list[str] = []
    _unknown_keys(errors, value, allowed=_TOOLCHAIN_KEYS, path=path)
    required = {"python", "torch", "transformers"}
    if edit_type == "lora_merge":
        required.add("peft")
    for package in sorted(required):
        version = value.get(package)
        if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
            errors.append(f"{path}.{package} must be an exact x.y.z[+build] version")
    if edit_type == "fine_tune" and "peft" in value:
        errors.append(f"{path}.peft is only valid for lora_merge profiles")
    return errors


def _validate_model_load(value: Any) -> list[str]:
    path = "model_load"
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    errors: list[str] = []
    _unknown_keys(errors, value, allowed=_MODEL_LOAD_KEYS, path=path)
    if value.get("loss_function") != "ForCausalLM":
        errors.append(f"{path}.loss_function must be ForCausalLM")
    unexpected = value.get("expected_unexpected_keys")
    if not isinstance(unexpected, list) or any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in unexpected
    ):
        errors.append(f"{path}.expected_unexpected_keys must be a string array")
    elif unexpected != sorted(set(unexpected)):
        errors.append(f"{path}.expected_unexpected_keys must be sorted and unique")
    return errors


def training_profile_errors(
    profile_id: str,
    profile: Any,
    *,
    expected_edit_type: str | None = None,
    repo_root: Path = _REPO_ROOT,
    verify_data_file: bool = True,
) -> list[str]:
    """Return fail-closed contract errors for one immutable profile mapping."""

    prefix = f"profile {profile_id!r}: "
    if not isinstance(profile_id, str) or not profile_id.strip():
        return ["profile_id must be a non-empty string"]
    if not isinstance(profile, dict):
        return [prefix + "profile must be an object"]

    errors: list[str] = []
    _unknown_keys(errors, profile, allowed=_PROFILE_KEYS, path="profile")
    digest = profile.get("profile_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        errors.append("profile_sha256 must be a canonical sha256 digest")
    elif digest != canonical_profile_digest(profile):
        errors.append("profile_sha256 does not match canonical profile content")

    edit_type = profile.get("edit_type")
    if edit_type not in _EDIT_TYPES:
        errors.append(f"edit_type must be one of: {', '.join(sorted(_EDIT_TYPES))}")
    if expected_edit_type is not None and edit_type != expected_edit_type:
        errors.append(
            f"edit_type mismatch: expected {expected_edit_type!r}, got {edit_type!r}"
        )

    model_id = profile.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append("model_id must be a non-empty string")
    revision = profile.get("model_revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        errors.append("model_revision must be a pinned 40-character commit digest")

    errors.extend(
        _validate_training_data(
            profile.get("training_data"),
            repo_root=repo_root,
            verify_file=verify_data_file,
        )
    )
    errors.extend(_validate_optimizer(profile.get("optimizer")))

    for key in (
        "steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_sequence_length",
    ):
        if not _is_int(profile.get(key), minimum=1):
            errors.append(f"{key} must be a positive integer")
    if not _is_int(profile.get("seed"), minimum=0):
        errors.append("seed must be a non-negative integer")
    if profile.get("deterministic_algorithms") is not True:
        errors.append("deterministic_algorithms must be true")
    if profile.get("device") not in _DEVICES:
        errors.append(f"device must be one of: {', '.join(sorted(_DEVICES))}")
    if profile.get("dtype") not in _DTYPES:
        errors.append(f"dtype must be one of: {', '.join(sorted(_DTYPES))}")
    errors.extend(_validate_toolchain(profile.get("toolchain"), edit_type=edit_type))
    errors.extend(_validate_model_load(profile.get("model_load")))

    if edit_type == "lora_merge":
        errors.extend(_validate_lora(profile.get("lora")))
    elif "lora" in profile:
        errors.append("fine_tune profiles must not contain a lora configuration")
    return [prefix + error for error in errors]


def _parse_training_data(value: Mapping[str, Any]) -> TrainingDataSpec:
    return TrainingDataSpec(
        path=str(value["path"]),
        sha256=str(value["sha256"]),
        rows=int(value["rows"]),
        text_field=str(value["text_field"]),
    )


def _parse_optimizer(value: Mapping[str, Any]) -> OptimizerSpec:
    betas = value["betas"]
    return OptimizerSpec(
        name=str(value["name"]),
        learning_rate=float(value["learning_rate"]),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(value["eps"]),
        weight_decay=float(value["weight_decay"]),
    )


def _parse_toolchain(value: Mapping[str, Any]) -> TrainingToolchainSpec:
    peft = value.get("peft")
    return TrainingToolchainSpec(
        python=str(value["python"]),
        torch=str(value["torch"]),
        transformers=str(value["transformers"]),
        peft=str(peft) if peft is not None else None,
    )


def _parse_model_load(value: Mapping[str, Any]) -> ModelLoadSpec:
    return ModelLoadSpec(
        loss_function=str(value["loss_function"]),
        expected_unexpected_keys=tuple(
            str(item) for item in value["expected_unexpected_keys"]
        ),
    )


def _common_profile_fields(profile_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "profile_sha256": str(value["profile_sha256"]),
        "edit_type": str(value["edit_type"]),
        "model_id": str(value["model_id"]),
        "model_revision": str(value["model_revision"]),
        "training_data": _parse_training_data(value["training_data"]),
        "optimizer": _parse_optimizer(value["optimizer"]),
        "steps": int(value["steps"]),
        "micro_batch_size": int(value["micro_batch_size"]),
        "gradient_accumulation_steps": int(value["gradient_accumulation_steps"]),
        "max_sequence_length": int(value["max_sequence_length"]),
        "seed": int(value["seed"]),
        "deterministic_algorithms": bool(value["deterministic_algorithms"]),
        "device": str(value["device"]),
        "dtype": str(value["dtype"]),
        "toolchain": _parse_toolchain(value["toolchain"]),
        "model_load": _parse_model_load(value["model_load"]),
    }


def _profile_from_mapping(profile_id: str, value: Mapping[str, Any]) -> TrainingProfile:
    common = _common_profile_fields(profile_id, value)
    if value["edit_type"] == "fine_tune":
        return FineTuneTrainingProfile(**common)
    lora = value["lora"]
    return LoraTrainingProfile(
        **common,
        lora=LoraSpec(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=tuple(str(item) for item in lora["target_modules"]),
            bias=str(lora["bias"]),
            task_type=str(lora["task_type"]),
            fan_in_fan_out=bool(lora["fan_in_fan_out"]),
        ),
    )


def load_training_profile(
    profile_id: str,
    *,
    expected_edit_type: str | None = None,
    profiles_path: Path = DEFAULT_TRAINING_PROFILES_PATH,
    repo_root: Path = _REPO_ROOT,
) -> TrainingProfile:
    """Load one typed profile only after content and data digests validate."""

    try:
        payload = parse_json_bytes(
            read_regular_file_bytes(profiles_path, label="training profiles"),
            label="training profiles",
        )
    except StrictJsonError as exc:
        raise TrainingProfileError(f"unable to load training profiles: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingProfileError("training profiles document must be an object")
    if payload.get("schema") != TRAINING_PROFILES_SCHEMA:
        raise TrainingProfileError("training profiles document has unknown schema")
    if set(payload) != {"schema", "profiles"}:
        raise TrainingProfileError("training profiles document contains unknown fields")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise TrainingProfileError("training profiles document has no profiles")
    raw_profile = profiles.get(profile_id)
    if raw_profile is None:
        raise TrainingProfileError(f"unknown training profile: {profile_id!r}")
    errors = training_profile_errors(
        profile_id,
        raw_profile,
        expected_edit_type=expected_edit_type,
        repo_root=repo_root,
        verify_data_file=True,
    )
    if errors:
        raise TrainingProfileError("; ".join(errors))
    return _profile_from_mapping(profile_id, raw_profile)


__all__ = [
    "BaseTrainingProfile",
    "DEFAULT_TRAINING_PROFILES_PATH",
    "FineTuneTrainingProfile",
    "LoraSpec",
    "LoraTrainingProfile",
    "ModelLoadSpec",
    "OptimizerSpec",
    "TRAINING_PROFILES_SCHEMA",
    "TrainingDataSpec",
    "TrainingProfile",
    "TrainingProfileError",
    "TrainingToolchainSpec",
    "canonical_json_bytes",
    "canonical_profile_digest",
    "canonical_sha256",
    "file_sha256",
    "load_training_profile",
    "lora_config_digest",
    "training_profile_errors",
]
