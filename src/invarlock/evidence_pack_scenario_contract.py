"""Closed, typed scenario records for evidence-pack proof dispatch.

Evidence reports are only meaningful if the scenario that selected their
proof path is unambiguous.  This module consumes the dispatch-relevant part
of one rendered scenario record and turns it into an immutable, canonical
contract.  It intentionally does not execute a proof handler: callers map
the returned :class:`ProofHandler` to their own verifier implementation.

The contract is deliberately closed at the generation boundary.  Scenario
metadata such as prose, suites, and guard thresholds remains owned by the
manifest schema, but an evidence report cannot introduce a new generation
kind, edit alias, backend, or callback-like handler name at verification
time.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

SCENARIO_CONTRACT_VERSION = "invarlock/evidence-pack-scenario-contract-v1"


class ScenarioContractError(ValueError):
    """Raised when a scenario cannot safely select a proof path."""


class GenerationKind(StrEnum):
    """The only scenario generation families accepted by the verifier."""

    EDIT = "edit"
    DEPLOYABLE_EDIT = "deployable_edit"
    ERROR = "error"
    EVIDENCE_ONLY = "evidence_only"


class ArtifactClass(StrEnum):
    """Artifact taxonomy bound to one generation family."""

    VALIDATION_SUBJECT_CHECKPOINT = "validation_subject_checkpoint"
    DEPLOYABLE_OPTIMIZED_SUBJECT = "deployable_optimized_subject"
    FAULT_INJECTION_FIXTURE = "fault_injection_fixture"
    EVIDENCE_ONLY_PACK = "evidence_only_pack"


class Strictness(StrEnum):
    """The only verdict roles a scenario can claim."""

    MUST_PASS = "must_pass"
    MUST_FAIL = "must_fail"
    MUST_DETECT = "must_detect"
    INFORMATIONAL = "informational"


class EditType(StrEnum):
    """The only edit labels that can reach proof dispatch."""

    QUANT_RTN = "quant_rtn"
    MAGNITUDE_PRUNE = "magnitude_prune"
    SYNTHETIC_LOWRANK_DELTA = "synthetic_lowrank_delta"
    SYNTHETIC_DENSE_UPDATE = "synthetic_dense_update"
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"
    LORA_MERGE = "lora_merge"
    FINE_TUNE = "fine_tune"


class ProofHandler(StrEnum):
    """Static proof routes; no handler name is accepted from a scenario."""

    TRANSFORMATION_REPLAY = "transformation_replay"
    MAGNITUDE_PRUNING_REPLAY = "magnitude_pruning_replay"
    DEPLOYABLE_BITSANDBYTES = "deployable_bitsandbytes"
    EXTERNAL_TRAINING = "external_training"
    ERROR_INJECTION = "error_injection"
    EVIDENCE_ONLY = "evidence_only"


@dataclass(frozen=True)
class EditSpecContract:
    """One canonical edit specification selected before proof dispatch."""

    edit_type: EditType
    canonical_spec: str
    version: str
    is_clean: bool
    parameters: tuple[tuple[str, int | float], ...]
    scope: str | None
    backend: str | None

    @property
    def parameter_dict(self) -> dict[str, int | float]:
        """Return a new mapping for consumers that need named parameters."""

        return dict(self.parameters)


@dataclass(frozen=True)
class ErrorSpecContract:
    """A canonical fault-injection request with constrained environment data."""

    error_type: str
    environment: tuple[tuple[str, str], ...]
    environment_by_model: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True)
class TrainingProfileBinding:
    """Immutable profile snapshot selected for a training-profile scenario.

    A training label alone is not enough to describe a training-profile edit.
    The scenario therefore carries the exact immutable profile identity and a
    package-local snapshot whose byte digest is checked before evidence proof
    validation.  The package verifier validates the snapshot's parameters
    against both the typed edit specification and the staged receipt.
    """

    profile_id: str
    profile_sha256: str
    snapshot_path: str
    snapshot_sha256: str


@dataclass(frozen=True)
class ScenarioContract:
    """Immutable, dispatch-ready projection of a scenario record."""

    scenario_id: str
    generation_kind: GenerationKind
    artifact_class: ArtifactClass
    strictness: Strictness
    proof_handler: ProofHandler
    runnable: bool
    edit: EditSpecContract | None
    error: ErrorSpecContract | None
    training_profile: TrainingProfileBinding | None

    @property
    def error_type(self) -> str | None:
        """Return the typed error label for error scenarios, if any."""

        return self.error.error_type if self.error is not None else None


SUPPORTED_ERROR_TYPES = frozenset(
    {
        "nan_injection",
        "inf_injection",
        "shape_mismatch",
        "missing_tensors",
        "extreme_quant",
        "scale_explosion",
        "rank_collapse",
        "norm_collapse",
        "weight_tying_break",
        "rmt_norm_noise",
        "spectral_moderate_scale",
        "ve_mlp_scale_skew",
        "rmt_norm_noise_strong",
        "spectral_moderate_scale_mlp",
        "spectral_moderate_scale_mlp_l31_up_s112",
        "spectral_moderate_scale_attn_l31_o_s105",
        "rmt_norm_noise_l31_ffn_up_b030",
        "ve_mlp_scale_skew_l31_down_s090",
        "ve_mlp_scale_skew_up",
    }
)

VALIDATION_EDIT_TYPES = frozenset(
    {
        EditType.QUANT_RTN,
        EditType.MAGNITUDE_PRUNE,
        EditType.SYNTHETIC_LOWRANK_DELTA,
        EditType.SYNTHETIC_DENSE_UPDATE,
        EditType.LORA_MERGE,
        EditType.FINE_TUNE,
    }
)
DEPLOYABLE_BITSANDBYTES_EDIT_TYPES = frozenset({EditType.BNB_4BIT, EditType.BNB_8BIT})
CLEAN_SELECTION_EDIT_TYPES = frozenset(
    {
        EditType.QUANT_RTN,
        EditType.MAGNITUDE_PRUNE,
        EditType.SYNTHETIC_LOWRANK_DELTA,
        EditType.SYNTHETIC_DENSE_UPDATE,
    }
)

_UNSUPPORTED_EDIT_TYPES = frozenset({"fp8_quant", "lowrank_svd"})
# Standalone pack assembly owns canonical report directories such as
# ``report-001`` while generated scenario families use snake_case.  Both are
# closed lowercase identifiers, not edit aliases; reject empty, repeated, or
# mixed separators rather than applying a lossy normalization.
_SCENARIO_ID_RE = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*\Z")
_CANONICAL_POSITIVE_INT_RE = re.compile(r"[1-9][0-9]*\Z")
_CANONICAL_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
_CANONICAL_SCOPE_INT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ENV_KEY_RE = re.compile(r"INVARLOCK_[A-Z0-9_]+\Z")
_PROFILE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ScenarioContractError(f"{label} must be an object")
    return value


def _closed_mapping(
    value: object,
    *,
    label: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    mapping = _mapping(value, label=label)
    keys = set(mapping)
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if missing or extra:
        raise ScenarioContractError(
            f"{label} has missing or unsupported fields "
            f"(missing={missing}, unsupported={extra})"
        )
    return mapping


def _canonical_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ScenarioContractError(f"{label} must be a non-empty canonical string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ScenarioContractError(f"{label} must be a non-empty canonical string")
    return value


def _scenario_id(value: object) -> str:
    scenario_id = _canonical_text(value, label="scenario id")
    if _SCENARIO_ID_RE.fullmatch(scenario_id) is None:
        raise ScenarioContractError(
            "scenario id must be a lowercase canonical identifier"
        )
    return scenario_id


def _artifact_class(value: object) -> ArtifactClass:
    if not isinstance(value, str):
        raise ScenarioContractError("artifact_class is required and must be canonical")
    try:
        return ArtifactClass(value)
    except ValueError as exc:
        raise ScenarioContractError(
            f"artifact_class {value!r} is unsupported or noncanonical"
        ) from exc


def _strictness(value: object) -> Strictness:
    if not isinstance(value, str):
        raise ScenarioContractError("strictness is required and must be canonical")
    try:
        return Strictness(value)
    except ValueError as exc:
        raise ScenarioContractError(
            f"strictness {value!r} is unsupported or noncanonical"
        ) from exc


def _generation_kind(value: object) -> GenerationKind:
    if not isinstance(value, str):
        raise ScenarioContractError("generation.kind is required and must be canonical")
    try:
        return GenerationKind(value)
    except ValueError as exc:
        raise ScenarioContractError(
            f"generation.kind {value!r} is unsupported or noncanonical"
        ) from exc


def _runnable(value: object) -> bool:
    if not isinstance(value, bool):
        raise ScenarioContractError("runnable must be a boolean when declared")
    if value is not True:
        raise ScenarioContractError(
            "runnable:false scenarios cannot be accepted as evidence reports"
        )
    return True


def _version(value: object, *, allowed: frozenset[str], kind: GenerationKind) -> str:
    version = _canonical_text(value, label="generation.version")
    if version not in allowed:
        raise ScenarioContractError(
            f"generation.version {version!r} is invalid for {kind.value}"
        )
    return version


def _positive_int(value: str, *, label: str) -> int:
    if _CANONICAL_POSITIVE_INT_RE.fullmatch(value) is None:
        raise ScenarioContractError(f"{label} must be a canonical positive integer")
    return int(value)


def _positive_decimal(value: str, *, label: str) -> float:
    if _CANONICAL_DECIMAL_RE.fullmatch(value) is None:
        raise ScenarioContractError(
            f"{label} must be a canonical finite positive decimal"
        )
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex rejects invalid text
        raise ScenarioContractError(
            f"{label} must be a canonical finite positive decimal"
        ) from exc
    if not decimal.is_finite() or decimal <= 0:
        raise ScenarioContractError(
            f"{label} must be a canonical finite positive decimal"
        )
    normalized = float(decimal)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ScenarioContractError(
            f"{label} must be a canonical finite positive decimal"
        )
    return normalized


def _sparsity(value: str) -> float:
    parsed = _positive_decimal(value, label="magnitude_prune target sparsity")
    if parsed >= 1.0:
        raise ScenarioContractError("magnitude_prune target sparsity must be in (0, 1)")
    return parsed


def _simple_scope(value: str, *, edit_type: EditType) -> str:
    if value not in {"ffn", "attn", "all"}:
        raise ScenarioContractError(
            f"{edit_type.value} scope must be canonical one of ['all', 'attn', 'ffn']"
        )
    return value


def _transformation_scope(value: str) -> str:
    """Parse the replay contract's scope grammar and require its canonical form."""

    if not value or value.count("@") > 1:
        raise ScenarioContractError("transformation scope must be canonical")
    if "@" in value:
        raw_base, raw_qualifiers = value.split("@", 1)
    else:
        raw_base, raw_qualifiers = value, ""
    if raw_base not in {"ffn", "attn", "all"}:
        raise ScenarioContractError("transformation scope must be canonical")
    if "@" not in value:
        return raw_base
    if not raw_qualifiers:
        raise ScenarioContractError("transformation scope must be canonical")

    qualifiers: dict[str, int] = {}
    for item in raw_qualifiers.split(","):
        if item.count("=") != 1:
            raise ScenarioContractError("transformation scope must be canonical")
        name, raw_number = item.split("=", 1)
        if name not in {"layers", "layer"} or name in qualifiers:
            raise ScenarioContractError("transformation scope must be canonical")
        if _CANONICAL_SCOPE_INT_RE.fullmatch(raw_number) is None:
            raise ScenarioContractError("transformation scope must be canonical")
        number = int(raw_number)
        if name == "layers" and number == 0:
            raise ScenarioContractError("transformation scope must be canonical")
        qualifiers[name] = number

    layer_limit = qualifiers.get("layers")
    layer = qualifiers.get("layer")
    if layer_limit is not None and layer is not None and layer >= layer_limit:
        raise ScenarioContractError("transformation scope must be canonical")
    canonical_parts: list[str] = []
    if layer_limit is not None:
        canonical_parts.append(f"layers={layer_limit}")
    if layer is not None:
        canonical_parts.append(f"layer={layer}")
    canonical = raw_base + "@" + ",".join(canonical_parts)
    if value != canonical:
        raise ScenarioContractError("transformation scope must be canonical")
    return canonical


def _edit_type(value: str) -> EditType:
    if value in _UNSUPPORTED_EDIT_TYPES:
        raise ScenarioContractError(
            f"unsupported edit type {value!r}; it needs a dedicated proof contract"
        )
    try:
        return EditType(value)
    except ValueError as exc:
        raise ScenarioContractError(
            f"edit type {value!r} is unsupported or noncanonical"
        ) from exc


def _proof_handler_for_edit(edit_type: EditType) -> ProofHandler:
    if edit_type in {
        EditType.QUANT_RTN,
        EditType.SYNTHETIC_LOWRANK_DELTA,
        EditType.SYNTHETIC_DENSE_UPDATE,
    }:
        return ProofHandler.TRANSFORMATION_REPLAY
    if edit_type is EditType.MAGNITUDE_PRUNE:
        return ProofHandler.MAGNITUDE_PRUNING_REPLAY
    if edit_type in DEPLOYABLE_BITSANDBYTES_EDIT_TYPES:
        return ProofHandler.DEPLOYABLE_BITSANDBYTES
    if edit_type in {EditType.LORA_MERGE, EditType.FINE_TUNE}:
        return ProofHandler.EXTERNAL_TRAINING
    raise AssertionError(f"unhandled edit type: {edit_type}")


def _clean_edit(
    *, edit_type: EditType, parts: list[str], version: str
) -> EditSpecContract:
    if len(parts) != 2:
        raise ScenarioContractError(
            "clean edit_spec must be exactly '<edit_type>:clean'"
        )
    if version != "clean":
        raise ScenarioContractError(
            "clean edit_spec requires generation.version='clean'"
        )
    if edit_type not in CLEAN_SELECTION_EDIT_TYPES:
        raise ScenarioContractError(
            f"clean edit_spec is unsupported for {edit_type.value}; "
            "add a selection and proof contract first"
        )
    return EditSpecContract(
        edit_type=edit_type,
        canonical_spec=f"{edit_type.value}:clean",
        version=version,
        is_clean=True,
        parameters=(),
        scope=None,
        backend=None,
    )


def _validation_edit(
    *, edit_type: EditType, parts: list[str], version: str
) -> EditSpecContract:
    if version == "clean":
        raise ScenarioContractError("clean edit_spec must use the exact ':clean' form")
    training_edit = edit_type in {EditType.LORA_MERGE, EditType.FINE_TUNE}
    expected_version = "trained" if training_edit else "stress"
    if version != expected_version:
        raise ScenarioContractError(
            "training edit generation.version must be 'trained'"
            if training_edit
            else "validation edit generation.version must be 'stress'"
        )
    if edit_type is EditType.QUANT_RTN:
        if len(parts) != 4:
            raise ScenarioContractError("quant_rtn edit_spec must have four fields")
        bits = _positive_int(parts[1], label="quant_rtn bits")
        if not 2 <= bits <= 8:
            raise ScenarioContractError("quant_rtn bits must be in [2, 8]")
        group_size = _positive_int(parts[2], label="quant_rtn group_size")
        scope = _transformation_scope(parts[3])
        canonical = f"quant_rtn:{bits}:{group_size}:{scope}"
        parameters: tuple[tuple[str, int | float], ...] = (
            ("bits", bits),
            ("group_size", group_size),
        )
    elif edit_type is EditType.MAGNITUDE_PRUNE:
        if len(parts) != 3:
            raise ScenarioContractError(
                "magnitude_prune edit_spec must have three fields"
            )
        sparsity = _sparsity(parts[1])
        scope = _simple_scope(parts[2], edit_type=edit_type)
        canonical = f"magnitude_prune:{parts[1]}:{scope}"
        parameters = (("target_sparsity", sparsity),)
    elif edit_type is EditType.SYNTHETIC_LOWRANK_DELTA:
        if len(parts) != 4:
            raise ScenarioContractError(
                "synthetic_lowrank_delta edit_spec must have four fields"
            )
        rank = _positive_int(parts[1], label="synthetic_lowrank_delta rank")
        scale = _positive_decimal(parts[2], label="synthetic_lowrank_delta scale")
        scope = _transformation_scope(parts[3])
        canonical = f"synthetic_lowrank_delta:{rank}:{parts[2]}:{scope}"
        parameters = (("rank", rank), ("scale", scale))
    elif edit_type is EditType.SYNTHETIC_DENSE_UPDATE:
        if len(parts) != 4:
            raise ScenarioContractError(
                "synthetic_dense_update edit_spec must have four fields"
            )
        step_size = _positive_decimal(
            parts[1], label="synthetic_dense_update step_size"
        )
        iterations = _positive_int(parts[2], label="synthetic_dense_update iterations")
        scope = _transformation_scope(parts[3])
        canonical = f"synthetic_dense_update:{parts[1]}:{iterations}:{scope}"
        parameters = (("step_size", step_size), ("iterations", iterations))
    elif edit_type is EditType.LORA_MERGE:
        if len(parts) != 4:
            raise ScenarioContractError("lora_merge edit_spec must have four fields")
        rank = _positive_int(parts[1], label="lora_merge rank")
        alpha = _positive_decimal(parts[2], label="lora_merge alpha")
        scope = _simple_scope(parts[3], edit_type=edit_type)
        canonical = f"lora_merge:{rank}:{parts[2]}:{scope}"
        parameters = (("rank", rank), ("alpha", alpha))
    elif edit_type is EditType.FINE_TUNE:
        if len(parts) != 4:
            raise ScenarioContractError("fine_tune edit_spec must have four fields")
        learning_rate = _positive_decimal(parts[1], label="fine_tune learning_rate")
        steps = _positive_int(parts[2], label="fine_tune steps")
        scope = _simple_scope(parts[3], edit_type=edit_type)
        canonical = f"fine_tune:{parts[1]}:{steps}:{scope}"
        parameters = (("learning_rate", learning_rate), ("steps", steps))
    else:
        raise AssertionError(f"non-validation edit type: {edit_type}")
    if ":".join(parts) != canonical:
        raise ScenarioContractError("edit_spec is not canonical")
    return EditSpecContract(
        edit_type=edit_type,
        canonical_spec=canonical,
        version=version,
        is_clean=False,
        parameters=parameters,
        scope=scope,
        backend=None,
    )


def _deployable_edit(
    *, edit_type: EditType, parts: list[str], version: str, backend: str
) -> EditSpecContract:
    if edit_type not in DEPLOYABLE_BITSANDBYTES_EDIT_TYPES:
        raise ScenarioContractError(
            "deployable_edit only supports the typed bitsandbytes edit families"
        )
    if backend != "bitsandbytes":
        raise ScenarioContractError(
            "deployable bitsandbytes edit requires generation.backend='bitsandbytes'"
        )
    if version not in {"deployable", "stress"}:
        raise ScenarioContractError(
            "deployable edit generation.version must be 'deployable' or 'stress'"
        )
    if len(parts) != 3:
        raise ScenarioContractError("bitsandbytes edit_spec must have three fields")
    expected_bits = 4 if edit_type is EditType.BNB_4BIT else 8
    bits = _positive_int(parts[1], label=f"{edit_type.value} bits")
    if bits != expected_bits:
        raise ScenarioContractError(
            f"{edit_type.value} bits must be exactly {expected_bits}"
        )
    scope = _simple_scope(parts[2], edit_type=edit_type)
    if scope != "all":
        raise ScenarioContractError("bitsandbytes scope must be exactly 'all'")
    canonical = f"{edit_type.value}:{expected_bits}:all"
    if ":".join(parts) != canonical:
        raise ScenarioContractError("edit_spec is not canonical")
    return EditSpecContract(
        edit_type=edit_type,
        canonical_spec=canonical,
        version=version,
        is_clean=False,
        parameters=(("bits", expected_bits),),
        scope=scope,
        backend=backend,
    )


def _parse_edit(
    generation: Mapping[str, object], *, kind: GenerationKind
) -> EditSpecContract:
    if kind is GenerationKind.EDIT:
        _closed_mapping(
            generation,
            label="edit generation",
            required=frozenset({"kind", "edit_spec", "version"}),
        )
        version = _version(
            generation["version"],
            allowed=frozenset({"clean", "stress", "trained"}),
            kind=kind,
        )
        backend = None
    elif kind is GenerationKind.DEPLOYABLE_EDIT:
        _closed_mapping(
            generation,
            label="deployable edit generation",
            required=frozenset({"kind", "edit_spec", "version", "backend"}),
        )
        version = _version(
            generation["version"],
            allowed=frozenset({"deployable", "stress"}),
            kind=kind,
        )
        backend = _canonical_text(generation["backend"], label="generation.backend")
    else:
        raise AssertionError(f"non-edit generation kind: {kind}")

    edit_spec = _canonical_text(generation["edit_spec"], label="generation.edit_spec")
    parts = edit_spec.split(":")
    if not parts or not parts[0]:
        raise ScenarioContractError("generation.edit_spec must begin with an edit type")
    edit_type = _edit_type(parts[0])
    if kind is GenerationKind.EDIT and edit_type not in VALIDATION_EDIT_TYPES:
        raise ScenarioContractError(
            f"{edit_type.value} requires generation.kind='deployable_edit'"
        )
    if kind is GenerationKind.DEPLOYABLE_EDIT and edit_type not in {
        EditType.BNB_4BIT,
        EditType.BNB_8BIT,
    }:
        raise ScenarioContractError(
            f"{edit_type.value} requires generation.kind='edit'"
        )

    if len(parts) >= 2 and parts[1] == "clean":
        return _clean_edit(edit_type=edit_type, parts=parts, version=version)
    if kind is GenerationKind.DEPLOYABLE_EDIT:
        assert backend is not None
        return _deployable_edit(
            edit_type=edit_type,
            parts=parts,
            version=version,
            backend=backend,
        )
    return _validation_edit(edit_type=edit_type, parts=parts, version=version)


def _environment(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    mapping = _mapping(value, label=label)
    if not mapping:
        raise ScenarioContractError(f"{label} must not be empty")
    normalized: list[tuple[str, str]] = []
    for key, raw in mapping.items():
        if _ENV_KEY_RE.fullmatch(key) is None:
            raise ScenarioContractError(f"{label} environment key {key!r} is invalid")
        normalized.append((key, _canonical_text(raw, label=f"{label}.{key}")))
    return tuple(sorted(normalized))


def _error_contract(generation: Mapping[str, object]) -> ErrorSpecContract:
    closed = _closed_mapping(
        generation,
        label="error generation",
        required=frozenset({"kind", "error_type"}),
        optional=frozenset({"env", "env_by_model"}),
    )
    error_type = _canonical_text(closed["error_type"], label="generation.error_type")
    if error_type not in SUPPORTED_ERROR_TYPES:
        raise ScenarioContractError(
            f"generation.error_type {error_type!r} is unsupported or noncanonical"
        )
    environment: tuple[tuple[str, str], ...] = ()
    if "env" in closed:
        environment = _environment(closed["env"], label="generation.env")

    environment_by_model: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    if "env_by_model" in closed:
        raw_by_model = _mapping(closed["env_by_model"], label="generation.env_by_model")
        if not raw_by_model:
            raise ScenarioContractError("generation.env_by_model must not be empty")
        for model_id, model_environment in raw_by_model.items():
            _canonical_text(model_id, label="generation.env_by_model model id")
            environment_by_model.append(
                (
                    model_id,
                    _environment(
                        model_environment,
                        label=f"generation.env_by_model[{model_id!r}]",
                    ),
                )
            )
    return ErrorSpecContract(
        error_type=error_type,
        environment=environment,
        environment_by_model=tuple(sorted(environment_by_model)),
    )


def _expected_artifact_class(kind: GenerationKind) -> ArtifactClass:
    if kind is GenerationKind.EDIT:
        return ArtifactClass.VALIDATION_SUBJECT_CHECKPOINT
    if kind is GenerationKind.DEPLOYABLE_EDIT:
        return ArtifactClass.DEPLOYABLE_OPTIMIZED_SUBJECT
    if kind is GenerationKind.ERROR:
        return ArtifactClass.FAULT_INJECTION_FIXTURE
    if kind is GenerationKind.EVIDENCE_ONLY:
        return ArtifactClass.EVIDENCE_ONLY_PACK
    raise AssertionError(f"unhandled generation kind: {kind}")


def _validate_deployment_flag(
    record: Mapping[str, object], *, kind: GenerationKind
) -> None:
    value = record.get("optimized_deployment_backend")
    if kind is GenerationKind.DEPLOYABLE_EDIT and value is not True:
        raise ScenarioContractError(
            "deployable_edit requires optimized_deployment_backend=true"
        )
    if (
        kind is not GenerationKind.DEPLOYABLE_EDIT
        and value is not None
        and value is not False
    ):
        raise ScenarioContractError(
            "non-deployable scenario cannot set optimized_deployment_backend=true"
        )


def _training_profile_binding(
    scenario: Mapping[str, object], *, edit: EditSpecContract
) -> TrainingProfileBinding:
    """Parse the profile snapshot binding required for optimizer-backed edits."""

    raw = _closed_mapping(
        scenario.get("training_profile"),
        label="training_profile",
        required=frozenset(
            {"profile_id", "profile_sha256", "snapshot_path", "snapshot_sha256"}
        ),
    )
    profile_id = _canonical_text(raw["profile_id"], label="training_profile.profile_id")
    if _PROFILE_ID_RE.fullmatch(profile_id) is None:
        raise ScenarioContractError("training_profile.profile_id is invalid")
    profile_sha256 = _canonical_text(
        raw["profile_sha256"], label="training_profile.profile_sha256"
    )
    if _SHA256_RE.fullmatch(profile_sha256) is None:
        raise ScenarioContractError(
            "training_profile.profile_sha256 must be a canonical sha256 digest"
        )
    snapshot_path = _canonical_text(
        raw["snapshot_path"], label="training_profile.snapshot_path"
    )
    expected_path = f"metadata/training_profiles/{profile_id}.json"
    if snapshot_path != expected_path:
        raise ScenarioContractError(
            "training_profile.snapshot_path must be the canonical profile snapshot path"
        )
    snapshot_sha256 = _canonical_text(
        raw["snapshot_sha256"], label="training_profile.snapshot_sha256"
    )
    if _SHA256_RE.fullmatch(snapshot_sha256) is None:
        raise ScenarioContractError(
            "training_profile.snapshot_sha256 must be a canonical sha256 digest"
        )
    if edit.edit_type not in {EditType.LORA_MERGE, EditType.FINE_TUNE}:
        raise ScenarioContractError(
            "training_profile is only valid for a training-profile edit"
        )
    return TrainingProfileBinding(
        profile_id=profile_id,
        profile_sha256=profile_sha256,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha256,
    )


def parse_scenario_contract(record: object) -> ScenarioContract:
    """Return a closed, canonical scenario contract before proof dispatch.

    The v1 manifest schema permits an omitted ``runnable`` field; omission
    canonicalizes to ``True``.  An explicit false value is always rejected:
    a placeholder must never be accepted as evidence.
    """

    scenario = _mapping(record, label="scenario")
    scenario_id = _scenario_id(scenario.get("id"))
    runnable = _runnable(scenario["runnable"]) if "runnable" in scenario else True
    artifact_class = _artifact_class(scenario.get("artifact_class"))
    strictness = _strictness(scenario.get("strictness"))
    raw_generation = scenario.get("generation")
    if raw_generation is None:
        raise ScenarioContractError("generation is required")
    generation = _mapping(raw_generation, label="generation")
    kind = _generation_kind(generation.get("kind"))
    expected_artifact_class = _expected_artifact_class(kind)
    if artifact_class is not expected_artifact_class:
        raise ScenarioContractError(
            f"artifact_class {artifact_class.value!r} conflicts with "
            f"generation.kind={kind.value!r}; expected "
            f"{expected_artifact_class.value!r}"
        )
    _validate_deployment_flag(scenario, kind=kind)

    if kind is GenerationKind.ERROR:
        if "training_profile" in scenario:
            raise ScenarioContractError(
                "training_profile is only valid for a training-profile edit"
            )
        error = _error_contract(generation)
        return ScenarioContract(
            scenario_id=scenario_id,
            generation_kind=kind,
            artifact_class=artifact_class,
            strictness=strictness,
            proof_handler=ProofHandler.ERROR_INJECTION,
            runnable=runnable,
            edit=None,
            error=error,
            training_profile=None,
        )

    if kind is GenerationKind.EVIDENCE_ONLY:
        if "training_profile" in scenario:
            raise ScenarioContractError(
                "training_profile is only valid for a training-profile edit"
            )
        _closed_mapping(
            generation,
            label="evidence-only generation",
            required=frozenset({"kind"}),
        )
        return ScenarioContract(
            scenario_id=scenario_id,
            generation_kind=kind,
            artifact_class=artifact_class,
            strictness=strictness,
            proof_handler=ProofHandler.EVIDENCE_ONLY,
            runnable=runnable,
            edit=None,
            error=None,
            training_profile=None,
        )

    edit = _parse_edit(generation, kind=kind)
    proof_handler = _proof_handler_for_edit(edit.edit_type)
    if proof_handler is ProofHandler.EXTERNAL_TRAINING:
        training_profile = _training_profile_binding(scenario, edit=edit)
    else:
        if "training_profile" in scenario:
            raise ScenarioContractError(
                "training_profile is only valid for a training-profile edit"
            )
        training_profile = None
    return ScenarioContract(
        scenario_id=scenario_id,
        generation_kind=kind,
        artifact_class=artifact_class,
        strictness=strictness,
        proof_handler=proof_handler,
        runnable=runnable,
        edit=edit,
        error=None,
        training_profile=training_profile,
    )


__all__ = [
    "ArtifactClass",
    "CLEAN_SELECTION_EDIT_TYPES",
    "DEPLOYABLE_BITSANDBYTES_EDIT_TYPES",
    "EditSpecContract",
    "EditType",
    "ErrorSpecContract",
    "GenerationKind",
    "ProofHandler",
    "SCENARIO_CONTRACT_VERSION",
    "SUPPORTED_ERROR_TYPES",
    "ScenarioContract",
    "ScenarioContractError",
    "Strictness",
    "TrainingProfileBinding",
    "VALIDATION_EDIT_TYPES",
    "parse_scenario_contract",
]
