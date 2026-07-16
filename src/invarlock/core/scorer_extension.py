"""Verifier-replayable scorer extension boundary.

Exact match and normalized NLL remain package-owned acceptance metrics.  This
module defines the smaller boundary for optional, task-specific scorers: an
authorized installed extension receives only authenticated per-record facts and
returns finite per-record values plus one deterministic aggregate.  Network,
external-model, and human/LLM judgment are deliberately outside this acceptance
boundary.

Extensions do not define aggregate or direction semantics.  Every replayed
record value is a finite unit-interval score where higher is better, and core
computes the arithmetic mean.  The paired evaluator can therefore apply one
generic percentage-point delta policy without importing a metric catalog.

Installing an extension authorizes its Python code to execute.  The registry
therefore discovers installed extensions only when third-party plugins are
explicitly enabled, while still validating the extension's identity, ABI,
configuration schema, configuration digest, replay compatibility, and output.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Any, Protocol, cast

from jsonschema import Draft202012Validator

from invarlock.runtime_security import third_party_plugins_allowed

SCORER_EXTENSION_ENTRY_POINT_GROUP = "invarlock.scorers"
SCORER_EXTENSION_ABI_VERSION = "1"
SCORER_EXTENSION_DESCRIPTOR_FORMAT = "invarlock/scorer-extension-descriptor-v1"
SCORER_EXTENSION_BINDING_FORMAT = "invarlock/scorer-extension-binding-v1"
SCORER_EXTENSION_RESULT_FORMAT = "invarlock/scorer-extension-result-v1"
SCORER_REPLAY_MODE = "authenticated_record_facts"
SCORER_RECORD_VALUE_SEMANTICS = "unit_interval_score"
SCORER_AGGREGATION = "arithmetic_mean"
SCORER_DIRECTION = "higher_is_better"
MAX_SCORER_CONFIGURATION_BYTES = 64 * 1024

# These metrics are implemented and replayed by core.  They are intentionally
# not entry points and cannot be shadowed by an extension.
CANONICAL_BUILTIN_ACCEPTANCE_METRICS = frozenset(
    {"exact_match", "normalized_nll_per_utf8_byte"}
)

_SCORER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{0,63})+$")
_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_CANONICAL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_INPUT_KINDS = frozenset({"text", "content"})
SCORER_AUTHENTICATED_FACTS = frozenset(
    {"expected_output", "output_text", "output_sha256"}
)


class ScorerExtensionError(ValueError):
    """Raised when an optional scorer cannot be trusted for replay."""


def _require_scorer_id(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SCORER_ID.fullmatch(value) is None:
        raise ScorerExtensionError(
            f"{field_name} must be a dotted canonical scorer identifier"
        )
    if value in CANONICAL_BUILTIN_ACCEPTANCE_METRICS:
        raise ScorerExtensionError(f"{field_name} cannot shadow a built-in metric")
    return value


def _require_version(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise ScorerExtensionError(f"{field_name} must be a stable major.minor.patch")
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScorerExtensionError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _CANONICAL_NAME.fullmatch(value) is None:
        raise ScorerExtensionError(f"{field_name} must be a canonical identifier")
    return value


def _require_name_tuple(
    value: object,
    *,
    field_name: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ScorerExtensionError(f"{field_name} must be a non-empty tuple")
    if len(value) != len(set(value)):
        raise ScorerExtensionError(f"{field_name} must not contain duplicates")
    for item in value:
        _require_name(item, field_name=f"{field_name} entry")
        if allowed is not None and item not in allowed:
            raise ScorerExtensionError(
                f"{field_name} contains unsupported value {item!r}"
            )
    return value


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScorerExtensionError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ScorerExtensionError(f"{field_name} must be a finite number")
    return 0.0 if normalized == 0.0 else normalized


def _unit_interval_score(value: object, *, field_name: str) -> float:
    normalized = _finite_number(value, field_name=field_name)
    if normalized < 0.0 or normalized > 1.0:
        raise ScorerExtensionError(f"{field_name} must be between zero and one")
    return normalized


def _freeze_json(value: object, *, field_name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScorerExtensionError(f"{field_name} contains a non-finite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScorerExtensionError(f"{field_name} contains a non-string key")
            frozen[key] = _freeze_json(item, field_name=f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ScorerExtensionError(
        f"{field_name} contains non-JSON value {type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerExtensionError("value is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ScorerExtensionDescriptor:
    """Stable capabilities and trust constraints declared by one scorer."""

    scorer_id: str
    scorer_version: str
    supported_tasks: tuple[str, ...]
    supported_input_kinds: tuple[str, ...]
    supported_output_kinds: tuple[str, ...]
    required_facts: tuple[str, ...]
    configuration_schema_sha256: str
    replay_mode: str = SCORER_REPLAY_MODE
    record_value_semantics: str = SCORER_RECORD_VALUE_SEMANTICS
    aggregation: str = SCORER_AGGREGATION
    direction: str = SCORER_DIRECTION
    uses_network: bool = False
    uses_external_model: bool = False
    uses_human_judgment: bool = False
    format_version: str = field(default=SCORER_EXTENSION_DESCRIPTOR_FORMAT, init=False)
    scorer_abi: str = field(default=SCORER_EXTENSION_ABI_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_scorer_id(self.scorer_id, field_name="scorer_id")
        _require_version(self.scorer_version, field_name="scorer_version")
        _require_name_tuple(self.supported_tasks, field_name="supported_tasks")
        _require_name_tuple(
            self.supported_input_kinds,
            field_name="supported_input_kinds",
            allowed=_INPUT_KINDS,
        )
        _require_name_tuple(
            self.supported_output_kinds, field_name="supported_output_kinds"
        )
        _require_name_tuple(self.required_facts, field_name="required_facts")
        if frozenset(self.required_facts) != SCORER_AUTHENTICATED_FACTS:
            raise ScorerExtensionError(
                "v1 scorer extensions must require exactly expected_output, "
                "output_text, and output_sha256"
            )
        _require_sha256(
            self.configuration_schema_sha256,
            field_name="configuration_schema_sha256",
        )
        if self.replay_mode != SCORER_REPLAY_MODE:
            raise ScorerExtensionError(
                "extension scorers must replay authenticated per-record facts"
            )
        if (
            self.record_value_semantics != SCORER_RECORD_VALUE_SEMANTICS
            or self.aggregation != SCORER_AGGREGATION
            or self.direction != SCORER_DIRECTION
        ):
            raise ScorerExtensionError(
                "extension scorers must return higher-is-better unit-interval "
                "record scores aggregated by the core arithmetic mean"
            )
        if (
            self.uses_network is not False
            or self.uses_external_model is not False
            or self.uses_human_judgment is not False
        ):
            raise ScorerExtensionError(
                "network, external-model, human, and LLM judges are not eligible "
                "for acceptance replay"
            )


def scorer_descriptor_payload(
    descriptor: ScorerExtensionDescriptor,
) -> dict[str, object]:
    """Return the canonical public descriptor mapping."""

    if not isinstance(descriptor, ScorerExtensionDescriptor):
        raise ScorerExtensionError("descriptor must be a ScorerExtensionDescriptor")
    return {
        "configuration_schema_sha256": descriptor.configuration_schema_sha256,
        "direction": descriptor.direction,
        "format_version": descriptor.format_version,
        "aggregation": descriptor.aggregation,
        "replay_mode": descriptor.replay_mode,
        "record_value_semantics": descriptor.record_value_semantics,
        "required_facts": list(descriptor.required_facts),
        "scorer_abi": descriptor.scorer_abi,
        "scorer_id": descriptor.scorer_id,
        "scorer_version": descriptor.scorer_version,
        "supported_input_kinds": list(descriptor.supported_input_kinds),
        "supported_output_kinds": list(descriptor.supported_output_kinds),
        "supported_tasks": list(descriptor.supported_tasks),
        "uses_external_model": descriptor.uses_external_model,
        "uses_human_judgment": descriptor.uses_human_judgment,
        "uses_network": descriptor.uses_network,
    }


def canonical_scorer_descriptor_json(
    descriptor: ScorerExtensionDescriptor,
) -> bytes:
    """Encode the descriptor used for identity and evidence binding."""

    return _canonical_json_bytes(scorer_descriptor_payload(descriptor))


def scorer_descriptor_sha256(descriptor: ScorerExtensionDescriptor) -> str:
    return _sha256(canonical_scorer_descriptor_json(descriptor))


def scorer_configuration_schema_sha256(schema: Mapping[str, object]) -> str:
    """Return the canonical digest of one JSON Schema object."""

    if not isinstance(schema, Mapping):
        raise ScorerExtensionError("configuration schema must be a JSON object")
    frozen = _freeze_json(schema, field_name="configuration schema")
    assert isinstance(frozen, Mapping)
    return _sha256(_canonical_json_bytes(frozen))


@dataclass(frozen=True)
class ScorerExtensionBinding:
    """Authenticated algorithm and configuration selected by a request."""

    scorer_id: str
    scorer_version: str
    descriptor_sha256: str
    configuration: Mapping[str, object]
    configuration_sha256: str
    format_version: str = field(default=SCORER_EXTENSION_BINDING_FORMAT, init=False)
    scorer_abi: str = field(default=SCORER_EXTENSION_ABI_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_scorer_id(self.scorer_id, field_name="scorer_id")
        _require_version(self.scorer_version, field_name="scorer_version")
        _require_sha256(self.descriptor_sha256, field_name="descriptor_sha256")
        if not isinstance(self.configuration, Mapping):
            raise ScorerExtensionError("configuration must be a JSON object")
        frozen = _freeze_json(self.configuration, field_name="configuration")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "configuration", frozen)
        encoded = _canonical_json_bytes(frozen)
        if len(encoded) > MAX_SCORER_CONFIGURATION_BYTES:
            raise ScorerExtensionError(
                "configuration exceeds the 65536-byte size limit"
            )
        _require_sha256(self.configuration_sha256, field_name="configuration_sha256")
        if self.configuration_sha256 != _sha256(encoded):
            raise ScorerExtensionError(
                "configuration_sha256 does not match canonical configuration"
            )


def build_scorer_binding(
    descriptor: ScorerExtensionDescriptor,
    configuration: Mapping[str, object],
) -> ScorerExtensionBinding:
    """Bind one concrete configuration to an exact extension descriptor."""

    frozen = _freeze_json(configuration, field_name="configuration")
    if not isinstance(frozen, Mapping):
        raise ScorerExtensionError("configuration must be a JSON object")
    return ScorerExtensionBinding(
        scorer_id=descriptor.scorer_id,
        scorer_version=descriptor.scorer_version,
        descriptor_sha256=scorer_descriptor_sha256(descriptor),
        configuration=frozen,
        configuration_sha256=_sha256(_canonical_json_bytes(frozen)),
    )


def decode_scorer_binding(value: object) -> ScorerExtensionBinding:
    """Decode one closed binding from an authenticated request or policy."""

    if not isinstance(value, Mapping):
        raise ScorerExtensionError("scorer extension binding must be a JSON object")
    expected = {
        "format_version",
        "scorer_abi",
        "scorer_id",
        "scorer_version",
        "descriptor_sha256",
        "configuration",
        "configuration_sha256",
    }
    if set(value) != expected:
        raise ScorerExtensionError("scorer extension binding fields are invalid")
    if value.get("format_version") != SCORER_EXTENSION_BINDING_FORMAT:
        raise ScorerExtensionError("scorer extension binding format is invalid")
    if value.get("scorer_abi") != SCORER_EXTENSION_ABI_VERSION:
        raise ScorerExtensionError("scorer extension binding ABI is invalid")
    configuration = value.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ScorerExtensionError("scorer extension configuration must be an object")
    return ScorerExtensionBinding(
        scorer_id=value.get("scorer_id"),  # type: ignore[arg-type]
        scorer_version=value.get("scorer_version"),  # type: ignore[arg-type]
        descriptor_sha256=value.get("descriptor_sha256"),  # type: ignore[arg-type]
        configuration=configuration,
        configuration_sha256=value.get("configuration_sha256"),  # type: ignore[arg-type]
    )


def scorer_binding_payload(binding: ScorerExtensionBinding) -> dict[str, object]:
    if not isinstance(binding, ScorerExtensionBinding):
        raise ScorerExtensionError("binding must be a ScorerExtensionBinding")
    configuration = _thaw_json(binding.configuration)
    assert isinstance(configuration, dict)
    return {
        "configuration": configuration,
        "configuration_sha256": binding.configuration_sha256,
        "descriptor_sha256": binding.descriptor_sha256,
        "format_version": binding.format_version,
        "scorer_abi": binding.scorer_abi,
        "scorer_id": binding.scorer_id,
        "scorer_version": binding.scorer_version,
    }


def canonical_scorer_binding_json(binding: ScorerExtensionBinding) -> bytes:
    return _canonical_json_bytes(scorer_binding_payload(binding))


@dataclass(frozen=True)
class AuthenticatedScorerRecord:
    """One ordered record and the facts authenticated by the evidence pack."""

    record_id: str
    input_sha256: str
    facts: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.record_id, str)
            or not self.record_id
            or self.record_id != self.record_id.strip()
            or any(ord(character) < 32 for character in self.record_id)
        ):
            raise ScorerExtensionError("record_id must be a non-empty safe string")
        _require_sha256(self.input_sha256, field_name="input_sha256")
        if not isinstance(self.facts, Mapping):
            raise ScorerExtensionError("facts must be a JSON object")
        frozen = _freeze_json(self.facts, field_name=f"facts for {self.record_id!r}")
        assert isinstance(frozen, Mapping)
        for fact_name in frozen:
            _require_name(fact_name, field_name="fact name")
        if set(frozen) != SCORER_AUTHENTICATED_FACTS:
            raise ScorerExtensionError(
                "authenticated scorer facts must contain exactly expected_output, "
                "output_text, and output_sha256"
            )
        expected_output = frozen.get("expected_output")
        output_text = frozen.get("output_text")
        output_sha256 = frozen.get("output_sha256")
        if not isinstance(expected_output, str) or not isinstance(output_text, str):
            raise ScorerExtensionError(
                "authenticated scorer expected_output and output_text must be strings"
            )
        digest = _require_sha256(output_sha256, field_name="output_sha256")
        if hashlib.sha256(output_text.encode("utf-8")).hexdigest() != digest:
            raise ScorerExtensionError(
                "authenticated scorer output_sha256 does not match output_text"
            )
        object.__setattr__(self, "facts", frozen)


def _source_record_payload(record: AuthenticatedScorerRecord) -> dict[str, object]:
    facts = _thaw_json(record.facts)
    assert isinstance(facts, dict)
    return {
        "facts": facts,
        "input_sha256": record.input_sha256,
        "record_id": record.record_id,
    }


@dataclass(frozen=True)
class ScorerReplayRequest:
    """Verifier-owned replay input assembled from authenticated evidence."""

    binding: ScorerExtensionBinding
    task: str
    input_kinds: tuple[str, ...]
    output_kind: str
    schedule_sha256: str
    records: tuple[AuthenticatedScorerRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ScorerExtensionBinding):
            raise ScorerExtensionError("binding must be a ScorerExtensionBinding")
        _require_name(self.task, field_name="task")
        _require_name_tuple(
            self.input_kinds, field_name="input_kinds", allowed=_INPUT_KINDS
        )
        _require_name(self.output_kind, field_name="output_kind")
        _require_sha256(self.schedule_sha256, field_name="schedule_sha256")
        if not isinstance(self.records, tuple) or not self.records:
            raise ScorerExtensionError("records must be a non-empty tuple")
        if not all(
            isinstance(record, AuthenticatedScorerRecord) for record in self.records
        ):
            raise ScorerExtensionError(
                "records must contain AuthenticatedScorerRecord values"
            )
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ScorerExtensionError("records must not contain duplicate record IDs")

    @property
    def source_records_sha256(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                [_source_record_payload(record) for record in self.records]
            )
        )


@dataclass(frozen=True)
class ScorerRecordResult:
    """One replayed numeric value bound to its authenticated record."""

    record_id: str
    input_sha256: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ScorerExtensionError("record result ID must be non-empty")
        _require_sha256(self.input_sha256, field_name="record result input_sha256")
        object.__setattr__(
            self,
            "value",
            _unit_interval_score(self.value, field_name="record result value"),
        )


def _record_result_payload(result: ScorerRecordResult) -> dict[str, object]:
    return {
        "input_sha256": result.input_sha256,
        "record_id": result.record_id,
        "value": result.value,
    }


def scorer_record_results_sha256(results: Sequence[ScorerRecordResult]) -> str:
    if not results:
        raise ScorerExtensionError("record results must not be empty")
    return _sha256(
        _canonical_json_bytes([_record_result_payload(result) for result in results])
    )


@dataclass(frozen=True)
class ScorerExtensionResult:
    """Deterministic verifier replay result from one extension scorer."""

    scorer_id: str
    scorer_version: str
    descriptor_sha256: str
    configuration_sha256: str
    schedule_sha256: str
    source_records_sha256: str
    record_results: tuple[ScorerRecordResult, ...]
    aggregate: float
    aggregate_source_sha256: str
    format_version: str = field(default=SCORER_EXTENSION_RESULT_FORMAT, init=False)
    scorer_abi: str = field(default=SCORER_EXTENSION_ABI_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_scorer_id(self.scorer_id, field_name="scorer_id")
        _require_version(self.scorer_version, field_name="scorer_version")
        _require_sha256(self.descriptor_sha256, field_name="descriptor_sha256")
        _require_sha256(self.configuration_sha256, field_name="configuration_sha256")
        _require_sha256(self.schedule_sha256, field_name="schedule_sha256")
        _require_sha256(self.source_records_sha256, field_name="source_records_sha256")
        if not isinstance(self.record_results, tuple) or not self.record_results:
            raise ScorerExtensionError("record_results must be a non-empty tuple")
        if not all(
            isinstance(result, ScorerRecordResult) for result in self.record_results
        ):
            raise ScorerExtensionError(
                "record_results must contain ScorerRecordResult values"
            )
        record_ids = [result.record_id for result in self.record_results]
        if len(record_ids) != len(set(record_ids)):
            raise ScorerExtensionError(
                "record_results must not contain duplicate record IDs"
            )
        object.__setattr__(
            self,
            "aggregate",
            _unit_interval_score(self.aggregate, field_name="aggregate"),
        )
        expected_aggregate = math.fsum(
            result.value for result in self.record_results
        ) / len(self.record_results)
        if self.aggregate != expected_aggregate:
            raise ScorerExtensionError(
                "aggregate must equal the core arithmetic mean of record results"
            )
        _require_sha256(
            self.aggregate_source_sha256, field_name="aggregate_source_sha256"
        )
        if self.aggregate_source_sha256 != scorer_record_results_sha256(
            self.record_results
        ):
            raise ScorerExtensionError(
                "aggregate_source_sha256 does not match record_results"
            )


def build_scorer_result(
    request: ScorerReplayRequest,
    values: Sequence[int | float],
) -> ScorerExtensionResult:
    """Bind record values and compute the sole core-owned aggregate."""

    if len(values) != len(request.records):
        raise ScorerExtensionError("one result value is required for every record")
    record_results = tuple(
        ScorerRecordResult(
            record_id=record.record_id,
            input_sha256=record.input_sha256,
            value=_unit_interval_score(value, field_name="record result value"),
        )
        for record, value in zip(request.records, values, strict=True)
    )
    return ScorerExtensionResult(
        scorer_id=request.binding.scorer_id,
        scorer_version=request.binding.scorer_version,
        descriptor_sha256=request.binding.descriptor_sha256,
        configuration_sha256=request.binding.configuration_sha256,
        schedule_sha256=request.schedule_sha256,
        source_records_sha256=request.source_records_sha256,
        record_results=record_results,
        aggregate=math.fsum(result.value for result in record_results)
        / len(record_results),
        aggregate_source_sha256=scorer_record_results_sha256(record_results),
    )


def scorer_result_payload(result: ScorerExtensionResult) -> dict[str, object]:
    if not isinstance(result, ScorerExtensionResult):
        raise ScorerExtensionError("result must be a ScorerExtensionResult")
    return {
        "aggregate": result.aggregate,
        "aggregate_source_sha256": result.aggregate_source_sha256,
        "configuration_sha256": result.configuration_sha256,
        "descriptor_sha256": result.descriptor_sha256,
        "format_version": result.format_version,
        "record_results": [
            _record_result_payload(record_result)
            for record_result in result.record_results
        ],
        "schedule_sha256": result.schedule_sha256,
        "scorer_abi": result.scorer_abi,
        "scorer_id": result.scorer_id,
        "scorer_version": result.scorer_version,
        "source_records_sha256": result.source_records_sha256,
    }


def canonical_scorer_result_json(result: ScorerExtensionResult) -> bytes:
    return _canonical_json_bytes(scorer_result_payload(result))


class VerifierReplayScorer(Protocol):
    """Pure scorer implemented by one authorized optional package."""

    abi_version: str

    def descriptor(self) -> ScorerExtensionDescriptor: ...

    def configuration_schema(self) -> Mapping[str, object]: ...

    def replay(self, request: ScorerReplayRequest) -> ScorerExtensionResult: ...


@dataclass(frozen=True)
class _EntryPointInfo:
    scorer_id: str
    module: str
    class_name: str
    entry_point: EntryPoint


@dataclass(frozen=True)
class _LoadedScorer:
    scorer: VerifierReplayScorer
    descriptor: ScorerExtensionDescriptor
    configuration_schema_json: bytes


def _select_scorer_entry_points(values: Any) -> list[EntryPoint]:
    if hasattr(values, "select"):
        return list(values.select(group=SCORER_EXTENSION_ENTRY_POINT_GROUP))
    return list(values.get(SCORER_EXTENSION_ENTRY_POINT_GROUP, []))


class ScorerExtensionRegistry:
    """Lazy registry for explicitly authorized installed scorer extensions."""

    def __init__(
        self,
        *,
        allow_installed: bool | None = None,
        authorized: Sequence[VerifierReplayScorer] = (),
    ) -> None:
        self._allow_installed = (
            third_party_plugins_allowed()
            if allow_installed is None
            else allow_installed
        )
        self._entries: dict[str, _EntryPointInfo] = {}
        self._authorized: dict[str, VerifierReplayScorer] = {}
        for scorer in authorized:
            descriptor = scorer.descriptor()
            if not isinstance(descriptor, ScorerExtensionDescriptor):
                raise ScorerExtensionError(
                    "authorized scorer descriptor must be a ScorerExtensionDescriptor"
                )
            if descriptor.scorer_id in self._authorized:
                raise ScorerExtensionError(
                    f"duplicate scorer extension ID: {descriptor.scorer_id}"
                )
            self._authorized[descriptor.scorer_id] = scorer
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._discover()
            self._initialized = True

    def _discover(self) -> None:
        if not self._allow_installed:
            return
        try:
            candidates = _select_scorer_entry_points(entry_points())
        except Exception as exc:
            raise ScorerExtensionError(
                f"scorer extension discovery failed: {exc}"
            ) from exc
        for candidate in candidates:
            scorer_id = _require_scorer_id(
                candidate.name, field_name="entry-point name"
            )
            if scorer_id in self._entries:
                raise ScorerExtensionError(
                    f"duplicate scorer extension ID: {scorer_id}"
                )
            value = getattr(candidate, "value", None)
            if not isinstance(value, str):
                raise ScorerExtensionError(
                    f"scorer extension {scorer_id!r} has a non-string entry point"
                )
            module, separator, class_name = value.partition(":")
            if not module or separator != ":" or not class_name or ":" in class_name:
                raise ScorerExtensionError(
                    f"scorer extension {scorer_id!r} has a malformed entry point"
                )
            self._entries[scorer_id] = _EntryPointInfo(
                scorer_id=scorer_id,
                module=module,
                class_name=class_name,
                entry_point=candidate,
            )

    def list_scorers(self) -> tuple[str, ...]:
        """List installed extension IDs without importing their code."""

        self._ensure_initialized()
        return tuple(sorted({*self._entries, *self._authorized}))

    def _load(self, scorer_id: str) -> _LoadedScorer:
        self._ensure_initialized()
        entry = self._entries.get(scorer_id)
        authorized = self._authorized.get(scorer_id)
        if entry is None and authorized is None:
            raise ScorerExtensionError(
                f"required scorer extension {scorer_id!r} is not installed or enabled"
            )
        try:
            if authorized is not None:
                instance = authorized
            else:
                assert entry is not None
                scorer_class = entry.entry_point.load()
                if not isinstance(scorer_class, type):
                    raise TypeError("entry point must resolve to a scorer class")
                module = importlib.import_module(scorer_class.__module__)
                module_abi = getattr(module, "INVARLOCK_SCORER_EXTENSION_ABI", None)
                if module_abi != SCORER_EXTENSION_ABI_VERSION:
                    raise TypeError(
                        f"module ABI {module_abi!r} does not match "
                        f"{SCORER_EXTENSION_ABI_VERSION!r}"
                    )
                instance = scorer_class()
            if getattr(instance, "abi_version", None) != SCORER_EXTENSION_ABI_VERSION:
                raise TypeError("scorer instance ABI does not match core")
            descriptor_method = getattr(instance, "descriptor", None)
            schema_method = getattr(instance, "configuration_schema", None)
            replay_method = getattr(instance, "replay", None)
            if not all(
                callable(method)
                for method in (descriptor_method, schema_method, replay_method)
            ):
                raise TypeError(
                    "scorer must implement descriptor, configuration_schema, and replay"
                )
            descriptor_function = cast(Callable[[], object], descriptor_method)
            schema_function = cast(Callable[[], object], schema_method)
            first_descriptor = descriptor_function()
            second_descriptor = descriptor_function()
            if not isinstance(
                first_descriptor, ScorerExtensionDescriptor
            ) or not isinstance(second_descriptor, ScorerExtensionDescriptor):
                raise TypeError("descriptor must return ScorerExtensionDescriptor")
            first_descriptor_json = canonical_scorer_descriptor_json(first_descriptor)
            if first_descriptor_json != canonical_scorer_descriptor_json(
                second_descriptor
            ):
                raise TypeError("scorer descriptor is nondeterministic")
            if first_descriptor.scorer_id != scorer_id:
                raise TypeError(
                    "scorer descriptor ID does not match entry-point or authorized ID"
                )
            first_schema = schema_function()
            second_schema = schema_function()
            if not isinstance(first_schema, Mapping) or not isinstance(
                second_schema, Mapping
            ):
                raise TypeError("configuration_schema must return a JSON object")
            first_schema_frozen = _freeze_json(
                first_schema, field_name="configuration schema"
            )
            second_schema_frozen = _freeze_json(
                second_schema, field_name="configuration schema"
            )
            first_schema_json = _canonical_json_bytes(first_schema_frozen)
            if first_schema_json != _canonical_json_bytes(second_schema_frozen):
                raise TypeError("scorer configuration schema is nondeterministic")
            if (
                _sha256(first_schema_json)
                != first_descriptor.configuration_schema_sha256
            ):
                raise TypeError("configuration schema does not match descriptor digest")
            schema_payload = json.loads(first_schema_json)
            if not isinstance(schema_payload, dict):
                raise TypeError("configuration schema must decode to an object")
            Draft202012Validator.check_schema(schema_payload)
        except Exception as exc:
            if isinstance(exc, ScorerExtensionError):
                raise
            raise ScorerExtensionError(
                f"failed to load scorer extension {scorer_id!r}: {exc}"
            ) from exc
        return _LoadedScorer(
            scorer=instance,
            descriptor=first_descriptor,
            configuration_schema_json=first_schema_json,
        )

    @staticmethod
    def _validate_request(loaded: _LoadedScorer, request: ScorerReplayRequest) -> None:
        descriptor = loaded.descriptor
        binding = request.binding
        if (
            binding.scorer_id != descriptor.scorer_id
            or binding.scorer_version != descriptor.scorer_version
            or binding.descriptor_sha256 != scorer_descriptor_sha256(descriptor)
            or binding.scorer_abi != descriptor.scorer_abi
        ):
            raise ScorerExtensionError(
                "scorer binding does not match the installed descriptor"
            )
        if request.task not in descriptor.supported_tasks:
            raise ScorerExtensionError(f"scorer does not support task {request.task!r}")
        if not set(request.input_kinds).issubset(descriptor.supported_input_kinds):
            raise ScorerExtensionError(
                "scorer does not support the request input kinds"
            )
        if request.output_kind not in descriptor.supported_output_kinds:
            raise ScorerExtensionError(
                f"scorer does not support output kind {request.output_kind!r}"
            )
        for record in request.records:
            missing = set(descriptor.required_facts) - set(record.facts)
            if missing:
                raise ScorerExtensionError(
                    f"record {record.record_id!r} lacks required fact "
                    f"{sorted(missing)[0]!r}"
                )
        schema_payload = json.loads(loaded.configuration_schema_json)
        if schema_payload.get("type") != "object":
            raise ScorerExtensionError(
                "scorer configuration schema must describe a JSON object"
            )
        configuration = _thaw_json(binding.configuration)
        errors = sorted(
            Draft202012Validator(schema_payload).iter_errors(configuration),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.absolute_path) or "<root>"
            raise ScorerExtensionError(
                f"scorer configuration violates its bound schema at {path}: "
                f"{error.message}"
            )

    @staticmethod
    def _validate_result(
        result: object,
        *,
        request: ScorerReplayRequest,
    ) -> ScorerExtensionResult:
        if not isinstance(result, ScorerExtensionResult):
            raise ScorerExtensionError(
                "scorer replay must return ScorerExtensionResult"
            )
        binding = request.binding
        if (
            result.scorer_id != binding.scorer_id
            or result.scorer_version != binding.scorer_version
            or result.descriptor_sha256 != binding.descriptor_sha256
            or result.configuration_sha256 != binding.configuration_sha256
            or result.schedule_sha256 != request.schedule_sha256
            or result.source_records_sha256 != request.source_records_sha256
        ):
            raise ScorerExtensionError(
                "scorer result does not match its replay request"
            )
        expected_pairing = tuple(
            (record.record_id, record.input_sha256) for record in request.records
        )
        observed_pairing = tuple(
            (record.record_id, record.input_sha256) for record in result.record_results
        )
        if observed_pairing != expected_pairing:
            raise ScorerExtensionError(
                "scorer result pairing does not match authenticated records"
            )
        return result

    def replay(self, request: ScorerReplayRequest) -> ScorerExtensionResult:
        """Replay twice and accept only an exact, valid deterministic result."""

        if not isinstance(request, ScorerReplayRequest):
            raise ScorerExtensionError("request must be a ScorerReplayRequest")
        loaded = self._load(request.binding.scorer_id)
        self._validate_request(loaded, request)
        try:
            first = self._validate_result(
                loaded.scorer.replay(request), request=request
            )
            second = self._validate_result(
                loaded.scorer.replay(request), request=request
            )
        except ScorerExtensionError:
            raise
        except Exception as exc:
            raise ScorerExtensionError(f"scorer replay failed closed: {exc}") from exc
        if canonical_scorer_result_json(first) != canonical_scorer_result_json(second):
            raise ScorerExtensionError("scorer replay is nondeterministic")
        current_descriptor = loaded.scorer.descriptor()
        if not isinstance(
            current_descriptor, ScorerExtensionDescriptor
        ) or canonical_scorer_descriptor_json(
            current_descriptor
        ) != canonical_scorer_descriptor_json(loaded.descriptor):
            raise ScorerExtensionError("scorer descriptor changed during replay")
        return first


__all__ = [
    "AuthenticatedScorerRecord",
    "CANONICAL_BUILTIN_ACCEPTANCE_METRICS",
    "MAX_SCORER_CONFIGURATION_BYTES",
    "SCORER_EXTENSION_ABI_VERSION",
    "SCORER_EXTENSION_BINDING_FORMAT",
    "SCORER_EXTENSION_DESCRIPTOR_FORMAT",
    "SCORER_EXTENSION_ENTRY_POINT_GROUP",
    "SCORER_EXTENSION_RESULT_FORMAT",
    "SCORER_AGGREGATION",
    "SCORER_AUTHENTICATED_FACTS",
    "SCORER_DIRECTION",
    "SCORER_RECORD_VALUE_SEMANTICS",
    "SCORER_REPLAY_MODE",
    "ScorerExtensionBinding",
    "ScorerExtensionDescriptor",
    "ScorerExtensionError",
    "ScorerExtensionRegistry",
    "ScorerExtensionResult",
    "ScorerRecordResult",
    "ScorerReplayRequest",
    "VerifierReplayScorer",
    "build_scorer_binding",
    "build_scorer_result",
    "canonical_scorer_binding_json",
    "canonical_scorer_descriptor_json",
    "canonical_scorer_result_json",
    "decode_scorer_binding",
    "scorer_binding_payload",
    "scorer_configuration_schema_sha256",
    "scorer_descriptor_payload",
    "scorer_descriptor_sha256",
    "scorer_record_results_sha256",
    "scorer_result_payload",
]
