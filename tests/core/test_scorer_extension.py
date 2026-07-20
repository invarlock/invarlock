from __future__ import annotations

import dataclasses
import hashlib
import sys
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import cast

import pytest
from jsonschema import Draft202012Validator

import invarlock.core.scorer_extension as scorer_module
from invarlock.core.scorer_extension import (
    CANONICAL_BUILTIN_ACCEPTANCE_METRICS,
    SCORER_EXTENSION_ABI_VERSION,
    AuthenticatedScorerRecord,
    ScorerExtensionBinding,
    ScorerExtensionDescriptor,
    ScorerExtensionError,
    ScorerExtensionRegistry,
    ScorerExtensionResult,
    ScorerRecordResult,
    ScorerReplayRequest,
    build_scorer_binding,
    build_scorer_result,
    scorer_binding_payload,
    scorer_configuration_schema_sha256,
    scorer_descriptor_payload,
    scorer_record_results_sha256,
    scorer_result_payload,
)
from invarlock.public_contracts import (
    load_scorer_extension_binding_schema,
    load_scorer_extension_descriptor_schema,
    load_scorer_extension_result_schema,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_SCORER_ID = "example.structured_score"

_CONFIGURATION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["scale"],
    "properties": {"scale": {"type": "number", "minimum": 0.0}},
}


class _EntryPoint:
    def __init__(self, name: str, value: str, scorer_class: type[object]) -> None:
        self.name = name
        self.value = value
        self._scorer_class = scorer_class
        self.load_count = 0

    def load(self) -> type[object]:
        self.load_count += 1
        return self._scorer_class


class _EntryPoints:
    def __init__(self, values: list[_EntryPoint]) -> None:
        self._values = values

    def select(self, *, group: str) -> list[_EntryPoint]:
        if group == scorer_module.SCORER_EXTENSION_ENTRY_POINT_GROUP:
            return list(self._values)
        return []


def _descriptor(**updates: object) -> ScorerExtensionDescriptor:
    values: dict[str, object] = {
        "scorer_id": _SCORER_ID,
        "scorer_version": "1.0.0",
        "supported_tasks": ("text_causal",),
        "supported_input_kinds": ("text",),
        "supported_output_kinds": ("text",),
        "required_facts": ("expected_output", "output_text", "output_sha256"),
        "configuration_schema_sha256": scorer_configuration_schema_sha256(
            _CONFIGURATION_SCHEMA
        ),
    }
    values.update(updates)
    return ScorerExtensionDescriptor(**values)  # type: ignore[arg-type]


def _facts(output_text: str) -> dict[str, object]:
    return {
        "expected_output": "target",
        "output_text": output_text,
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }


def _request(
    *,
    descriptor: ScorerExtensionDescriptor | None = None,
    configuration: Mapping[str, object] | None = None,
    task: str = "text_causal",
    input_kinds: tuple[str, ...] = ("text",),
    output_kind: str = "text",
    facts: tuple[Mapping[str, object], ...] | None = None,
) -> ScorerReplayRequest:
    selected_descriptor = descriptor or _descriptor()
    selected_facts = facts or (_facts("0.25"), _facts("0.5"))
    return ScorerReplayRequest(
        binding=build_scorer_binding(
            selected_descriptor, configuration or {"scale": 2.0}
        ),
        task=task,
        input_kinds=input_kinds,
        output_kind=output_kind,
        schedule_sha256=_DIGEST_A,
        records=tuple(
            AuthenticatedScorerRecord(
                record_id=f"row-{index}",
                input_sha256=character * 64,
                facts=record_facts,
            )
            for index, (character, record_facts) in enumerate(
                zip(("c", "d"), selected_facts, strict=True), start=1
            )
        ),
    )


def _valid_replay(request: ScorerReplayRequest) -> ScorerExtensionResult:
    scale = cast(float, request.binding.configuration["scale"])
    values = [
        float(cast(str, record.facts["output_text"])) * scale
        for record in request.records
    ]
    return build_scorer_result(request, values)


def _install_scorer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scorer_id: str = _SCORER_ID,
    module_abi: str = SCORER_EXTENSION_ABI_VERSION,
    instance_abi: str = SCORER_EXTENSION_ABI_VERSION,
    descriptor: Callable[[], object] | None = None,
    configuration_schema: Callable[[], object] | None = None,
    replay: Callable[[ScorerReplayRequest], object] | None = None,
    include_protocol: bool = True,
    duplicate: bool = False,
) -> tuple[ScorerExtensionRegistry, _EntryPoint]:
    module_name = f"test_scorer_extension_{len(sys.modules)}"

    class Scorer:
        abi_version = instance_abi

        if include_protocol:

            def descriptor(self) -> object:
                return (descriptor or _descriptor)()

            def configuration_schema(self) -> object:
                return (configuration_schema or (lambda: _CONFIGURATION_SCHEMA))()

            def replay(self, request: ScorerReplayRequest) -> object:
                return (replay or _valid_replay)(request)

    Scorer.__module__ = module_name
    module = ModuleType(module_name)
    module.__dict__["INVARLOCK_SCORER_EXTENSION_ABI"] = module_abi
    module.__dict__["Scorer"] = Scorer
    monkeypatch.setitem(sys.modules, module_name, module)
    entry = _EntryPoint(scorer_id, f"{module_name}:Scorer", Scorer)
    installed = [entry]
    if duplicate:
        installed.append(_EntryPoint(scorer_id, f"{module_name}:Scorer", Scorer))
    monkeypatch.setattr(scorer_module, "entry_points", lambda: _EntryPoints(installed))
    return ScorerExtensionRegistry(allow_installed=True), entry


def test_only_exact_match_and_normalized_nll_are_canonical_builtins() -> None:
    assert CANONICAL_BUILTIN_ACCEPTANCE_METRICS == {
        "exact_match",
        "normalized_nll_per_utf8_byte",
    }
    assert "relative_perplexity" not in CANONICAL_BUILTIN_ACCEPTANCE_METRICS


def test_descriptor_binding_and_result_match_shipped_schemas() -> None:
    descriptor = _descriptor()
    request = _request(descriptor=descriptor)
    result = _valid_replay(request)

    Draft202012Validator(load_scorer_extension_descriptor_schema()).validate(
        scorer_descriptor_payload(descriptor)
    )
    Draft202012Validator(load_scorer_extension_binding_schema()).validate(
        scorer_binding_payload(request.binding)
    )
    Draft202012Validator(load_scorer_extension_result_schema()).validate(
        scorer_result_payload(result)
    )


def test_public_scorer_helpers_reject_untyped_contract_values() -> None:
    with pytest.raises(ScorerExtensionError, match="descriptor must be"):
        scorer_descriptor_payload(cast(ScorerExtensionDescriptor, object()))
    with pytest.raises(ScorerExtensionError, match="schema must be a JSON object"):
        scorer_configuration_schema_sha256(cast(Mapping[str, object], object()))
    with pytest.raises(
        ScorerExtensionError, match="configuration must be a JSON object"
    ):
        build_scorer_binding(_descriptor(), cast(Mapping[str, object], []))
    with pytest.raises(ScorerExtensionError, match="binding must be a JSON object"):
        scorer_module.decode_scorer_binding(object())
    with pytest.raises(ScorerExtensionError, match="binding must be"):
        scorer_binding_payload(cast(ScorerExtensionBinding, object()))
    with pytest.raises(ScorerExtensionError, match="result must be"):
        scorer_result_payload(cast(ScorerExtensionResult, object()))
    with pytest.raises(ScorerExtensionError, match="must not be empty"):
        scorer_record_results_sha256(())


def test_authenticated_scorer_records_reject_untyped_and_unbound_facts() -> None:
    with pytest.raises(ScorerExtensionError, match="facts must be a JSON object"):
        AuthenticatedScorerRecord(
            record_id="row",
            input_sha256=_DIGEST_A,
            facts=cast(Mapping[str, object], object()),
        )
    with pytest.raises(ScorerExtensionError, match="must be strings"):
        AuthenticatedScorerRecord(
            record_id="row",
            input_sha256=_DIGEST_A,
            facts={
                "expected_output": 1,
                "output_text": "value",
                "output_sha256": hashlib.sha256(b"value").hexdigest(),
            },
        )
    with pytest.raises(ScorerExtensionError, match="does not match output_text"):
        AuthenticatedScorerRecord(
            record_id="row",
            input_sha256=_DIGEST_A,
            facts={
                "expected_output": "target",
                "output_text": "value",
                "output_sha256": _DIGEST_B,
            },
        )


def test_replay_and_record_results_require_typed_nonempty_identity() -> None:
    request = _request()
    with pytest.raises(ScorerExtensionError, match="binding must be"):
        dataclasses.replace(
            request,
            binding=cast(ScorerExtensionBinding, object()),
        )
    with pytest.raises(ScorerExtensionError, match="ID must be non-empty"):
        ScorerRecordResult(record_id="", input_sha256=_DIGEST_A, value=0.5)


@pytest.mark.parametrize(
    "updates",
    [
        {"uses_network": True},
        {"uses_external_model": True},
        {"uses_human_judgment": True},
    ],
)
def test_external_and_llm_judges_are_ineligible_for_acceptance(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ScorerExtensionError, match="not eligible"):
        _descriptor(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"record_value_semantics": "unbounded"},
        {"aggregation": "custom"},
        {"direction": "lower_is_better"},
    ],
)
def test_extensions_cannot_customize_value_aggregation_or_direction(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ScorerExtensionError, match="core arithmetic mean"):
        _descriptor(**updates)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"scorer_version": "latest"}, "major.minor.patch"),
        ({"supported_tasks": ()}, "non-empty tuple"),
        ({"supported_tasks": ("text_causal", "text_causal")}, "duplicates"),
        ({"supported_input_kinds": ("audio",)}, "unsupported value"),
        ({"supported_output_kinds": ("not-canonical",)}, "canonical identifier"),
        ({"replay_mode": "provider_aggregate"}, "authenticated per-record facts"),
    ],
)
def test_descriptor_identity_and_capabilities_are_canonical(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ScorerExtensionError, match=message):
        _descriptor(**updates)


def test_binding_authenticates_immutable_configuration() -> None:
    original: dict[str, object] = {"scale": 2.0, "labels": ["a", "b"]}
    binding = build_scorer_binding(_descriptor(), original)
    original["scale"] = 999.0
    cast(list[str], original["labels"]).append("c")

    assert binding.configuration["scale"] == 2.0
    assert binding.configuration["labels"] == ("a", "b")
    with pytest.raises(TypeError):
        binding.configuration["scale"] = 3.0  # type: ignore[index]
    with pytest.raises(ScorerExtensionError, match="does not match"):
        ScorerExtensionBinding(
            scorer_id=binding.scorer_id,
            scorer_version=binding.scorer_version,
            descriptor_sha256=binding.descriptor_sha256,
            configuration=binding.configuration,
            configuration_sha256=_DIGEST_B,
        )


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ({"scale": float("inf")}, "non-finite"),
        ({1: "value"}, "non-string key"),
        ({"scale": object()}, "non-JSON value"),
        ({"payload": "x" * 65_536}, "size limit"),
    ],
)
def test_binding_rejects_ambiguous_or_unbounded_configuration(
    configuration: Mapping[object, object], message: str
) -> None:
    with pytest.raises(ScorerExtensionError, match=message):
        build_scorer_binding(_descriptor(), cast(Mapping[str, object], configuration))


def test_authenticated_record_facts_are_deeply_immutable_and_digest_bound() -> None:
    source = _facts("0.25")
    request = _request(facts=(source, _facts("0.5")))
    initial_digest = request.source_records_sha256
    source["output_text"] = "changed"

    assert request.source_records_sha256 == initial_digest
    assert request.records[0].facts["output_text"] == "0.25"


@pytest.mark.parametrize(
    ("record_id", "facts", "message"),
    [
        (" bad ", _facts("0.5"), "safe string"),
        ("row", {"not-canonical": 0.5}, "canonical identifier"),
    ],
)
def test_authenticated_records_reject_ambiguous_identity_or_facts(
    record_id: str, facts: Mapping[str, object], message: str
) -> None:
    with pytest.raises(ScorerExtensionError, match=message):
        AuthenticatedScorerRecord(
            record_id=record_id, input_sha256=_DIGEST_A, facts=facts
        )


def test_replay_request_rejects_empty_nonrecord_and_duplicate_records() -> None:
    request = _request()
    with pytest.raises(ScorerExtensionError, match="non-empty tuple"):
        dataclasses.replace(request, records=())
    with pytest.raises(ScorerExtensionError, match="AuthenticatedScorerRecord"):
        dataclasses.replace(request, records=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ScorerExtensionError, match="duplicate record IDs"):
        dataclasses.replace(request, records=(request.records[0], request.records[0]))


def test_registry_is_lazy_and_replays_valid_extension_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def replay(request: ScorerReplayRequest) -> ScorerExtensionResult:
        nonlocal calls
        calls += 1
        return _valid_replay(request)

    registry, entry = _install_scorer(monkeypatch, replay=replay)

    assert registry.list_scorers() == (_SCORER_ID,)
    assert entry.load_count == 0
    result = registry.replay(_request())

    assert entry.load_count == 1
    assert calls == 2
    assert result.aggregate == 0.75
    assert [record.value for record in result.record_results] == [0.5, 1.0]


def test_registry_qualifies_exact_binding_without_replaying_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_calls = 0

    def replay(request: ScorerReplayRequest) -> ScorerExtensionResult:
        nonlocal replay_calls
        replay_calls += 1
        return _valid_replay(request)

    registry, entry = _install_scorer(monkeypatch, replay=replay)
    binding = _request().binding

    registry.validate_binding(
        binding,
        task="text_causal",
        input_kinds=("text",),
        output_kind="text",
    )

    assert entry.load_count == 1
    assert replay_calls == 0


def test_registry_binding_qualification_rejects_installed_descriptor_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _request().binding
    registry, _entry = _install_scorer(
        monkeypatch,
        descriptor=lambda: _descriptor(scorer_version="2.0.0"),
    )

    with pytest.raises(ScorerExtensionError, match="installed descriptor"):
        registry.validate_binding(
            binding,
            task="text_causal",
            input_kinds=("text",),
            output_kind="text",
        )


def test_registry_binding_qualification_requires_typed_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _entry = _install_scorer(monkeypatch)

    with pytest.raises(ScorerExtensionError, match="ScorerExtensionBinding"):
        registry.validate_binding(
            cast(ScorerExtensionBinding, object()),
            task="text_causal",
            input_kinds=("text",),
            output_kind="text",
        )


@pytest.mark.parametrize(
    ("task", "input_kinds", "output_kind", "configuration", "message"),
    [
        ("vision_text_generation", ("text",), "text", {"scale": 2.0}, "task"),
        ("text_causal", ("content",), "text", {"scale": 2.0}, "input kinds"),
        ("text_causal", ("text",), "structured", {"scale": 2.0}, "output kind"),
        (
            "text_causal",
            ("text",),
            "text",
            {"scale": -1.0},
            "bound schema",
        ),
    ],
)
def test_registry_binding_qualification_rejects_incompatible_request(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    input_kinds: tuple[str, ...],
    output_kind: str,
    configuration: dict[str, object],
    message: str,
) -> None:
    registry, _entry = _install_scorer(monkeypatch)
    binding = build_scorer_binding(_descriptor(), configuration)

    with pytest.raises(ScorerExtensionError, match=message):
        registry.validate_binding(
            binding,
            task=task,
            input_kinds=input_kinds,
            output_kind=output_kind,
        )


def test_registry_requires_explicit_installed_plugin_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, entry = _install_scorer(monkeypatch)
    registry = ScorerExtensionRegistry(allow_installed=False)

    assert registry.list_scorers() == ()
    assert entry.load_count == 0
    with pytest.raises(ScorerExtensionError, match="not installed or enabled"):
        registry.replay(_request())


def test_registry_fails_closed_for_missing_duplicate_and_malformed_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer_module, "entry_points", lambda: _EntryPoints([]))
    with pytest.raises(ScorerExtensionError, match="not installed or enabled"):
        ScorerExtensionRegistry(allow_installed=True).replay(_request())

    duplicate_registry, _ = _install_scorer(monkeypatch, duplicate=True)
    with pytest.raises(ScorerExtensionError, match="duplicate scorer extension"):
        duplicate_registry.list_scorers()

    malformed_registry, _ = _install_scorer(monkeypatch, scorer_id="not-a-stable-id")
    with pytest.raises(ScorerExtensionError, match="dotted canonical"):
        malformed_registry.list_scorers()


def test_registry_rejects_discovery_errors_and_malformed_entry_point_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_discovery() -> object:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(scorer_module, "entry_points", broken_discovery)
    with pytest.raises(ScorerExtensionError, match="discovery failed"):
        ScorerExtensionRegistry(allow_installed=True).list_scorers()

    registry, entry = _install_scorer(monkeypatch)
    entry.value = "module-without-class"
    with pytest.raises(ScorerExtensionError, match="malformed entry point"):
        registry.list_scorers()


def test_registry_supports_legacy_entry_point_metadata_shape() -> None:
    entry = _EntryPoint(_SCORER_ID, "example.module:Scorer", object)
    selected = scorer_module._select_scorer_entry_points(  # noqa: SLF001
        {scorer_module.SCORER_EXTENSION_ENTRY_POINT_GROUP: [entry]}
    )
    assert selected == [entry]


@pytest.mark.parametrize(
    ("module_abi", "instance_abi"),
    [("9", "1"), ("1", "9")],
)
def test_registry_rejects_module_and_instance_abi_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    module_abi: str,
    instance_abi: str,
) -> None:
    registry, _ = _install_scorer(
        monkeypatch, module_abi=module_abi, instance_abi=instance_abi
    )
    with pytest.raises(ScorerExtensionError, match="ABI"):
        registry.replay(_request())


def test_registry_rejects_incomplete_protocol_and_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete, _ = _install_scorer(monkeypatch, include_protocol=False)
    with pytest.raises(ScorerExtensionError, match="must implement"):
        incomplete.replay(_request())

    mismatch, _ = _install_scorer(
        monkeypatch,
        scorer_id="example.different",
    )
    with pytest.raises(ScorerExtensionError, match="does not match entry-point"):
        mismatch.replay(_request(descriptor=_descriptor(scorer_id="example.different")))


def test_registry_rejects_nonclass_descriptor_and_schema_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, entry = _install_scorer(monkeypatch)
    entry._scorer_class = cast(type[object], object())
    with pytest.raises(ScorerExtensionError, match="scorer class"):
        registry.replay(_request())

    registry, _ = _install_scorer(monkeypatch, descriptor=lambda: {})
    with pytest.raises(ScorerExtensionError, match="descriptor must return"):
        registry.replay(_request())

    registry, _ = _install_scorer(monkeypatch, configuration_schema=lambda: [])
    with pytest.raises(ScorerExtensionError, match="must return a JSON object"):
        registry.replay(_request())


def test_registry_rejects_invalid_or_nonobject_configuration_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_schema = {"type": "not-a-json-schema-type"}
    invalid_descriptor = _descriptor(
        configuration_schema_sha256=scorer_configuration_schema_sha256(invalid_schema)
    )
    registry, _ = _install_scorer(
        monkeypatch,
        descriptor=lambda: invalid_descriptor,
        configuration_schema=lambda: invalid_schema,
    )
    with pytest.raises(ScorerExtensionError, match="not-a-json-schema-type"):
        registry.replay(_request(descriptor=invalid_descriptor))

    nonobject_schema: dict[str, object] = {}
    nonobject_descriptor = _descriptor(
        configuration_schema_sha256=scorer_configuration_schema_sha256(nonobject_schema)
    )
    registry, _ = _install_scorer(
        monkeypatch,
        descriptor=lambda: nonobject_descriptor,
        configuration_schema=lambda: nonobject_schema,
    )
    with pytest.raises(ScorerExtensionError, match="must describe a JSON object"):
        registry.replay(_request(descriptor=nonobject_descriptor))


def test_registry_rejects_nondeterministic_descriptor_and_configuration_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_call = 0

    def changing_descriptor() -> ScorerExtensionDescriptor:
        nonlocal descriptor_call
        descriptor_call += 1
        return _descriptor(scorer_version=f"1.0.{descriptor_call}")

    registry, _ = _install_scorer(monkeypatch, descriptor=changing_descriptor)
    with pytest.raises(ScorerExtensionError, match="descriptor is nondeterministic"):
        registry.replay(_request())

    schema_call = 0

    def changing_schema() -> dict[str, object]:
        nonlocal schema_call
        schema_call += 1
        return {**_CONFIGURATION_SCHEMA, "title": f"schema-{schema_call}"}

    registry, _ = _install_scorer(monkeypatch, configuration_schema=changing_schema)
    with pytest.raises(
        ScorerExtensionError, match="configuration schema is nondeterministic"
    ):
        registry.replay(_request())


def test_registry_validates_schema_digest_and_concrete_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_digest, _ = _install_scorer(
        monkeypatch,
        descriptor=lambda: _descriptor(configuration_schema_sha256=_DIGEST_B),
    )
    with pytest.raises(ScorerExtensionError, match="does not match descriptor"):
        bad_digest.replay(_request())

    registry, _ = _install_scorer(monkeypatch)
    with pytest.raises(ScorerExtensionError, match="bound schema"):
        registry.replay(_request(configuration={"scale": -1.0}))


@pytest.mark.parametrize(
    ("request_updates", "message"),
    [
        ({"task": "text_seq2seq"}, "does not support task"),
        ({"input_kinds": ("content",)}, "input kinds"),
        ({"output_kind": "execution_result"}, "output kind"),
        ({"facts": ({"other": 1}, _facts("0.5"))}, "facts must contain exactly"),
    ],
)
def test_registry_rejects_unsupported_request_contract(
    monkeypatch: pytest.MonkeyPatch,
    request_updates: dict[str, object],
    message: str,
) -> None:
    registry, _ = _install_scorer(monkeypatch)
    with pytest.raises(ScorerExtensionError, match=message):
        registry.replay(_request(**request_updates))  # type: ignore[arg-type]


def test_registry_rejects_invalid_rebound_and_mispaired_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid, _ = _install_scorer(monkeypatch, replay=lambda _request: {"value": 1})
    with pytest.raises(ScorerExtensionError, match="must return"):
        invalid.replay(_request())

    def rebound(request: ScorerReplayRequest) -> ScorerExtensionResult:
        return dataclasses.replace(_valid_replay(request), schedule_sha256=_DIGEST_B)

    registry, _ = _install_scorer(monkeypatch, replay=rebound)
    with pytest.raises(ScorerExtensionError, match="does not match its replay"):
        registry.replay(_request())

    def mispaired(request: ScorerReplayRequest) -> ScorerExtensionResult:
        result = _valid_replay(request)
        reversed_records = tuple(reversed(result.record_results))
        return dataclasses.replace(
            result,
            record_results=reversed_records,
            aggregate_source_sha256=scorer_record_results_sha256(reversed_records),
        )

    registry, _ = _install_scorer(monkeypatch, replay=mispaired)
    with pytest.raises(ScorerExtensionError, match="pairing"):
        registry.replay(_request())


def test_registry_rejects_nondeterministic_or_exceptional_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = 0

    def changing(request: ScorerReplayRequest) -> ScorerExtensionResult:
        nonlocal call
        call += 1
        values = [0.25 * call, 0.25 * call]
        return build_scorer_result(request, values)

    registry, _ = _install_scorer(monkeypatch, replay=changing)
    with pytest.raises(ScorerExtensionError, match="nondeterministic"):
        registry.replay(_request())

    def broken(_request: ScorerReplayRequest) -> ScorerExtensionResult:
        raise RuntimeError("boom")

    registry, _ = _install_scorer(monkeypatch, replay=broken)
    with pytest.raises(ScorerExtensionError, match="failed closed"):
        registry.replay(_request())


def test_result_rejects_nonfinite_values_and_unbound_aggregate_source() -> None:
    with pytest.raises(ScorerExtensionError, match="finite number"):
        ScorerRecordResult(record_id="row", input_sha256=_DIGEST_A, value=float("nan"))

    result = _valid_replay(_request())
    with pytest.raises(ScorerExtensionError, match="between zero and one"):
        ScorerRecordResult(record_id="row", input_sha256=_DIGEST_A, value=1.01)
    with pytest.raises(ScorerExtensionError, match="arithmetic mean"):
        dataclasses.replace(result, aggregate=0.5)
    with pytest.raises(ScorerExtensionError, match="does not match record_results"):
        dataclasses.replace(result, aggregate_source_sha256=_DIGEST_B)


def test_result_and_builder_reject_missing_or_duplicate_record_values() -> None:
    request = _request()
    with pytest.raises(ScorerExtensionError, match="every record"):
        build_scorer_result(request, [0.5])
    result = _valid_replay(request)
    with pytest.raises(ScorerExtensionError, match="duplicate record IDs"):
        duplicated = (result.record_results[0], result.record_results[0])
        dataclasses.replace(
            result,
            record_results=duplicated,
            aggregate=0.5,
            aggregate_source_sha256=scorer_record_results_sha256(duplicated),
        )


def test_registry_rejects_descriptor_drift_during_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def drifting_descriptor() -> ScorerExtensionDescriptor:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return _descriptor()
        return _descriptor(scorer_version="1.0.1")

    registry, _ = _install_scorer(monkeypatch, descriptor=drifting_descriptor)
    with pytest.raises(ScorerExtensionError, match="changed during replay"):
        registry.replay(_request())
