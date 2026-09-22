from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from invarlock import public_contracts

FIXTURES = Path(__file__).parents[2] / "fixtures" / "judge_measurements"


@pytest.mark.parametrize("artifact", ["plan", "measurements"])
def test_judge_contract_golden_valid(artifact: str) -> None:
    loader = getattr(
        public_contracts,
        f"load_judge_measurement{'_plan' if artifact == 'plan' else 's'}_schema",
    )
    schema = loader()
    Draft202012Validator.check_schema(schema)
    payload = json.loads((FIXTURES / f"{artifact}.json").read_text())
    Draft202012Validator(schema).validate(payload)
    assert loader() == schema
    assert loader() is not schema


@pytest.mark.parametrize("artifact", ["plan", "measurements"])
def test_judge_contract_rejects_unknown_fields_and_versions(artifact: str) -> None:
    loader = getattr(
        public_contracts,
        f"load_judge_measurement{'_plan' if artifact == 'plan' else 's'}_schema",
    )
    validator = Draft202012Validator(loader())
    payload = json.loads((FIXTURES / f"{artifact}.json").read_text())
    for field, value in [("format", "invarlock/unknown-v1"), ("hidden_trial", True)]:
        invalid = copy.deepcopy(payload)
        invalid[field] = value
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_judge_contract_golden_invalid() -> None:
    mutations = json.loads((FIXTURES / "invalid.json").read_text())
    for case in mutations:
        artifact = case["artifact"]
        loader = getattr(
            public_contracts,
            f"load_judge_measurement{'_plan' if artifact == 'plan' else 's'}_schema",
        )
        payload = json.loads((FIXTURES / f"{artifact}.json").read_text())
        target = payload
        for part in case["path"][:-1]:
            target = target[part]
        target[case["path"][-1]] = case["value"]
        with pytest.raises(ValidationError):
            Draft202012Validator(loader()).validate(payload)


@pytest.mark.parametrize("artifact", ["plan", "measurements"])
def test_judge_contract_requires_every_binding(artifact: str) -> None:
    loader = getattr(
        public_contracts,
        f"load_judge_measurement{'_plan' if artifact == 'plan' else 's'}_schema",
    )
    validator = Draft202012Validator(loader())
    payload = json.loads((FIXTURES / f"{artifact}.json").read_text())

    def check_object_fields(value: object, path: tuple[str | int, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                invalid = copy.deepcopy(payload)
                target = invalid
                for segment in path:
                    target = target[segment]
                del target[key]
                with pytest.raises(ValidationError):
                    validator.validate(invalid)
                check_object_fields(child, (*path, key))
            invalid = copy.deepcopy(payload)
            target = invalid
            for segment in path:
                target = target[segment]
            target["unapproved"] = True
            with pytest.raises(ValidationError):
                validator.validate(invalid)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_object_fields(child, (*path, index))

    check_object_fields(payload, ())


def test_judge_contract_retains_incomplete_attempts_and_unavailable_fields() -> None:
    payload = json.loads((FIXTURES / "measurements.json").read_text())
    trial = payload["trials"][0]
    trial.update(
        status="incomplete",
        selected_attempt=None,
        parse={"status": "unavailable", "rating": None, "value": None},
    )
    trial["attempts"][0].update(
        status="timeout_ambiguous",
        resolved_model=None,
        response=None,
        request_id=None,
        finish_reason=None,
        error={"code": "timeout", "message": "Provider completion is unknown."},
        usage=None,
    )
    payload["completeness"].update(status="incomplete", completed_trials=1)
    Draft202012Validator(public_contracts.load_judge_measurements_schema()).validate(
        payload
    )


def test_judge_contracts_bound_all_collections_and_text() -> None:
    approved_optional = {
        ("plan", "properties", "prompt"): {"reference_mode", "runtime_format"},
        ("plan", "properties", "judge"): {"service_identity"},
        ("plan", "properties", "judge", "properties", "service_identity"): {
            "response_format"
        },
    }
    observed_optional = {}

    def check(value: object, path: tuple[str | int, ...]) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                properties = set(value["properties"])
                required = set(value["required"])
                assert required <= properties
                optional = properties - required
                assert optional == approved_optional.get(path, set())
                if optional:
                    observed_optional[path] = optional
            if value.get("type") == "array":
                assert isinstance(value["maxItems"], int)
            if value.get("type") == "string":
                assert isinstance(value["maxLength"], int)
            for name, child in value.items():
                check(child, (*path, name))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check(child, (*path, index))

    check(public_contracts.load_judge_measurement_plan_schema(), ("plan",))
    check(public_contracts.load_judge_measurements_schema(), ("measurements",))
    assert observed_optional == approved_optional


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        (("prompt", "reference_mode"), ("none", "per_case")),
        (("prompt", "runtime_format"), ("canonical-json-v1", "chatml-v1")),
        (
            ("judge", "service_identity", "service"),
            ("vllm", "ollama", "lm_studio", "openai_compatible"),
        ),
        (
            ("judge", "service_identity", "response_format"),
            ("json_object", "json_schema"),
        ),
    ],
)
def test_judge_plan_optional_controls_are_closed(path, allowed) -> None:
    validator = Draft202012Validator(
        public_contracts.load_judge_measurement_plan_schema()
    )
    payload = json.loads((FIXTURES / "plan.json").read_text())
    payload["judge"]["service_identity"] = {
        "service": "ollama",
        "endpoint_sha256": "a" * 64,
    }
    target = payload
    for part in path[:-1]:
        target = target[part]
    for value in allowed:
        target[path[-1]] = value
        validator.validate(payload)
    for invalid in ("unknown", "", True, None, 1, {}, []):
        target[path[-1]] = invalid
        with pytest.raises(ValidationError):
            validator.validate(payload)
    del target[path[-1]]
    if path[-1] == "service":
        with pytest.raises(ValidationError):
            validator.validate(payload)
    else:
        validator.validate(payload)
    target["unapproved"] = True
    with pytest.raises(ValidationError):
        validator.validate(payload)
