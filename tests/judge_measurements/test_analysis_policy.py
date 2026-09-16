from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator, ValidationError

from invarlock import public_contracts
from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.analysis import (
    ANALYSIS_POLICY_MAX_BYTES,
    analyze_measurements,
    decode_analysis_policy,
    load_analysis_policy,
    validate_analysis_policy,
)
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    measurement_plan_digest,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"
REPO_ROOT = Path(__file__).parents[2]


def _plan() -> JudgeMeasurementPlan:
    return cast(JudgeMeasurementPlan, json.loads((FIXTURES / "plan.json").read_text()))


def _wire() -> JudgeAnalysisPolicyDocument:
    return cast(
        JudgeAnalysisPolicyDocument,
        json.loads((FIXTURES / "analysis_policy.json").read_text()),
    )


def test_policy_golden_is_closed_packaged_and_decodes_every_explicit_field():
    schema = public_contracts.load_judge_analysis_policy_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_wire())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == set(_wire())
    assert not any("default" in entry for entry in schema["properties"].values())
    assert public_contracts.JUDGE_ANALYSIS_POLICY_FORMAT_VERSION == _wire()["format"]
    assert "load_judge_analysis_policy_schema" in public_contracts.__all__
    assert (REPO_ROOT / "contracts/judge_analysis_policy.schema.json").read_bytes() == (
        REPO_ROOT / "src/invarlock/_data/contracts/judge_analysis_policy.schema.json"
    ).read_bytes()
    policy = decode_analysis_policy(_wire(), plan=_plan())
    assert policy.plan_sha256 == measurement_plan_digest(_plan())
    assert policy.metric_name == "factual-correctness"
    assert policy.required is True
    assert policy.direction == "higher"
    assert policy.alpha == Decimal("0.05")
    assert policy.allowed_degradation == Decimal("0.05")
    assert policy.minimum_units == 20
    assert policy.comparison_family_size == 2
    assert policy.maximum_interval_width == Decimal("0.3")
    assert policy.subject_bound is None
    with pytest.raises(FrozenInstanceError):
        policy.metric_name = "mutated"


def test_policy_loader_uses_package_schema_and_returns_fresh_objects(
    tmp_path, monkeypatch
):
    original = public_contracts.load_judge_analysis_policy_schema()
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts/judge_analysis_policy.schema.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    loaded = public_contracts.load_judge_analysis_policy_schema()
    assert loaded == original and loaded is not original
    loaded["properties"].clear()
    assert public_contracts.load_judge_analysis_policy_schema() == original
    assert load_analysis_policy(
        FIXTURES / "analysis_policy.json", plan=_plan()
    ) == decode_analysis_policy(_wire(), plan=_plan())


def test_each_policy_field_is_required_and_unknown_fields_fail():
    schema = Draft202012Validator(public_contracts.load_judge_analysis_policy_schema())
    for field in _wire():
        value = _wire()
        del value[field]
        with pytest.raises(ValidationError):
            schema.validate(value)
        with pytest.raises(JudgeMeasurementContractError):
            decode_analysis_policy(value, plan=_plan())
    value = _wire()
    value["unscheduled_slice"] = "hidden"
    with pytest.raises(ValidationError):
        schema.validate(value)
    with pytest.raises(JudgeMeasurementContractError):
        decode_analysis_policy(value, plan=_plan())


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "invarlock/comparison-policy-v2"),
        ("method", "empirical-bernstein-v1"),
        ("method", None),
        ("metric_name", ""),
        ("metric_name", "line\nbreak"),
        ("metric_name", "trailing\n"),
        ("metric_name", "x" * 129),
        ("plan_sha256", "sha256:" + "a" * 64),
        ("plan_sha256", "A" * 64),
        ("decision_role", "optional"),
        ("direction", "larger"),
        ("alpha", "0"),
        ("alpha", "1"),
        ("alpha", "-0.05"),
        ("alpha", "1e-2"),
        ("alpha", "0.050"),
        ("alpha", "00.05"),
        ("alpha", ".05"),
        ("alpha", "0.05\n"),
        ("alpha", "NaN"),
        ("alpha", "Infinity"),
        ("alpha", "0.0000000000000001"),
        ("alpha", 0.05),
        ("alpha", True),
        ("allowed_degradation", "1.1"),
        ("allowed_degradation", "-0"),
        ("allowed_degradation", "0.0"),
        ("allowed_degradation", "0.1\n"),
        ("allowed_degradation", 0),
        ("maximum_interval_width", "0"),
        ("maximum_interval_width", "2.1"),
        ("maximum_interval_width", "1.0"),
        ("maximum_interval_width", "1.2\n"),
        ("maximum_interval_width", 2),
        ("subject_bound", "1.000000000000001"),
        ("subject_bound", "-0.1"),
        ("subject_bound", " 0.5"),
        ("subject_bound", 0.5),
        ("comparison_family_size", 1),
        ("comparison_family_size", True),
        ("minimum_units", 0),
        ("minimum_units", "2"),
    ],
)
def test_schema_and_decoder_reject_noncanonical_or_out_of_range_values(field, value):
    policy = _wire()
    policy[field] = value
    with pytest.raises(ValidationError):
        Draft202012Validator(
            public_contracts.load_judge_analysis_policy_schema()
        ).validate(policy)
    with pytest.raises(JudgeMeasurementContractError):
        decode_analysis_policy(policy, plan=_plan())


@pytest.mark.parametrize("field", ["minimum_units", "comparison_family_size"])
@pytest.mark.parametrize("value", [2.0, 2.5, True, False])
def test_decoder_requires_strict_integer_fields(field, value):
    policy = _wire()
    policy[field] = value
    with pytest.raises(JudgeMeasurementContractError):
        decode_analysis_policy(policy, plan=_plan())


@pytest.mark.parametrize("role", ["required", "advisory"])
@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_roles_directions_and_inclusive_decimal_boundaries(role, direction):
    value = _wire()
    value.update(
        decision_role=role,
        direction=direction,
        alpha="0.000000000000001",
        maximum_interval_width="2",
        allowed_degradation="1",
        subject_bound="0",
    )
    policy = decode_analysis_policy(value, plan=_plan())
    assert policy.required is (role == "required")
    assert policy.direction == direction
    assert policy.alpha == Decimal("1e-15")
    assert policy.maximum_interval_width == 2
    assert policy.allowed_degradation == 1
    assert policy.subject_bound == 0
    value.update(
        allowed_degradation="0",
        subject_bound="1",
        maximum_interval_width="0.000000000000001",
    )
    policy = decode_analysis_policy(value, plan=_plan())
    assert policy.allowed_degradation == 0 and policy.subject_bound == 1


def test_policy_binding_is_checked_during_decode_and_analysis():
    value = _wire()
    value["plan_sha256"] = "0" * 64
    with pytest.raises(JudgeMeasurementContractError, match="bind"):
        validate_analysis_policy(value, plan=_plan())
    valid = decode_analysis_policy(_wire(), plan=_plan())
    substituted = replace(valid, plan_sha256="0" * 64)
    data = cast(
        JudgeMeasurements, json.loads((FIXTURES / "measurements.json").read_text())
    )
    baseline = json.loads((FIXTURES / "baseline_run.json").read_text())
    subject = json.loads((FIXTURES / "subject_run.json").read_text())
    with pytest.raises(JudgeMeasurementContractError, match="policy does not bind"):
        analyze_measurements(
            _plan(), data, substituted, baseline_run=baseline, subject_run=subject
        )


@pytest.mark.parametrize(
    "text", ['{"format":1,"format":2}', "[]", "{", '{"minimum_units":NaN}']
)
def test_file_loader_refuses_duplicate_keys_malformed_json_and_nonobjects(
    tmp_path, text
):
    path = tmp_path / "policy.json"
    path.write_text(text)
    with pytest.raises(JudgeMeasurementContractError):
        load_analysis_policy(path, plan=_plan())


def test_file_loader_and_in_memory_validation_enforce_byte_limits(tmp_path):
    path = tmp_path / "policy.json"
    path.write_bytes(b" " * (ANALYSIS_POLICY_MAX_BYTES + 1))
    with pytest.raises(JudgeMeasurementContractError):
        load_analysis_policy(path, plan=_plan())
    value = _wire()
    value["metric_name"] = "x" * ANALYSIS_POLICY_MAX_BYTES
    with pytest.raises(JudgeMeasurementContractError, match="byte limit"):
        validate_analysis_policy(value, plan=_plan())
