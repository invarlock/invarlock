"""Hosted identities bind observed service descriptions without claiming weights."""

import copy
import json

import pytest

from invarlock.engine import make_run, run_digest
from invarlock.evaluation_comparison.comparison import _check_run
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evaluation_records.adapters import _parse_run_bytes
from invarlock.evaluator_capture import (
    capture_evaluator_run,
    evaluator_input_capabilities,
)


def identity():
    return {
        "kind": "hosted_service",
        "provider": "example-provider",
        "service": "responses",
        "deployment": "production",
        "requested_model": "model-alias",
        "observed_model": None,
        "exposed_revision": None,
        "configuration": {"temperature": 0},
        "configuration_digest": digest({"temperature": 0}),
        "harness": {
            "name": "evaluator",
            "version": "1.0",
            "source_digest": digest("source"),
        },
        "observation_window": {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
        },
    }


def row():
    return {"id": "one", "input": "question", "expected": "yes", "output": "yes"}


def hosted(**changes):
    return capture_evaluator_run(
        [row()],
        source={"name": "evaluator", "version": "1.0"},
        run_id="run",
        artifact_digest=None,
        service_identity=identity(),
        **changes,
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(provider=""),
        lambda s: s.update(provider=" "),
        lambda s: s["harness"].update(source_digest=digest("source") + "\n"),
        lambda s: s.update(deployment=None),
        lambda s: s.pop("observed_model"),
        lambda s: s.update(weights_digest=digest("pretend")),
        lambda s: s["harness"].pop("source_digest"),
        lambda s: s["observation_window"].update(started_at="2026-01-01T00:02:00Z"),
        lambda s: s["observation_window"].update(started_at="2026-02-30T00:00:00Z"),
        lambda s: s["observation_window"].update(
            started_at="2026-01-01T00:00:00+00:00"
        ),
    ],
)
def test_rejects_missing_ambiguous_or_invalid_service_facts(change):
    run = hosted()
    change(run["service_identity"])
    with pytest.raises(EvaluationRecordsError):
        _check_run(run)


@pytest.mark.parametrize(
    "path",
    [
        ("provider",),
        ("service",),
        ("deployment",),
        ("requested_model",),
        ("observed_model",),
        ("exposed_revision",),
        ("configuration_digest",),
        ("harness", "name"),
        ("harness", "version"),
        ("harness", "source_digest"),
        ("observation_window", "started_at"),
        ("observation_window", "ended_at"),
    ],
)
@pytest.mark.parametrize("suffix", ["\n", "\r", "\x00", "\x7f"])
def test_identity_strings_reject_trailing_controls_in_every_contract(path, suffix):
    from jsonschema import Draft202012Validator

    from invarlock.engine import validate_service_identity
    from invarlock.evaluation_record_contracts.contracts import _validator
    from invarlock.public_contracts import (
        load_captured_evaluation_request_schema,
        load_normalized_captured_request_schema,
    )

    descriptor = identity()
    parent = descriptor
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = (parent[path[-1]] or "observed-value") + suffix
    schemas = [_validator("run").schema["properties"]["service_identity"]]
    schemas.extend(
        schema["$defs"]["source_spec"]["properties"]["service_identity"]
        for schema in (
            load_captured_evaluation_request_schema(),
            load_normalized_captured_request_schema(),
        )
    )
    for schema in schemas:
        assert not Draft202012Validator(schema).is_valid(descriptor)
    with pytest.raises(EvaluationRecordsError):
        validate_service_identity(descriptor)
    run = hosted()
    run["service_identity"] = descriptor
    with pytest.raises(EvaluationRecordsError):
        _check_run(run)


def test_local_identity_is_unchanged_and_hosted_identity_is_bound():
    local = make_run(
        [row()],
        source={"name": "evaluator", "version": "1.0"},
        run_id="run",
        artifact_digest=digest("weights"),
    )
    assert "service_identity" not in local
    assert run_digest(local) == digest(local)
    run = hosted()
    assert run["artifact_digest"] is None
    modified = copy.deepcopy(run)
    modified["service_identity"]["requested_model"] = "another-alias"
    assert run_digest(run) != run_digest(modified)
    for bad in (
        {**run, "artifact_digest": digest("weights")},
        {**local, "artifact_digest": None},
        {**local, "service_identity": None},
    ):
        with pytest.raises(EvaluationRecordsError):
            _check_run(bad)


def test_generic_import_forwards_hosted_identity_and_forbids_override():
    run = _parse_run_bytes(
        json.dumps(row()).encode(),
        adapter="jsonl",
        source={"name": "evaluator", "version": "1.0"},
        run_id="run",
        artifact_digest=None,
        service_identity=identity(),
    )
    assert run["service_identity"] == identity()
    assert evaluator_input_capabilities(run)["exact_match"]["usable_count"] == 1
    with pytest.raises(EvaluationRecordsError, match="overridden"):
        _parse_run_bytes(json.dumps(run).encode(), service_identity=identity())


def test_likelihood_binds_service_and_declared_configuration_without_weight_claim():
    from invarlock.engine import compare_runs, evaluated_subject_digest
    from tests.evaluation_comparison.test_likelihood import (
        policy,
    )
    from tests.evaluation_comparison.test_likelihood import (
        row as likelihood_row,
    )

    record = likelihood_row()
    descriptor = identity()
    record["likelihood"].update(
        artifact_digest=None,
        service_identity_digest=digest(descriptor),
        configuration_digest=descriptor["configuration_digest"],
    )
    run = capture_evaluator_run(
        [record],
        source=record["likelihood"]["source"],
        run_id="run",
        artifact_digest=None,
        service_identity=descriptor,
    )
    approved = policy()
    approved["metrics"][0]["configuration"]["configuration_digest"] = descriptor[
        "configuration_digest"
    ]
    comparison = compare_runs(run, run, approved)
    assert comparison["decision"] == "pass"
    assert any(
        "do not identify underlying weights" in limit
        for limit in comparison["limitations"]
    )
    assert evaluated_subject_digest(run) == digest(descriptor)
    assert (
        evaluator_input_capabilities(run)["normalized_nll_per_utf8_byte"][
            "usable_count"
        ]
        == 1
    )
    for change in (
        lambda r: r["likelihood"].pop("service_identity_digest"),
        lambda r: r["likelihood"].update(service_identity_digest=digest("other")),
        lambda r: r["likelihood"].update(artifact_digest=digest(descriptor)),
        lambda r: r["likelihood"].update(configuration_digest=digest("other")),
        lambda r: r["likelihood"].pop("tokenizer_digest"),
        lambda r: r["likelihood"].pop("token_count"),
    ):
        modified = copy.deepcopy(run)
        change(modified["records"][0])
        with pytest.raises(EvaluationRecordsError):
            _check_run(modified)
    local = copy.deepcopy(run)
    local.pop("service_identity")
    local["artifact_digest"] = digest("weights")
    local["records"][0]["likelihood"]["artifact_digest"] = local["artifact_digest"]
    with pytest.raises(EvaluationRecordsError):
        _check_run(local)


def test_likelihood_validates_and_hashes_service_once_per_run(monkeypatch):
    from invarlock.evaluation_records import identity as service_module
    from tests.evaluation_comparison.test_likelihood import SOURCE
    from tests.evaluation_comparison.test_likelihood import row as likelihood_row

    descriptor = identity()
    descriptor["configuration"] = {"instructions": "x" * 900_000}
    descriptor["configuration_digest"] = digest(descriptor["configuration"])
    record = likelihood_row()
    record["likelihood"].update(
        artifact_digest=None,
        service_identity_digest=digest(descriptor),
        configuration_digest=descriptor["configuration_digest"],
    )
    run = make_run(
        [record],
        source=SOURCE,
        run_id="run",
        artifact_digest=None,
        service_identity=descriptor,
    )
    original_validate = service_module.validate_service_identity
    original_digest = service_module.digest
    calls = {"validation": 0, "digest": 0}

    def validate(value):
        calls["validation"] += 1
        return original_validate(value)

    def hash_descriptor(value):
        calls["digest"] += 1
        return original_digest(value)

    monkeypatch.setattr(service_module, "validate_service_identity", validate)
    monkeypatch.setattr(service_module, "digest", hash_descriptor)
    for count in (1, 100):
        run["records"] = [
            {**copy.deepcopy(run["records"][0]), "id": str(index)}
            for index in range(count)
        ]
        calls.update(validation=0, digest=0)
        _check_run(run)
        assert calls == {"validation": 1, "digest": 2}
    run["records"][-1]["likelihood"]["service_identity_digest"] = digest("other")
    with pytest.raises(EvaluationRecordsError, match="service_identity_digest binding"):
        _check_run(run)


@pytest.mark.parametrize("field", ["configuration", "configuration_digest"])
def test_configuration_cannot_change_without_its_exact_binding(field):
    run = hosted()
    run["service_identity"][field] = (
        {"temperature": 1} if field == "configuration" else digest("other")
    )
    with pytest.raises(EvaluationRecordsError, match="configuration digest"):
        _check_run(run)


def test_descriptor_size_limit_and_standalone_helpers_fail_closed():
    from invarlock.engine import evaluated_subject_digest, validate_service_identity

    descriptor = identity()
    descriptor["configuration"] = {"instructions": "x" * (1024 * 1024)}
    descriptor["configuration_digest"] = digest(descriptor["configuration"])
    with pytest.raises(EvaluationRecordsError, match="byte limit"):
        validate_service_identity(descriptor)
    for run in (
        {},
        {"artifact_digest": "not-a-digest"},
        {"service_identity": identity()},
    ):
        with pytest.raises(EvaluationRecordsError):
            evaluated_subject_digest(run)


def test_multiline_declared_configuration_roundtrips_external_request():
    from invarlock.captured_normalization import normalize_captured_request
    from tests.evaluation_comparison.test_likelihood import policy

    descriptor = identity()
    descriptor["configuration"] = {
        "instructions": "first line\nsecond line",
        "tools": [{"description": "\tretained data"}],
    }
    descriptor["configuration_digest"] = digest(descriptor["configuration"])
    raw = json.dumps(row()).encode()
    source = {
        "adapter": "jsonl",
        "source": {"name": "evaluator", "version": "1"},
        "run_id": "run",
        "artifact_digest": None,
        "service_identity": descriptor,
    }
    run = _parse_run_bytes(raw, **source)
    authored = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            side: {"path": side + ".jsonl", **source}
            for side in ("baseline", "subject")
        },
        "output": {"evidence": "evidence"},
    }
    authored["comparison"]["policy"] = "policy.json"
    normalized = normalize_captured_request(
        authored, baseline=run, subject=run, policy=policy()
    )
    assert normalized["comparison"]["subject"]["service_identity"] == descriptor
    authored["comparison"]["subject"]["source"]["name"] = "evaluator\nforged"
    with pytest.raises(EvaluationRecordsError, match="control characters"):
        normalize_captured_request(authored, baseline=run, subject=run, policy=policy())


def test_public_subject_digest_helper_never_falls_back_from_invalid_hosted_identity():
    from invarlock.engine import evaluated_subject_digest, validate_service_identity

    artifact = digest("weights")
    assert evaluated_subject_digest({"artifact_digest": artifact}) == artifact
    for bad in (None, {**identity(), "unknown": True}):
        with pytest.raises(EvaluationRecordsError):
            validate_service_identity(bad)
    with pytest.raises(EvaluationRecordsError, match="cannot claim an artifact"):
        evaluated_subject_digest(
            {"artifact_digest": artifact, "service_identity": identity()}
        )
