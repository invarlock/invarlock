"""Neutral contracts retain closed fields, JSON types, and exact schema bounds."""

import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from invarlock.evaluation_record_contracts import contracts
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError

SCHEMAS = {
    "run": "evaluation_run",
    "policy": "comparison_policy",
    "case_set": "evaluation_case_set",
    "comparison": "multi_metric_comparison",
}
PROVENANCE_PATHS = [
    ("run", ("score_provenance", "quality")),
    ("policy", ("metrics", 0, "accepted_provenance")),
]
CLOSED_OBJECTS = [
    *((name, ()) for name in SCHEMAS),
    ("run", ("source",)),
    ("run", ("records", 0)),
    *PROVENANCE_PATHS,
    ("policy", ("metrics", 0)),
    ("policy", ("slices", 0)),
    ("case_set", ("cases", 0)),
    ("comparison", ("bindings",)),
    ("comparison", ("metrics", 0)),
    ("comparison", ("metrics", 0, "interval")),
]
METADATA_PATHS = [
    ("run", ("records", 0, "metadata")),
    ("case_set", ("cases", 0, "metadata")),
    ("policy", ("slices", 0, "where")),
]


def test_retired_standalone_evidence_has_no_module_schema_or_report_reader(tmp_path):
    from importlib.util import find_spec

    from invarlock.captured_reporting import CapturedReportError, is_captured_manifest

    assert find_spec("invarlock.record_evidence") is None
    filename = "record_evidence.schema.json"
    assert not (Path(__file__).resolve().parents[2] / "contracts" / filename).exists()
    assert not files("invarlock").joinpath("_data", "contracts", filename).is_file()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format": "invarlock/captured-evaluation-v1"})
    )
    with pytest.raises(CapturedReportError, match="unsupported"):
        is_captured_manifest(tmp_path)


def test_captured_comparison_subject_keyword_and_wire_fields_have_no_aliases():
    from invarlock.engine import compare_runs
    from invarlock.evaluation_records.templates import example_project

    baseline, subject, policy = example_project("classification")
    result = compare_runs(baseline=baseline, subject=subject, policy=policy)
    assert set(result["bindings"]) == {"baseline", "subject", "policy"}
    assert all(
        "subject_mean" in metric and "candidate_mean" not in metric
        for metric in result["metrics"]
    )
    with pytest.raises(TypeError, match="candidate"):
        compare_runs(baseline=baseline, candidate=subject, policy=policy)
    for old, new in (
        ("candidate_minimum", "subject_minimum"),
        ("candidate_maximum", "subject_maximum"),
    ):
        changed = deepcopy(policy)
        metric = next(item for item in changed["metrics"] if new in item)
        metric[old] = metric.pop(new)
        with pytest.raises(EvaluationRecordsError, match=old):
            compare_runs(baseline, subject, changed)


def at(value, path):
    for key in path:
        value = value[key]
    return value


@pytest.fixture
def documents():
    digest = "sha256:" + "a" * 64
    provenance = {
        "kind": "judge",
        "source": "judge-model",
        "version": "1",
        "unit": "score",
        "rubric_digest": digest,
    }
    case = {"id": "one", "input": "q", "expected": "a", "metadata": {"group": "a"}}
    return {
        "run": {
            "format": "invarlock/evaluation-run-v1",
            "source": {"name": "evaluation", "version": "1"},
            "run_id": "run",
            "artifact_digest": digest,
            "source_digest": digest,
            "score_provenance": {"quality": deepcopy(provenance)},
            "records": [
                {
                    **deepcopy(case),
                    "output": "a",
                    "scores": {"quality": 0.5},
                    "error": None,
                    "context": None,
                }
            ],
        },
        "policy": {
            "format": "invarlock/comparison-policy-v1",
            "metrics": [
                {
                    "name": "quality",
                    "kind": "recorded",
                    "configuration": {},
                    "direction": "higher",
                    "unit": "score",
                    "aggregation": "mean",
                    "minimum_count": 1,
                    "maximum_regression": 0,
                    "maximum_interval_width": 1,
                    "subject_minimum": 0,
                    "subject_maximum": 1,
                    "score_key": "quality",
                    "accepted_provenance": provenance,
                }
            ],
            "slices": [{"name": "group-a", "where": {"group": "a"}}],
            "expected_case_set_digest": digest,
        },
        "case_set": {"format": "invarlock/evaluation-case-set-v1", "cases": [case]},
        "comparison": {
            "format": "invarlock/multi-metric-comparison-v1",
            "decision": "pass",
            "bindings": dict.fromkeys(("baseline", "subject", "policy"), digest),
            "metrics": [
                {
                    "name": "quality",
                    "slice": "overall",
                    "kind": "recorded",
                    "unit": "score",
                    "direction": "higher",
                    "aggregation": "mean",
                    "scoring_assurance": "recorded",
                    "count": 1,
                    "missing_ids": ["missing"],
                    "baseline_mean": 0.5,
                    "subject_mean": 0.5,
                    "delta": 0,
                    "interval": {
                        "lower": 0,
                        "upper": 0,
                        "method": "bootstrap",
                        "mass": 0.95,
                        "replicates": 1000,
                    },
                    "decision": "pass",
                    "reasons": ["within bounds"],
                }
            ],
            "limitations": ["recorded scores"],
        },
    }


@pytest.mark.parametrize("name", SCHEMAS)
def test_source_and_packaged_schemas_match_and_accept_valid_records(documents, name):
    filename = SCHEMAS[name] + ".schema.json"
    source = (Path(__file__).resolve().parents[2] / "contracts" / filename).read_bytes()
    packaged = files("invarlock").joinpath("_data", "contracts", filename).read_bytes()
    assert source == packaged
    schema = json.loads(source)
    Draft202012Validator.check_schema(schema)
    assert schema["$id"] == f"https://invarlock.dev/contracts/{filename}"
    contracts.validate(documents[name], name)
    documents[name]["format"] = "unknown"
    with pytest.raises(EvaluationRecordsError):
        contracts.validate(documents[name], name)


@pytest.mark.parametrize("name,path", CLOSED_OBJECTS)
def test_unknown_fields_are_rejected_at_every_closed_object(documents, name, path):
    at(documents[name], path)["unexpected"] = True
    with pytest.raises(EvaluationRecordsError):
        contracts.validate(documents[name], name)


@pytest.mark.parametrize("name,path", CLOSED_OBJECTS)
def test_required_fields_cannot_be_omitted(documents, name, path):
    optional = {
        "expected_case_set_digest",
        "subject_minimum",
        "subject_maximum",
        "score_key",
        "accepted_provenance",
    }
    for field in at(documents[name], path):
        changed = deepcopy(documents[name])
        del at(changed, path)[field]
        if field in optional:
            contracts.validate(changed, name)
        else:
            with pytest.raises(EvaluationRecordsError):
                contracts.validate(changed, name)


@pytest.mark.parametrize("score", [True, False, "0.5", None, [], {}])
def test_scores_reject_booleans_and_nonnumeric_values(documents, score):
    documents["run"]["records"][0]["scores"]["quality"] = score
    with pytest.raises(EvaluationRecordsError, match="scores.quality"):
        contracts.validate(documents["run"], "run")


@pytest.mark.parametrize("score", [-1, 0, 0.5, 2])
def test_numeric_scores_are_not_restricted_to_a_unit_interval(documents, score):
    documents["run"]["records"][0]["scores"]["quality"] = score
    contracts.validate(documents["run"], "run")


@pytest.mark.parametrize("name,path", METADATA_PATHS)
@pytest.mark.parametrize("value", [True, 1, None, [], {}])
def test_metadata_and_slice_values_must_be_strings(documents, name, path, value):
    at(documents[name], path)["group"] = value
    with pytest.raises(EvaluationRecordsError):
        contracts.validate(documents[name], name)


@pytest.mark.parametrize("name,path", PROVENANCE_PATHS)
@pytest.mark.parametrize(
    "field,value",
    [("kind", "recomputed"), ("source", 1), ("version", None), ("unit", True)],
)
def test_invalid_provenance_fields_are_rejected(documents, name, path, field, value):
    at(documents[name], path)[field] = value
    with pytest.raises(EvaluationRecordsError):
        contracts.validate(documents[name], name)


@pytest.mark.parametrize("name,path", PROVENANCE_PATHS)
@pytest.mark.parametrize("kind", ["judge", "human", "measurement", "external_metric"])
def test_all_original_provenance_kinds_and_null_rubrics_are_allowed(
    documents, name, path, kind
):
    at(documents[name], path).update(kind=kind, rubric_digest=None)
    contracts.validate(documents[name], name)


@pytest.mark.parametrize(
    "name,path",
    [
        ("run", ("artifact_digest",)),
        ("run", ("source_digest",)),
        *((name, (*path, "rubric_digest")) for name, path in PROVENANCE_PATHS),
        ("policy", ("expected_case_set_digest",)),
        *(
            ("comparison", ("bindings", key))
            for key in ("baseline", "subject", "policy")
        ),
    ],
)
def test_digest_spelling_and_exact_length(documents, name, path):
    for value in (
        True,
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
    ):
        at(documents[name], path[:-1])[path[-1]] = value
        with pytest.raises(EvaluationRecordsError):
            contracts.validate(documents[name], name)


@pytest.mark.parametrize(
    "name,field,minimum,maximum",
    [
        ("run", "records", 1, 50000),
        ("case_set", "cases", 1, 50000),
        ("policy", "metrics", 1, 16),
        ("policy", "slices", 0, 16),
        ("comparison", "metrics", 0, 272),
    ],
)
def test_exact_array_counts(documents, name, field, minimum, maximum):
    value = documents[name]
    item = value[field][0]
    assert contracts._validator(name).schema["properties"][field]["maxItems"] == maximum
    value[field] = [item] * maximum
    contracts.validate(value, name)
    value[field].append(item)
    with pytest.raises(EvaluationRecordsError, match="too long"):
        contracts.validate(value, name)
    value[field] = []
    if minimum:
        with pytest.raises(EvaluationRecordsError):
            contracts.validate(value, name)
    else:
        contracts.validate(value, name)


@pytest.mark.parametrize(
    "name,path",
    [
        *METADATA_PATHS,
        ("run", ("records", 0, "scores")),
        ("run", ("score_provenance",)),
        ("policy", ("metrics", 0, "configuration")),
    ],
)
def test_exact_mapping_sizes(documents, name, path):
    mapping = at(documents[name], path)
    item = next(iter(mapping.values()), None)
    mapping.clear()
    mapping.update((str(index), item) for index in range(100))
    contracts.validate(documents[name], name)
    mapping["extra"] = item
    with pytest.raises(EvaluationRecordsError):
        contracts.validate(documents[name], name)
    mapping.clear()
    if path[-1] == "where":
        with pytest.raises(EvaluationRecordsError):
            contracts.validate(documents[name], name)
    else:
        contracts.validate(documents[name], name)


@pytest.mark.parametrize(
    "name,path,maximum",
    [
        ("run", ("run_id",), 128),
        *(("run", ("source", key), 128) for key in ("name", "version")),
        ("run", ("records", 0, "id"), 128),
        ("case_set", ("cases", 0, "id"), 128),
        *(
            (name, (*path, key), 128)
            for name, path in PROVENANCE_PATHS
            for key in ("source", "version", "unit")
        ),
        *(
            ("policy", ("metrics", 0, key), 128)
            for key in ("name", "unit", "score_key")
        ),
        ("policy", ("slices", 0, "name"), 128),
        *(
            ("comparison", ("metrics", 0, key), 128)
            for key in ("name", "slice", "kind", "unit")
        ),
        ("comparison", ("metrics", 0, "missing_ids", 0), 128),
        ("comparison", ("metrics", 0, "interval", "method"), 128),
        *((name, (*path, "group"), 4096) for name, path in METADATA_PATHS),
        ("run", ("records", 0, "error"), 4096),
        ("comparison", ("metrics", 0, "reasons", 0), 4096),
        ("comparison", ("limitations", 0), 4096),
    ],
)
def test_exact_string_sizes_and_identifier_types(documents, name, path, maximum):
    parent = at(documents[name], path[:-1])
    for length in (1 if maximum == 128 else 0, maximum):
        parent[path[-1]] = "x" * length
        contracts.validate(documents[name], name)
    invalid = ["x" * (maximum + 1), 1, True]
    if maximum == 128:
        invalid.extend(["", "x\x00y", "x\x1fy", "x\x7fy"])
    for value in invalid:
        parent[path[-1]] = value
        with pytest.raises(EvaluationRecordsError):
            contracts.validate(documents[name], name)


@pytest.mark.parametrize(
    "name,path,allowed,rejected",
    [
        ("policy", ("metrics", 0, "minimum_count"), [1, 50000], [0, 50001, 1.5]),
        ("policy", ("metrics", 0, "maximum_regression"), [0, 0.5], [-0.1]),
        ("policy", ("metrics", 0, "maximum_interval_width"), [0.1, 1], [0, -1]),
        *(
            ("policy", ("metrics", 0, key), [-1, 0, 2], [])
            for key in ("subject_minimum", "subject_maximum")
        ),
        ("comparison", ("metrics", 0, "count"), [0, 50001], [-1, 1.5]),
        *(
            ("comparison", ("metrics", 0, key), [None, -1, 0.5], [])
            for key in ("baseline_mean", "subject_mean", "delta")
        ),
        *(
            ("comparison", ("metrics", 0, "interval", key), [-1, 0.5, 2], [None])
            for key in ("lower", "upper", "mass")
        ),
        (
            "comparison",
            ("metrics", 0, "interval", "replicates"),
            [-1, 0, 1000],
            [None, 1.5],
        ),
    ],
)
def test_numeric_types_and_original_bounds(documents, name, path, allowed, rejected):
    parent = at(documents[name], path[:-1])
    for value in allowed:
        parent[path[-1]] = value
        contracts.validate(documents[name], name)
    for value in [*rejected, True, False, "0.5", [], {}]:
        parent[path[-1]] = value
        with pytest.raises(EvaluationRecordsError):
            contracts.validate(documents[name], name)


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
def test_unicode_configuration_remains_closed_and_versioned(documents, kind):
    policy = documents["policy"]
    metric = policy["metrics"][0]
    metric["kind"] = kind
    for configuration in (
        {"unicode_version": "16.0.0"},
        {"unicode_version": "1" * 28 + ".0.0", "casefold": False},
    ):
        metric["configuration"] = configuration
        contracts.validate(policy, "policy")
    for configuration in (
        {},
        {"unicode_version": "16.0"},
        {"unicode_version": "1" * 29 + ".0.0"},
        {"unicode_version": "16.0.0", "casefold": "true"},
        {"unicode_version": "16.0.0", "unexpected": True},
    ):
        metric["configuration"] = configuration
        with pytest.raises(EvaluationRecordsError):
            contracts.validate(policy, "policy")


@pytest.mark.parametrize("value", [None, True, 1, "text", [], {"nested": [False, 2]}])
def test_record_payload_fields_remain_unrestricted_json(documents, value):
    for field in ("input", "expected", "output", "context"):
        documents["run"]["records"][0][field] = value
    for field in ("input", "expected"):
        documents["case_set"]["cases"][0][field] = value
    contracts.validate(documents["run"], "run")
    contracts.validate(documents["case_set"], "case_set")


def test_nullable_source_digest_and_interval_remain_allowed(documents):
    documents["run"]["source_digest"] = None
    documents["comparison"]["metrics"][0]["interval"] = None
    contracts.validate(documents["run"], "run")
    contracts.validate(documents["comparison"], "comparison")
