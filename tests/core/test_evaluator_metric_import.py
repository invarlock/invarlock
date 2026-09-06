"""Broader import must bind both reference text and exact scoring semantics."""

import hashlib
import json

import pytest

from invarlock.evaluator_qualification import (
    EvaluatorQualificationError,
    qualify_evaluator_export,
)
from tests.core.test_evaluator_qualification import _write_json, qualification_fixture


def setup(paths, kind, configuration, reference, output, value):
    profile, schedule, export, _ = paths
    p, s, e = [json.loads(f.read_text()) for f in paths[:3]]
    p["authority"]["metric"] = {"kind": kind, "configuration": configuration}
    for expected, observed in zip(s["records"], e["records"], strict=True):
        expected["reference_output"] = reference
        expected["reference_output_sha256"] = (
            "sha256:" + hashlib.sha256(reference.encode()).hexdigest()
        )
        observed["output_text"] = output
        observed["output_sha256"] = (
            "sha256:" + hashlib.sha256(output.encode()).hexdigest()
        )
        observed["reported_score"] = value
    _write_json(profile, p)
    _write_json(schedule, s)
    e["bindings"]["profile_sha256"] = (
        "sha256:" + hashlib.sha256(profile.read_bytes()).hexdigest()
    )
    e["bindings"]["schedule_sha256"] = (
        "sha256:" + hashlib.sha256(schedule.read_bytes()).hexdigest()
    )
    _write_json(export, e)


def qualify(paths):
    return qualify_evaluator_export(
        profile_path=paths[0],
        schedule_path=paths[1],
        export_path=paths[2],
        raw_output_path=paths[3],
    )


@pytest.mark.parametrize(
    "kind,config,target,output,value",
    [
        ("normalized_match", {}, "yes", " YES ", 1.0),
        ("numeric_tolerance", {"absolute": 0.1}, "10", "10.05", 1.0),
        ("token_f1", {}, "a b", "a", 2 / 3),
        ("json_fields", {"fields": ["/a", "/b"]}, '{"a":1,"b":2}', '{"a":1}', 0.5),
    ],
)
def test_non_exact_metric_qualifies_and_exposes_the_bound_scorer(
    tmp_path, kind, config, target, output, value
):
    paths = qualification_fixture(tmp_path)
    setup(paths, kind, config, target, output, value)
    result = qualify(paths)
    assert result.scores == (value, value)
    assert len(result.runtime_records()) == 2
    assert result.scorer_binding().scorer_id == f"invarlock.{kind}"


@pytest.mark.parametrize(
    "change",
    ["wrong_score", "reference_digest", "missing_reference", "unknown_configuration"],
)
def test_broader_import_rejects_false_or_ambiguous_scoring(tmp_path, change):
    paths = qualification_fixture(tmp_path)
    setup(paths, "normalized_match", {}, "yes", "YES", 1.0)
    profile, schedule, export, _ = paths
    p, s, e = [json.loads(f.read_text()) for f in paths[:3]]
    if change == "wrong_score":
        e["records"][0]["reported_score"] = 0.5
    elif change == "reference_digest":
        s["records"][0]["reference_output"] = "different"
    elif change == "missing_reference":
        del s["records"][0]["reference_output"]
    else:
        p["authority"]["metric"]["configuration"] = {"ignore_anything": True}
    _write_json(profile, p)
    _write_json(schedule, s)
    e["bindings"]["profile_sha256"] = (
        "sha256:" + hashlib.sha256(profile.read_bytes()).hexdigest()
    )
    e["bindings"]["schedule_sha256"] = (
        "sha256:" + hashlib.sha256(schedule.read_bytes()).hexdigest()
    )
    _write_json(export, e)
    with pytest.raises(EvaluatorQualificationError):
        qualify(paths)
