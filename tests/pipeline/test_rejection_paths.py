"""Untrusted exports and inconsistent policies must never yield a passing gate."""

import copy
import json
import os
from decimal import localcontext

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from typer.testing import CliRunner

from invarlock.core.builtin_scorers import BuiltinScorer
from invarlock.pipeline import (
    PipelineError,
    compare_runs,
    create_evidence,
    load_run,
    make_run,
    verify_evidence,
)
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import digest, read_json, write_directory, write_new
from invarlock.pipeline.metrics import UNICODE_VERSION, MetricError, score
from invarlock.pipeline.templates import example_project


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_metric",
        "duplicate_slice",
        "reserved_slice",
        "no_key",
        "configuration",
        "metric_unit",
        "no_rubric",
        "recomputed_provenance",
        "direction",
        "scorer_config",
        "absolute_bounds",
    ],
)
def test_policy_contradictions_are_integration_errors(change):
    base, candidate, policy = example_project("judge")
    metric = policy["metrics"][0]
    if change == "duplicate_metric":
        policy["metrics"].append(copy.deepcopy(metric))
    elif change == "duplicate_slice":
        policy["slices"].append(copy.deepcopy(policy["slices"][0]))
    elif change == "reserved_slice":
        policy["slices"][0]["name"] = "overall"
    elif change == "no_key":
        del metric["score_key"]
    elif change == "configuration":
        metric["configuration"] = {"casefold": True}
    elif change == "metric_unit":
        metric["unit"] = "percent"
    elif change == "no_rubric":
        metric["accepted_provenance"]["rubric_digest"] = None
    elif change == "recomputed_provenance":
        metric["kind"] = "normalized_match"
        metric["configuration"] = {"unicode_version": UNICODE_VERSION}
    elif change in ("direction", "scorer_config"):
        metric.update(kind="normalized_match")
        metric["configuration"] = {"unicode_version": UNICODE_VERSION}
        del metric["accepted_provenance"], metric["score_key"]
        if change == "direction":
            metric["direction"] = "lower"
        else:
            metric["configuration"] = {"absolute": 0.1}
    else:
        metric.update(candidate_minimum=1, candidate_maximum=0)
    with pytest.raises(PipelineError):
        compare_runs(base, candidate, policy)


@pytest.mark.parametrize(
    "kind,expected,output,configuration",
    [
        ("unknown", "x", "x", {}),
        ("json_fields", {}, {}, {}),
        ("json_fields", {}, {}, {"fields": ["/a", "/a"]}),
        ("numeric_tolerance", 1, 1, {"absolute": True}),
        ("normalized_match", "x", "x", {"absolute": 0}),
    ],
)
def test_scorer_configurations_do_not_guess(kind, expected, output, configuration):
    with pytest.raises(MetricError):
        score(kind, expected, output, configuration)


def test_structured_array_pointers_and_bad_answers():
    assert (
        score("json_fields", {"a": [1, 2]}, {"a": [1, 3]}, {"fields": ["/a/0", "/a/1"]})
        == 0.5
    )
    assert score("json_fields", {"a": [1]}, {"a": 3}, {"fields": ["/a/0"]}) == 0
    assert (
        score("normalized_match", "answer", {}, {"unicode_version": UNICODE_VERSION})
        == 0
    )
    with pytest.raises(ValueError):
        BuiltinScorer("unknown")


def test_numeric_tolerance_does_not_round_distinct_large_integers_together():
    reference = 9007199254740992
    assert score("numeric_tolerance", reference, reference + 1, {}) == 0
    assert (
        score("numeric_tolerance", reference, str(reference + 1), {"absolute": 1}) == 1
    )
    assert (
        score(
            "numeric_tolerance",
            "0.123456789012345678901",
            "0.123456789012345678902",
            {},
        )
        == 0
    )
    assert score("numeric_tolerance", 1, "1e999999999", {}) == 0
    assert score("numeric_tolerance", 1, "not a number", {}) == 0
    with localcontext() as context:
        context.prec = 2
        context.Emax = 5
        assert (
            score("numeric_tolerance", reference, str(reference + 1), {"absolute": 1})
            == 1
        )


@pytest.mark.parametrize(
    "native,adapter",
    [
        ([], "jsonl"),
        ({"version": True}, "inspect-json"),
        ({"version": 2, "status": "unknown"}, "inspect-json"),
        ({"version": 2, "status": "cancelled"}, "inspect-json"),
        ({"version": 2, "status": "success", "samples": []}, "inspect-json"),
        ([{"doc_id": 0}], "lm-eval-samples"),
        ([{"testIdx": 0}], "promptfoo-jsonl"),
    ],
)
def test_unsupported_native_shapes_fail(tmp_path, native, adapter):
    path = tmp_path / "export"
    path.write_text(
        json.dumps(native)
        if isinstance(native, dict)
        else "\n".join(json.dumps(r) for r in native)
    )
    with pytest.raises(PipelineError):
        load_run(path, adapter=adapter)


@pytest.mark.parametrize(
    "change", ["choices", "nontext", "targets", "epoch", "id", "score", "metadata"]
)
def test_ambiguous_inspect_records_fail(tmp_path, change):
    row = {
        "id": 1,
        "input": "q",
        "target": "yes",
        "output": {"choices": [{"message": {"content": "yes"}}]},
    }
    if change == "choices":
        row["output"]["choices"] = []
    elif change == "nontext":
        row["output"]["choices"][0]["message"]["content"] = [{"type": "image"}]
    elif change == "targets":
        row["target"] = ["yes", "no"]
    elif change == "epoch":
        row["epoch"] = 2
    elif change == "id":
        row["id"] = None
    elif change == "score":
        row["scores"] = {"judgment": {"value": "maybe"}}
    else:
        row["metadata"] = "not tags"
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"version": 2, "status": "success", "samples": [row]}))
    with pytest.raises(PipelineError):
        load_run(path, adapter="inspect-json")


def test_inspect_text_parts_single_target_and_error_preservation(tmp_path):
    row = {
        "id": 1,
        "input": "q",
        "target": ["yes"],
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "y"},
                            {"type": "text", "text": "es"},
                        ]
                    }
                }
            ]
        },
    }
    path = tmp_path / "export.json"
    options = {
        "adapter": "inspect-json",
        "source": {"name": "inspect", "version": "0.3.254"},
        "run_id": "one",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    path.write_text(json.dumps({"version": 2, "status": "success", "samples": [row]}))
    run = load_run(path, **options)
    assert run["records"][0]["output"] == "yes"
    row["error"] = {"message": "failed"}
    row["output"]["choices"] = []
    path.write_text(json.dumps({"version": 2, "status": "success", "samples": [row]}))
    assert load_run(path, **options)["records"][0]["error"] == "upstream_error"


def test_invalid_file_and_identity_operations(tmp_path):
    base, candidate, policy = example_project("classification")
    with pytest.raises(PipelineError):
        make_run(
            [{"unknown": 1}],
            source=base["source"],
            run_id="r",
            artifact_digest=base["artifact_digest"],
        )
    with pytest.raises(PipelineError):
        load_run(tmp_path / "absent", adapter="guess")
    with pytest.raises(PipelineError):
        read_json(tmp_path / "absent")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(base))
    with pytest.raises(PipelineError):
        load_run(path, run_id="override")
    with pytest.raises(PipelineError):
        write_new(path, b"changed")
    assert json.loads(path.read_text()) == base
    with pytest.raises(PipelineError):
        write_directory(tmp_path / "result", {"../escape": b"bad"})
    with pytest.raises(PipelineError):
        create_evidence(base, candidate, policy, object())
    with pytest.raises(PipelineError):
        verify_evidence(
            {},
            public_key=object(),
            expected_baseline=digest(base),
            expected_candidate=digest(candidate),
            policy=policy,
        )


def test_cli_import_digest_and_invalid_keys(tmp_path):
    runner = CliRunner()
    export = tmp_path / "records.jsonl"
    export.write_text('{"id":"1","input":"q","expected":"yes","output":"yes"}\n')
    target = tmp_path / "run.json"
    arguments = [
        "import",
        str(export),
        "--adapter",
        "jsonl",
        "--source-version",
        "1",
        "--run-id",
        "one",
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--output",
        str(target),
    ]
    assert runner.invoke(app, arguments).exit_code == 0
    assert runner.invoke(app, arguments).exit_code == 2
    assert runner.invoke(
        app, ["digest", str(target), "--run"]
    ).output.strip() == digest(json.loads(target.read_text()))
    assert runner.invoke(app, ["digest", str(target)]).output.startswith("sha256:")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert runner.invoke(app, ["digest", str(link)]).exit_code == 2
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    assert runner.invoke(app, ["digest", str(pipe)]).exit_code == 2
    assert (
        runner.invoke(
            app, ["init", str(tmp_path / "bad"), "--example", "unknown"]
        ).exit_code
        == 2
    )
    project = tmp_path / "project"
    assert runner.invoke(app, ["init", str(project)]).exit_code == 0
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = tmp_path / "rsa.pem"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public = tmp_path / "rsa.pub"
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    assert (
        runner.invoke(
            app,
            [
                "compare",
                str(project / "pipeline.json"),
                "--output",
                str(tmp_path / "result"),
                "--signing-key",
                str(private),
            ],
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            app,
            [
                "verify",
                str(target),
                "--public-key",
                str(public),
                "--policy",
                str(project / "policy.json"),
                "--expected-baseline",
                "sha256:" + "a" * 64,
                "--expected-candidate",
                "sha256:" + "b" * 64,
            ],
        ).exit_code
        == 2
    )
