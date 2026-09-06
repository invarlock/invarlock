"""Independent verification and native export failures on real production paths."""

import base64
import copy
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline import (
    PipelineError,
    compare_runs,
    create_evidence,
    load_run,
    verify_evidence,
)
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import digest
from invarlock.pipeline.evidence import DOMAIN
from invarlock.pipeline.metrics import UNICODE_VERSION, MetricError, score
from invarlock.pipeline.report import render_html, render_junit
from invarlock.pipeline.templates import example_project


@pytest.mark.parametrize(
    "kind,target,output,configuration,expected",
    [
        (
            "normalized_match",
            "Straße  YES",
            " STRASSE yes ",
            {"unicode_version": UNICODE_VERSION},
            1,
        ),
        (
            "normalized_match",
            "YES",
            "yes",
            {"casefold": False, "unicode_version": UNICODE_VERSION},
            0,
        ),
        ("exact_match", " yes", "yes", {}, 0),
        ("numeric_tolerance", 100, "100.5", {"relative": 0.01}, 1),
        ("numeric_tolerance", 0, "0.001", {"absolute": 0.0001}, 0),
        ("numeric_tolerance", 1, "nan", {}, 0),
        ("numeric_tolerance", 1, True, {}, 0),
        ("json_fields", {"a": 1, "b": 2}, {"a": 1}, {"fields": ["/a", "/b"]}, 0.5),
        ("json_fields", '{"a":true}', '{"a":1}', {"fields": ["/a"]}, 0),
        ("json_fields", '{"a":1}', '{"a":1,"a":2}', {"fields": ["/a"]}, 0),
        (
            "json_fields",
            {"a/b": {"~": 3}},
            {"a/b": {"~": 3}},
            {"fields": ["/a~1b/~0"]},
            1,
        ),
        ("token_f1", "a a b", "a b", {"unicode_version": UNICODE_VERSION}, 0.8),
        ("token_f1", "", "", {"unicode_version": UNICODE_VERSION}, 1),
    ],
)
def test_metric_meanings(kind, target, output, configuration, expected):
    assert score(kind, target, output, configuration) == expected


@pytest.mark.parametrize(
    "kind,target,config",
    [
        ("numeric_tolerance", "nan", {}),
        ("numeric_tolerance", 1, {"relative": -1}),
        ("json_fields", {}, {"fields": ["/missing"]}),
        ("json_fields", {}, {"fields": ["/a~2b"]}),
        (
            "normalized_match",
            "yes",
            {"casefold": "true", "unicode_version": UNICODE_VERSION},
        ),
        ("token_f1", [], {"unicode_version": UNICODE_VERSION}),
    ],
)
def test_invalid_metric_configuration_or_reference_fails(kind, target, config):
    with pytest.raises(MetricError):
        score(kind, target, "anything", config)


def test_signed_evidence_requires_independent_key_inputs_and_policy():
    base, candidate, policy = example_project("extraction")
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(base, candidate, policy, key)
    kwargs = {
        "public_key": key.public_key(),
        "expected_baseline": digest(base),
        "expected_candidate": digest(candidate),
        "policy": policy,
    }
    assert verify_evidence(evidence, **kwargs)["decision"] == "pass"
    with pytest.raises(PipelineError, match="unsigned"):
        verify_evidence(create_evidence(base, candidate, policy), **kwargs)
    with pytest.raises(PipelineError, match="signature"):
        verify_evidence(
            evidence,
            **{**kwargs, "public_key": Ed25519PrivateKey.generate().public_key()},
        )
    with pytest.raises(PipelineError, match="digest"):
        verify_evidence(
            evidence, **{**kwargs, "expected_candidate": "sha256:" + "a" * 64}
        )
    changed_policy = copy.deepcopy(policy)
    changed_policy["metrics"][0]["maximum_regression"] = 0.9
    with pytest.raises(PipelineError, match="policy"):
        verify_evidence(evidence, **{**kwargs, "policy": changed_policy})
    changed = copy.deepcopy(evidence)
    changed["comparison"]["metrics"][0]["candidate_mean"] = 0.5
    # Even an authorized publisher cannot pass off a different computed report.
    payload = {k: v for k, v in changed.items() if k != "signature"}
    changed["signature"]["value"] = base64.b64encode(
        key.sign(DOMAIN + canonical_json_bytes(payload))
    ).decode()
    with pytest.raises(PipelineError, match="replay"):
        verify_evidence(changed, **kwargs)


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize(
    "change",
    [
        lambda run: run.update(artifact_digest="sha256:" + "a" * 64),
        lambda run: run.update(source_digest="sha256:" + "b" * 64),
        lambda run: run["source"].update(version="another-version"),
        lambda run: run["records"][0]["context"].update(
            generation={"temperature": 0.7}, tools=["search"]
        ),
    ],
    ids=["artifact", "source-bytes", "evaluator-version", "execution-context"],
)
def test_resigned_context_changes_cannot_reuse_independent_run_anchors(side, change):
    baseline, candidate, policy = example_project("extraction")
    key = Ed25519PrivateKey.generate()
    original = create_evidence(baseline, candidate, policy, key)
    anchors = {
        "expected_baseline": digest(baseline),
        "expected_candidate": digest(candidate),
    }
    assert (
        verify_evidence(
            original, public_key=key.public_key(), policy=policy, **anchors
        )["decision"]
        == "pass"
    )
    changed_runs = {
        "baseline": copy.deepcopy(baseline),
        "candidate": copy.deepcopy(candidate),
    }
    change(changed_runs[side])
    changed = create_evidence(
        changed_runs["baseline"], changed_runs["candidate"], policy, key
    )
    # The display name, outputs, scores, and valid signer are unchanged. A
    # context change still needs the recipient's approval of the new run bytes.
    with pytest.raises(PipelineError, match="independently expected run"):
        verify_evidence(changed, public_key=key.public_key(), policy=policy, **anchors)
    new_anchors = {
        "expected_baseline": digest(changed_runs["baseline"]),
        "expected_candidate": digest(changed_runs["candidate"]),
    }
    assert (
        verify_evidence(
            changed, public_key=key.public_key(), policy=policy, **new_anchors
        )["decision"]
        == "pass"
    )


def test_order_is_paired_by_id_and_overall_pass_cannot_hide_slice_failure():
    base, candidate, policy = example_project("extraction")
    candidate["records"].reverse()
    assert compare_runs(base, candidate, policy)["decision"] == "pass"
    for row in candidate["records"]:
        if row["metadata"]["category"] == "exception":
            row["output"] = {"amount": -1, "currency": "USD"}
    policy["metrics"][0].update(maximum_regression=1, candidate_minimum=0.6)
    result = compare_runs(base, candidate, policy)
    assert result["metrics"][0]["decision"] == "pass"
    assert result["metrics"][2]["decision"] == "regression"
    assert result["decision"] == "regression"


@pytest.mark.parametrize(
    "adapter", ["jsonl", "lm-eval-samples", "inspect-json", "promptfoo-jsonl"]
)
def test_native_exports_and_sdk_use_the_same_comparison(tmp_path, adapter):
    rows = [
        {"id": str(i), "input": f"q{i}", "expected": "yes", "output": " YES "}
        for i in range(40)
    ]
    if adapter == "lm-eval-samples":
        native = [
            {
                "doc_id": int(r["id"]),
                "doc": {"question": r["input"]},
                "arguments": [[r["input"], {}]],
                "target": r["expected"],
                "filtered_resps": [r["output"]],
                "exact_match": 0,
            }
            for r in rows
        ]
    elif adapter == "inspect-json":
        native = {
            "version": 2,
            "status": "success",
            "samples": [
                {
                    "id": r["id"],
                    "input": r["input"],
                    "target": "yes",
                    "scores": {"match": {"value": "C"}},
                    "output": {"choices": [{"message": {"content": r["output"]}}]},
                }
                for r in rows
            ],
        }
    elif adapter == "promptfoo-jsonl":
        native = [
            {
                "testIdx": int(r["id"]),
                "promptIdx": 0,
                "prompt": {"raw": r["input"]},
                "testCase": {"vars": {}, "metadata": {"invarlock_expected": "yes"}},
                "response": {"output": r["output"]},
                "score": 1,
            }
            for r in rows
        ]
    else:
        native = rows
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(native)
        if isinstance(native, dict)
        else "\n".join(json.dumps(r) for r in native)
    )
    run = load_run(
        path,
        adapter=adapter,
        source={"name": adapter, "version": "fixture-1"},
        run_id="run",
        artifact_digest="sha256:" + "a" * 64,
    )
    _, _, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    assert compare_runs(run, run, policy)["decision"] == "pass"
    assert run["source_digest"]


def test_export_ambiguity_is_rejected(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text('{"id":"x","id":"y"}\n')
    with pytest.raises(PipelineError, match="duplicate"):
        load_run(path, adapter="jsonl")
    path.write_text('{"version":999,"samples":[]}')
    with pytest.raises(PipelineError, match="version"):
        load_run(path, adapter="inspect-json")
    path.write_text(
        '{"doc_id":0,"doc":{},"target":"x","arguments":[],"filtered_resps":["x","y"]}\n'
    )
    with pytest.raises(PipelineError, match="ambiguous"):
        load_run(path, adapter="lm-eval-samples")


@pytest.mark.parametrize("example", ["classification", "extraction", "judge"])
def test_installed_command_journey_and_repeated_runs(tmp_path, example):
    runner = CliRunner()
    project = tmp_path / example
    initialized = runner.invoke(app, ["init", str(project), "--example", example])
    assert initialized.exit_code == 0, initialized.output
    args = [
        "compare",
        str(project / "pipeline.json"),
        "--output",
        str(project / "run-1"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["authentication"] == "unsigned_local"
    assert (project / "run-1" / "junit.xml").is_file()
    assert runner.invoke(app, args).exit_code == 2  # never overwrite prior evidence
    args[-1] = str(project / "run-2")
    assert runner.invoke(app, args).exit_code == 0
    candidate = json.loads((project / "candidate.json").read_text())
    candidate["records"][0]["error"] = "timeout"
    (project / "candidate.json").write_text(json.dumps(candidate))
    args[-1] = str(project / "missing")
    assert runner.invoke(app, args).exit_code == 3


def test_rendering_does_not_execute_or_hide_untrusted_metric_names():
    base, candidate, policy = example_project("extraction")
    policy["metrics"][0]["name"] = '<script>alert("unsafe")</script>'
    comparison = compare_runs(base, candidate, policy)
    html = render_html(comparison)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert b'failures="0"' in render_junit(comparison)
