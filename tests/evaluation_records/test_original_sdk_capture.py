"""Pinned native logger/result serialization through the common exporter.

These tests serialize authored frozen strings using actual SDK classes; they do
not execute a model, provider, benchmark, or new qualification campaign. Set
INVARLOCK_REQUIRE_EVALUATOR_SDK to the selected evaluator (or 1) to require it.
Promptfoo uses a local installation, optionally INVARLOCK_PROMPTFOO_PACKAGE.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from invarlock.engine import (
    EvaluationRecordsError,
    digest,
    evaluator_input_capabilities,
    export_evaluator_result,
    load_run,
)


def _required(evaluator):
    return os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in ("1", evaluator)


def _unavailable(evaluator, message):
    if _required(evaluator):
        pytest.fail(message)
    pytest.skip(message)


def _no_network(*args, **kwargs):
    raise AssertionError("native serialization smoke must not access the network")


def _harness_rows(tmp_path, monkeypatch, *, likelihood=False):
    evaluator = "lm-evaluation-harness"
    try:
        installed = importlib.metadata.version("lm-eval")
    except importlib.metadata.PackageNotFoundError:
        _unavailable(evaluator, "requires installed lm-eval==0.4.12")
    if installed != "0.4.12":
        _unavailable(evaluator, f"requires lm-eval==0.4.12, found {installed}")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    from lm_eval.api.instance import Instance
    from lm_eval.loggers.evaluation_tracker import EvaluationTracker

    # These are the same public Instance fields consumed by Harness's sample
    # logger. Frozen responses are supplied explicitly; no LM is instantiated.
    samples = []
    for index, answer in enumerate(("Answer", "Different")):
        observed = (-0.5 - index, True) if likelihood else answer
        request = Instance(
            request_type="loglikelihood" if likelihood else "generate_until",
            doc={"question": f"Question {index}?"},
            arguments=(
                f"Question {index}?",
                "Answer" if likelihood else {"until": ["\n"]},
            ),
            idx=0,
            metadata=("capture-smoke", index, 1),
            resps=[observed],
            filtered_resps={"none": observed},
        )
        samples.append(
            {
                "doc_id": request.doc_id,
                "doc": request.doc,
                "target": "Answer",
                "arguments": [request.args],
                "resps": [request.resps],
                "filtered_resps": [request.filtered_resps["none"]],
                "filter": "none",
                "metrics": [],
                "metadata": {"category": "authored"},
            }
        )
        if likelihood:
            # Supplied synthetic reference-continuation values test the SDK
            # serialization boundary only; they are not a fresh LM measurement.
            samples[-1]["metadata"]["invarlock_likelihood"] = {
                "basis": "reference_continuation",
                "logprob_sum": observed[0],
                "token_count": 1,
                "utf8_byte_count": len(b"Answer"),
                "input_digest": digest(request.doc),
                "reference_digest": digest("Answer"),
                "artifact_digest": "sha256:" + "a" * 64,
                "configuration_digest": digest("authored-configuration"),
                "tokenizer_digest": digest("authored-tokenizer"),
                "source": {"name": "lm-evaluation-harness", "version": "0.4.12"},
            }
    tracker = EvaluationTracker(output_path=str(tmp_path / "harness"))
    tracker.general_config_tracker.log_experiment_args(
        model_source="frozen-fixture",
        model_args={"model": "authored-strings"},
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    tracker.save_results_aggregated({"results": {}})
    tracker.save_results_samples("capture-smoke", samples)
    files = list((tmp_path / "harness").rglob("samples_*.jsonl"))
    assert len(files) == 1, "actual Harness logger must produce a sample file"
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    # The SDK logger changes request tuples to its documented on-disk arguments.
    assert rows[0]["arguments"]["gen_args_0"]["arg_0"] == "Question 0?"
    return (
        rows,
        "0.4.12",
        ["0", "1"],
        {"kind": "json-pointer", "pointer": "/input/question"},
    )


def _promptfoo_rows(tmp_path):
    evaluator = "promptfoo"
    node = shutil.which("node")
    if node is None:
        _unavailable(evaluator, "Promptfoo capture smoke requires local Node.js")
    supplied = os.environ.get("INVARLOCK_PROMPTFOO_PACKAGE")
    if supplied:
        package = Path(supplied)
    else:
        resolved = subprocess.run(
            [node, "-p", "require.resolve('promptfoo')"],
            capture_output=True,
            text=True,
            check=False,
        )
        if resolved.returncode:
            _unavailable(
                evaluator,
                "install promptfoo==0.121.19 locally or set INVARLOCK_PROMPTFOO_PACKAGE",
            )
        package = Path(resolved.stdout.strip()).parents[2]
    metadata = json.loads((package / "package.json").read_text())
    if metadata.get("version") != "0.121.19":
        _unavailable(evaluator, "requires pinned promptfoo==0.121.19")
    # The package's actual result model implements the same toEvaluateResult
    # and JSONL sanitizer used by upstream output. It has no persistence here.
    modules = list((package / "dist/src").glob("evalResult-*.js"))
    assert len(modules) == 1, "pinned Promptfoo result-model module changed"
    script = tmp_path / "promptfoo-capture.mjs"
    script.write_text("""
import fs from 'node:fs';
import net from 'node:net';
import http from 'node:http';
import https from 'node:https';
const blocked = () => { throw new Error('network forbidden during SDK serialization'); };
globalThis.fetch = blocked;
net.Socket.prototype.connect = blocked;
http.request = http.get = https.request = https.get = blocked;
const native = await import(process.argv[2]);
const EvalResult = Object.values(native).find(value => typeof value === 'function' && value.name === 'EvalResult');
const sanitize = Object.values(native).find(value => typeof value === 'function' && value.name === 'sanitizeResultForJsonlArtifact');
if (!EvalResult || !sanitize) throw new Error('pinned native serializers missing');
const rows = ['Answer', 'Different'].map((answer, index) => {
  const correct = answer === 'Answer';
  const reason = correct ? 'matched' : 'did not match';
  const result = new EvalResult({
    id: `result-${index}`, evalId: 'capture-smoke', testIdx: index, promptIdx: 0,
    prompt: {raw: `Question ${index}?`, label: 'authored'},
    provider: {id: 'frozen-fixture', label: 'No provider execution'},
    testCase: {vars: {question: `Question ${index}?`},
      metadata: {invarlock_id: String(index), invarlock_expected: 'Answer', category: 'authored'}},
    response: {output: answer}, score: correct ? 1 : 0, success: correct,
    failureReason: correct ? 0 : 1, error: correct ? undefined : reason,
    gradingResult: {pass: correct, score: correct ? 1 : 0, reason},
    metadata: {category: 'authored'}, persisted: false,
  });
  return sanitize(result.toEvaluateResult());
});
fs.writeFileSync(process.argv[3], JSON.stringify(rows));
""")
    output = tmp_path / "promptfoo-native.json"
    environment = {
        **os.environ,
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_CONFIG_DIR": str(tmp_path / "promptfoo-config"),
        "LOG_LEVEL": "error",
    }
    completed = subprocess.run(
        [node, str(script), modules[0].resolve().as_uri(), str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = json.loads(output.read_text())
    assert rows[1]["failureReason"] == 1
    return rows, "0.121.19", ["0", "1"], None


@pytest.mark.parametrize("evaluator", ["lm-evaluation-harness", "promptfoo"])
def test_pinned_original_sdk_serialization(tmp_path, monkeypatch, evaluator):
    rows, version, ids, projection = (
        _harness_rows(tmp_path, monkeypatch)
        if evaluator == "lm-evaluation-harness"
        else _promptfoo_rows(tmp_path)
    )
    path = tmp_path / "export.json"
    identity = {
        "source": {"name": evaluator, "version": version},
        "run_id": "capture-smoke",
        "artifact_digest": "sha256:" + "a" * 64,
        "input_projection": projection,
    }
    run = export_evaluator_result(
        evaluator,
        rows,
        path,
        expected_ids=ids,
        source_version=version,
        **{key: value for key, value in identity.items() if key != "source"},
    )
    assert [row["id"] for row in run["records"]] == ids
    assert [row["output"] for row in run["records"]] == ["Answer", "Different"]
    assert all(
        row["expected"] == "Answer" and row["error"] is None for row in run["records"]
    )
    assert run == load_run(path, adapter="evaluator-json", **identity)
    with pytest.raises(EvaluationRecordsError):
        export_evaluator_result(
            evaluator,
            rows,
            tmp_path / "incomplete.json",
            expected_ids=ids + ["missing"],
            source_version=version,
            **{key: value for key, value in identity.items() if key != "source"},
        )
    assert not (tmp_path / "incomplete.json").exists()
    if evaluator == "lm-evaluation-harness":
        likelihood_rows, _, _, _ = _harness_rows(
            tmp_path / "likelihood", monkeypatch, likelihood=True
        )
        assert likelihood_rows[0]["arguments"] == {
            "gen_args_0": {"arg_0": "Question 0?", "arg_1": "Answer"}
        }
        assert likelihood_rows[0]["filtered_resps"] == [["-0.5", "True"]]
        likelihood_path = tmp_path / "likelihood-export.json"
        likelihood_run = export_evaluator_result(
            evaluator,
            likelihood_rows,
            likelihood_path,
            expected_ids=ids,
            source_version=version,
            **{key: value for key, value in identity.items() if key != "source"},
        )
        assert all(row["output"] is None for row in likelihood_run["records"])
        assert (
            evaluator_input_capabilities(likelihood_run)[
                "normalized_nll_per_utf8_byte"
            ]["usable_count"]
            == 2
        )
        assert likelihood_run == load_run(
            likelihood_path, adapter="evaluator-json", **identity
        )
        likelihood_rows[0]["filtered_resps"][0][0] = "-9.0"
        with pytest.raises(EvaluationRecordsError):
            export_evaluator_result(
                evaluator,
                likelihood_rows,
                tmp_path / "changed-likelihood.json",
                expected_ids=ids,
                source_version=version,
                **{key: value for key, value in identity.items() if key != "source"},
            )
        assert not (tmp_path / "changed-likelihood.json").exists()
