"""Actual SDK serialization of retained records, without fresh model execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path


def roundtrip(evaluator, payload, *, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pins = {
        "inspect-ai": ("inspect-ai", "0.3.254"),
        "langfuse": ("langfuse", "4.14.1"),
        "lm-evaluation-harness": ("lm-eval", "0.4.12"),
    }
    if evaluator in pins:
        package, expected = pins[evaluator]
        assert version(package) == expected
    if evaluator == "inspect-ai":
        from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec

        from invarlock.evaluation_records.integrations import _serialize

        value = deepcopy(payload)
        for sample in value["samples"]:
            sample["epoch"] = 1
        value["eval"] = EvalSpec(
            # This is the SDK export creation time, not the model execution time.
            created=datetime.now(UTC).isoformat(),
            task="retained-serialization",
            dataset=EvalDataset(samples=len(payload["samples"])),
            model="retained-model-measurements",
            config=EvalConfig(),
        )
        return _serialize(
            evaluator, EvalLog.model_validate(value), version("inspect-ai")
        )
    if evaluator == "langfuse":
        from langfuse.experiment import ExperimentItemResult, ExperimentResult

        from invarlock.evaluation_records.langfuse import serialize_experiment_result

        result = payload["result"]
        value = ExperimentResult(
            name=result["name"],
            run_name=result["run_name"],
            description="SDK serialization of retained observations; no new model calls",
            item_results=[
                ExperimentItemResult(**item) for item in result["item_results"]
            ],
            run_evaluations=[],
            experiment_id=result["experiment_id"],
        )
        return serialize_experiment_result(value, version("langfuse"))["result"]
    if evaluator == "lm-evaluation-harness":
        from lm_eval.api.instance import Instance
        from lm_eval.loggers.evaluation_tracker import EvaluationTracker

        samples = []
        for row in payload:
            facts = row["metadata"].get("invarlock_likelihood")
            # Only the likelihood number is an observed measurement. The unused
            # greedy flag exercises the logger's tuple representation and is
            # explicitly a serialization fixture, not a retained measurement.
            response = (
                (facts["logprob_sum"], False) if facts else row["filtered_resps"][0]
            )
            instance = Instance(
                request_type="loglikelihood" if facts else "generate_until",
                doc=row["doc"],
                arguments=tuple(row["arguments"][0]),
                idx=0,
                metadata=("retained-serialization", row["doc_id"], 1),
                resps=[response],
                filtered_resps={"none": response},
            )
            samples.append(
                {
                    **deepcopy(row),
                    "arguments": [instance.args],
                    "resps": [instance.resps],
                    "filtered_resps": [instance.filtered_resps["none"]],
                    "filter": "none",
                    "metrics": [],
                }
            )
        tracker = EvaluationTracker(output_path=str(tmp_path / "harness"))
        tracker.general_config_tracker.log_experiment_args(
            model_source="retained-serialization",
            model_args={"model": "retained-model-measurements"},
            system_instruction=None,
            chat_template=None,
            fewshot_as_multiturn=False,
        )
        tracker.save_results_aggregated({"results": {}})
        tracker.save_results_samples("retained-serialization", samples)
        files = list((tmp_path / "harness").rglob("samples_*.jsonl"))
        assert len(files) == 1
        return [json.loads(line) for line in files[0].read_text().splitlines()]
    if evaluator != "promptfoo":
        raise ValueError("unknown original evaluator")
    package = Path(os.environ["INVARLOCK_PROMPTFOO_PACKAGE"])
    assert json.loads((package / "package.json").read_text())["version"] == "0.121.19"
    modules = list((package / "dist/src").glob("evalResult-*.js"))
    assert len(modules) == 1
    source, output, script = (
        tmp_path / name for name in ("input.json", "output.json", "capture.mjs")
    )
    source.write_text(json.dumps(payload, allow_nan=False))
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
const EvalResult = Object.values(native).find(v => typeof v === 'function' && v.name === 'EvalResult');
const sanitize = Object.values(native).find(v => typeof v === 'function' && v.name === 'sanitizeResultForJsonlArtifact');
if (!EvalResult || !sanitize) throw Error('pinned serializers missing');
const rows = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')).map((row, index) => {
  const result = new EvalResult({...row, testIdx:index, id:`retained-${index}`,
    evalId:'retained-serialization', persisted:false,
    provider:{id:'retained-model-measurements',label:'No new provider execution'}});
  return sanitize(result.toEvaluateResult());
});
fs.writeFileSync(process.argv[4], JSON.stringify(rows));
""")
    node = shutil.which("node")
    assert node
    subprocess.run(
        [node, str(script), modules[0].resolve().as_uri(), str(source), str(output)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PROMPTFOO_DISABLE_TELEMETRY": "1",
            "PROMPTFOO_CONFIG_DIR": str(tmp_path / "config"),
            "LOG_LEVEL": "error",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(output.read_bytes())
