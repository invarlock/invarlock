"""Execute local task callbacks through pinned evaluator orchestration APIs.

The callback owns model execution. These custom model/provider implementations
are invoked by each evaluator, and keep its native export plus callback facts.
Optional evaluator packages are imported only by the selected execution path.
"""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path

EVALUATORS = (
    "lm-evaluation-harness",
    "inspect-ai",
    "promptfoo",
    "lighteval",
    "garak",
    "openai-evals",
    "langfuse",
)


PROMPTFOO_NETWORK_GUARD = """
import net from 'node:net';
import http from 'node:http';
import https from 'node:https';
import tls from 'node:tls';
import dns from 'node:dns';
import dgram from 'node:dgram';
import http2 from 'node:http2';
import {syncBuiltinESMExports} from 'node:module';
const endpoint = new URL(process.env.LIVE_TASK_URL);
if(endpoint.protocol!=='http:' || endpoint.hostname!=='127.0.0.1' || endpoint.pathname!=='/task')
  throw Error('invalid local callback endpoint');
const blocked=()=>{throw Error('live Promptfoo capture forbids external networking');};
const socketConnect=net.Socket.prototype.connect;
net.Socket.prototype.connect=function(...args){
  let options=args[0];
  if(Array.isArray(options)) options=options[0];
  if(typeof options!=='object') options={port:args[0],host:typeof args[1]==='string'?args[1]:'localhost'};
  if(!options || options.path || (options.host??options.hostname)!==endpoint.hostname ||
     Number(options.port)!==Number(endpoint.port)) blocked();
  return socketConnect.apply(this,args);
};
const request=http.request;
function localRequest(...args){
  const first=args[0];
  if(args[1] && typeof args[1]==='object' &&
     ['host','hostname','port','path','protocol','createConnection','socketPath','agent'].some(k=>k in args[1])) blocked();
  const options=typeof first==='string'||first instanceof URL ? new URL(first) : first;
  if(!options || (options.hostname??options.host)!==endpoint.hostname ||
     Number(options.port)!==Number(endpoint.port) ||
     (options.pathname??options.path??'/')!==endpoint.pathname ||
     (options.protocol??'http:')!=='http:') blocked();
  return request.apply(this,args);
}
http.request=localRequest;
http.get=function(...args){const request=localRequest(...args);request.end();return request;};
https.request=https.get=tls.connect=http2.connect=blocked;
dgram.Socket.prototype.send=dgram.Socket.prototype.connect=blocked;
for(const name of Object.keys(dns))
  if(name.startsWith('resolve')||name==='lookup'||name==='lookupService'||name==='reverse') dns[name]=blocked;
for(const name of Object.keys(dns.promises))
  if(name.startsWith('resolve')||name==='lookup'||name==='lookupService'||name==='reverse') dns.promises[name]=blocked;
const originalFetch=globalThis.fetch;
globalThis.fetch=(input,options={})=>{
  const url=new URL(typeof input==='string'||input instanceof URL?input:input.url);
  if(url.href!==endpoint.href) blocked();
  return originalFetch(input,{...options,redirect:'error'});
};
syncBuiltinESMExports();
"""


def _json(value):
    if dataclasses.is_dataclass(value):
        return _json(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


class Capture:
    """Record every attempted callback before propagating execution failures."""

    def __init__(self, cases, task, workdir):
        if not cases or any(
            not isinstance(c.get("id"), str)
            or not c["id"]
            or not isinstance(c.get("input"), str)
            or not isinstance(c.get("expected"), str)
            or not isinstance(c.get("metadata", {}), dict)
            for c in cases
        ):
            raise ValueError(
                "live capture requires text inputs, references and nonempty IDs"
            )
        self.cases = {c["id"]: copy.deepcopy(c) for c in cases}
        if len(self.cases) != len(cases):
            raise ValueError("live capture requires unique case IDs")
        self.task, self.workdir, self.results = task, Path(workdir), {}
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.journal = self.workdir / "task-results.jsonl"
        self.journal.touch(exist_ok=False)

    def call(self, ident):
        if ident not in self.cases or ident in self.results:
            raise ValueError("evaluator requested an unknown or repeated case")
        self.results[ident] = None
        try:
            result = self.task(copy.deepcopy(self.cases[ident]))
            if not isinstance(result, dict) or "output" not in result:
                raise ValueError("task must return a result with output")
            if result["output"] is not None and not isinstance(result["output"], str):
                raise ValueError("task output must be text or null")
            if not isinstance(result.get("metadata", {}), dict):
                raise ValueError("task metadata must be an object")
            metadata = dict(self.cases[ident].get("metadata", {}))
            for key, value in result.get("metadata", {}).items():
                if key in metadata and metadata[key] != value:
                    raise ValueError(
                        "task metadata conflicts with frozen case metadata"
                    )
                metadata[key] = value
            result = {**result, "metadata": metadata}
            json.dumps(result, allow_nan=False)
        except Exception as exc:
            self._record(
                ident,
                {
                    "output": None,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )
            raise
        self._record(ident, result)
        return result

    def _record(self, ident, result):
        self.results[ident] = result
        with self.journal.open("a") as stream:
            stream.write(json.dumps({"id": ident, **result}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def generation(self, ident):
        result = self.call(ident)
        if result.get("error") or result["output"] is None:
            raise RuntimeError(result.get("error") or "generation returned no text")
        return result["output"]

    def complete(self):
        if set(self.results) != set(self.cases) or any(
            v is None for v in self.results.values()
        ):
            raise ValueError("evaluator did not execute every frozen case")

    def check_prompt(self, ident, prompt):
        if ident not in self.cases or prompt != self.cases[ident]["input"]:
            raise ValueError("evaluator changed the frozen prompt")


def _langfuse(capture):
    from langfuse import Langfuse

    client = Langfuse(
        public_key="local-live-capture",
        secret_key="local-disabled",
        tracing_enabled=False,
    )
    data = [
        {
            "input": case["input"],
            "expected_output": case["expected"],
            "metadata": {**case.get("metadata", {}), "invarlock_id": case["id"]},
        }
        for case in capture.cases.values()
    ]

    def callback(*, item, **kwargs):
        ident = item["metadata"]["invarlock_id"]
        capture.check_prompt(ident, item["input"])
        result = capture.call(ident)
        item["metadata"].update(result["metadata"])
        if result.get("error"):
            item["metadata"]["invarlock_error"] = result["error"]
        return result["output"]

    result = client.run_experiment(
        name="local-live-capture",
        run_name="local-live-capture",
        data=data,
        task=callback,
        max_concurrency=1,
    )
    return {
        "name": result.name,
        "run_name": result.run_name,
        "experiment_id": result.experiment_id,
        "item_results": [
            {
                "item": row.item,
                "output": row.output,
                "evaluations": [],
                "trace_id": row.trace_id,
                "dataset_run_id": row.dataset_run_id,
            }
            for row in result.item_results
        ],
        "run_evaluations": [],
    }


def _inspect(capture):
    from inspect_ai import Task, eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import write_eval_log
    from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, modelapi
    from inspect_ai.solver import solver

    active = contextvars.ContextVar("live_inspect_sample")

    @solver
    def callback_solver():
        async def solve(state, generate):
            token = active.set(state)
            try:
                return await generate(state)
            finally:
                active.reset(token)

        return solve

    @modelapi(name="local_live_callback")
    class CallbackModel(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            state = active.get()
            ident = str(state.sample_id)
            if len(input) != 1 or input[0].role != "user":
                raise ValueError("Inspect changed the frozen message inventory")
            capture.check_prompt(ident, input[0].content)
            result = capture.call(ident)
            state.metadata.update(result["metadata"])
            if result.get("error") or result["output"] is None:
                raise RuntimeError(result.get("error") or "generation returned no text")
            return ModelOutput.from_content(
                model=self.model_name, content=result["output"]
            )

    samples = [
        Sample(
            id=c["id"],
            input=c["input"],
            target=c["expected"],
            metadata=c.get("metadata", {}),
        )
        for c in capture.cases.values()
    ]
    logs = eval(
        Task(
            dataset=MemoryDataset(samples),
            solver=callback_solver(),
            name="local-live-capture",
        ),
        model=Model(
            CallbackModel("local-live-capture"), config=GenerateConfig(max_retries=0)
        ),
        log_dir=str(capture.workdir / "inspect"),
        max_samples=1,
        fail_on_error=False,
        display="none",
        log_format="json",
    )
    if len(logs) != 1:
        raise ValueError("Inspect returned an unexpected log inventory")
    destination = capture.workdir / "inspect-native.json"
    write_eval_log(logs[0], str(destination), format="json")
    return json.loads(destination.read_bytes())


def _harness(capture):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bindings import rebind_harness_metadata
    from datasets import Dataset
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.loggers.evaluation_tracker import EvaluationTracker

    class LocalTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {"test": Dataset.from_list(list(capture.cases.values()))}

    class CallbackLM(LM):
        def generate_until(self, requests):
            outputs = []
            for request in requests:
                ident = request.doc["id"]
                capture.check_prompt(ident, request.args[0])
                outputs.append(capture.generation(ident))
            return outputs

        def loglikelihood(self, requests):
            raise ValueError(
                "this generation campaign requires generated text; likelihood-only tasks need a separate native request"
            )

        def loglikelihood_rolling(self, requests):
            raise ValueError(
                "rolling likelihood is outside the reference-continuation campaign"
            )

    task = LocalTask(
        config={
            "task": "local_live_capture",
            "dataset_path": "local",
            "test_split": "test",
            "doc_to_text": "input",
            "doc_to_target": "expected",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "generation_kwargs": {"until": []},
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
        }
    )
    task.set_fewshot_seed(0)
    results = evaluator.evaluate(
        lm=CallbackLM(),
        task_dict={"local_live_capture": task},
        log_samples=True,
        bootstrap_iters=0,
        verbosity="ERROR",
    )
    rows = results["samples"]["local_live_capture"]
    for row in rows:
        ident = row["doc"]["id"]
        row["metadata"] = {
            **rebind_harness_metadata(
                capture.results[ident]["metadata"],
                capture.cases[ident],
                list(capture.cases.values()),
                row["doc"],
            ),
            "invarlock_id": ident,
        }
    tracker = EvaluationTracker(output_path=str(capture.workdir / "harness"))
    tracker.general_config_tracker.log_experiment_args(
        model_source="local-callback",
        model_args={"model": "live-task"},
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    tracker.save_results_aggregated(results)
    tracker.save_results_samples("local_live_capture", rows)
    paths = list((capture.workdir / "harness").rglob("samples_*.jsonl"))
    if len(paths) != 1:
        raise ValueError("Harness sample log inventory differs")
    (capture.workdir / "capture-profile.json").write_text(
        json.dumps(
            {"input_projection": {"kind": "json-pointer", "pointer": "/input/input"}}
        )
    )
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def _openai(capture):
    # The SDK creates an unused OpenAI client at import time, even for a custom
    # local CompletionFn. This public placeholder authorizes no service calls.
    from evals import Eval, record
    from evals.api import CompletionResult
    from evals.base import RunSpec
    from evals.record import LocalRecorder

    class Result(CompletionResult):
        def __init__(self, output):
            self.output = output

        def get_completions(self):
            return [self.output]

    class Callback:
        def __call__(self, prompt, **kwargs):
            ident = kwargs["case_id"]
            capture.check_prompt(ident, prompt)
            result = capture.call(ident)
            record.record_event(
                "sampling",
                {
                    "prompt": prompt,
                    "sampled": result["output"],
                    "metadata": result["metadata"],
                },
                sample_id=ident,
            )
            if result.get("error"):
                record.record_event(
                    "error", {"error": result["error"]}, sample_id=ident
                )
            return Result(result["output"])

    class LocalEval(Eval):
        def eval_sample(self, sample, rng):
            output = self.completion_fn(
                sample["input"], case_id=sample["id"]
            ).get_completions()[0]
            record.record_event(
                "match",
                {
                    "correct": output == sample["expected"],
                    "expected": sample["expected"],
                },
                sample_id=sample["id"],
            )

        def run(self, recorder):
            return self.eval_all_samples(
                recorder, list(capture.cases.values()), show_progress=False
            )

    path = capture.workdir / "openai-events.jsonl"
    recorder = LocalRecorder(
        str(path),
        RunSpec(
            completion_fns=["local-callback"],
            eval_name="live.capture",
            base_eval="live",
            split="capture",
            run_config={},
            created_by="local-live-capture",
            run_id="local-live-capture",
        ),
    )
    evaluation = LocalEval(
        completion_fns=[Callback()],
        eval_registry_path=capture.workdir,
        name="live.capture",
    )
    evaluation.run(recorder)
    recorder.flush_events()
    return {
        "events": [
            json.loads(line)
            for line in path.read_text().splitlines()
            if "type" in json.loads(line)
        ]
    }


def _garak(capture):
    from garak import _config
    from garak.detectors.any import AnyOutput
    from garak.evaluators.base import ZeroToleranceEvaluator
    from garak.generators.function import Single
    from garak.harnesses.base import Harness
    from garak.probes.test import Test

    _config.load_config(site_config_filename=str(capture.workdir / "absent-site.yaml"))
    _config.run.generations = 1
    _config.system.parallel_attempts = 1
    _config.system.parallel_requests = 1
    _config.system.show_z = False
    _config.run.langproviders = []
    _config.run.target_lang = "en"
    _config.buffmanager.buffs = []
    _config.transient.run_id = "local-live-capture"
    report = capture.workdir / "garak.report.jsonl"
    _config.transient.report_filename = str(report)
    module_name = "live_garak_callback"
    callback_module = types.ModuleType(module_name)
    ordered = list(capture.cases)
    invoked = []

    def callback(prompt, **kwargs):
        ident = ordered[len(invoked)]
        capture.check_prompt(ident, prompt)
        invoked.append(ident)
        result = capture.call(ident)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return [result["output"]] if result["output"] is not None else []

    callback_module.generate = callback
    if module_name in sys.modules:
        raise ValueError("Garak callback module already exists")
    sys.modules[module_name] = callback_module
    try:
        with report.open("x") as stream:
            _config.transient.reportfile = stream
            probe = Test()
            probe.prompts = [c["input"] for c in capture.cases.values()]
            probe.generations = 1
            probe.parallel_attempts = 1
            Harness().run(
                Single(name=f"{module_name}#generate"),
                [probe],
                [AnyOutput()],
                ZeroToleranceEvaluator(),
            )
    finally:
        sys.modules.pop(module_name, None)
        _config.transient.reportfile = None
        hitlog = getattr(_config.transient, "hitlogfile", None)
        if hitlog is not None:
            hitlog.close()
            _config.transient.hitlogfile = None
    entries = [json.loads(line) for line in report.read_text().splitlines()]
    attempts = [
        r for r in entries if r.get("entry_type") == "attempt" and r["status"] == 2
    ]
    sources = []
    for row in attempts:
        ident = ordered[row["seq"]]
        result = capture.results[ident]
        row["metadata"] = result["metadata"]
        sources.append(
            {
                "native_id": f"{row['uuid']}:0",
                "id": ident,
                "input": capture.cases[ident]["input"],
                "expected": capture.cases[ident]["expected"],
                "output": result["output"],
                "metadata": result["metadata"],
            }
        )
    return {
        "attempts": attempts,
        "source_cases": sources,
        "capture_scope": "Custom frozen prompts; AnyOutput is an output-presence detector, not a quality score.",
    }


NLTK_ARCHIVES = {
    "punkt": "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec",
    "punkt_tab": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
}


@contextmanager
def _nltk_resources(workdir):
    """Admit pinned public NLP data before unrelated registry task imports.

    Setup supplies nltk_data/packages/tokenizers/{name}.zip from the public
    nltk/nltk_data repository. Capture never downloads or trusts loose files.
    """
    import nltk

    archives = Path(
        os.environ.get("INVARLOCK_NLTK_ARCHIVE_DIR", str(Path.home() / "nltk_data"))
    )
    destination = workdir / "nltk-data"
    destination.mkdir()
    inventory = {}
    for name, expected in NLTK_ARCHIVES.items():
        path = archives / "tokenizers" / f"{name}.zip"
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                f"LightEval setup requires pinned NLTK tokenizers/{name}.zip ({expected})"
            )
        with zipfile.ZipFile(path) as bundle:
            entries = bundle.infolist()
            if sum(row.file_size for row in entries) > 100 * 1024 * 1024:
                raise ValueError("NLTK archive exceeds extraction bound")
            for entry in entries:
                relative = Path(entry.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.parts[0] != name
                ):
                    raise ValueError("invalid pinned NLTK archive member")
                if entry.is_dir():
                    continue
                raw = bundle.read(entry)
                target = destination / "tokenizers" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                inventory[str(target.relative_to(destination))] = hashlib.sha256(
                    raw
                ).hexdigest()
    (workdir / "nltk-resources.json").write_text(
        json.dumps({"archives": NLTK_ARCHIVES, "files": inventory}, indent=2)
    )
    previous_paths, previous_download = nltk.data.path, nltk.download

    def local_only(name, *args, **kwargs):
        if name not in NLTK_ARCHIVES:
            raise ValueError(
                f"NLTK resource {name!r} is outside the admitted local setup"
            )
        nltk.data.find(f"tokenizers/{name}")
        return True

    nltk.data.path = [str(destination)]
    nltk.download = local_only
    try:
        yield
    finally:
        nltk.data.path, nltk.download = previous_paths, previous_download


LIGHTEVAL_REGISTRY_RESOURCE = {
    "url": "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/74577e0bd1921dd2adf489d3487a6e1ccbb08b82/tinyBenchmarks/tinyBenchmarks.pkl",
    "sha256": "c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284",
    "size": 5230626,
}


def lighteval_resource(archive=None):
    """Stage or verify opaque registry bytes; never deserialize this resource.

    LightEval imports unrelated benchmark definitions when building its registry.
    Their constructors download this file when absent. Setup supplies exact bytes
    first; our custom task selects only Metrics.exact_match and never loads it.
    """
    spec = importlib.util.find_spec("lighteval")
    if spec is None or spec.origin is None:
        raise ValueError("install the pinned LightEval SDK before resource setup")
    target = Path(spec.origin).parent / "tasks/tasks/tinyBenchmarks.pkl"
    source = Path(archive) if archive is not None else target
    try:
        with source.open("rb") as stream:
            raw = stream.read(LIGHTEVAL_REGISTRY_RESOURCE["size"] + 1)
    except OSError as exc:
        raise ValueError(
            "stage the pinned LightEval registry resource before capture"
        ) from exc
    if (
        len(raw) != LIGHTEVAL_REGISTRY_RESOURCE["size"]
        or hashlib.sha256(raw).hexdigest() != LIGHTEVAL_REGISTRY_RESOURCE["sha256"]
    ):
        raise ValueError("LightEval registry resource differs from its independent pin")
    if archive is not None:
        if target.exists():
            with target.open("rb") as stream:
                existing = stream.read(LIGHTEVAL_REGISTRY_RESOURCE["size"] + 1)
            if existing != raw:
                raise ValueError("refuse to replace an unexpected LightEval resource")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
    return {
        **LIGHTEVAL_REGISTRY_RESOURCE,
        "scope": "Opaque registry setup bytes; exact-match task does not deserialize this file",
    }


def _lighteval(capture):
    (capture.workdir / "registry-resource.json").write_text(
        json.dumps(lighteval_resource())
    )
    with _nltk_resources(capture.workdir):
        return _lighteval_pipeline(capture)


def _lighteval_pipeline(capture):
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.metrics.metrics import Metrics
    from lighteval.models.abstract_model import LightevalModel, ModelConfig
    from lighteval.models.model_output import ModelResponse
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    from lighteval.tasks.lighteval_task import LightevalTaskConfig
    from lighteval.tasks.requests import Doc
    from lighteval.utils.cache_management import SampleCache

    dataset = capture.workdir / "dataset"
    dataset.mkdir()
    (dataset / "test.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in capture.cases.values())
    )

    def prompt(line, task_name):
        return Doc(
            query=line["input"],
            choices=[line["expected"]],
            gold_index=0,
            specific={"id": line["id"]},
            task_name=task_name,
        )

    custom = types.ModuleType("live_lighteval_tasks")
    custom.TASKS_TABLE = [
        LightevalTaskConfig(
            name="local_live_capture",
            prompt_function=prompt,
            hf_repo=str(dataset),
            hf_subset="default",
            metrics=[Metrics.exact_match],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            num_fewshots=0,
            num_samples=[1],
        )
    ]

    class CallbackModel(LightevalModel):
        def __init__(self):
            self.config = ModelConfig(
                model_name="local-live-callback",
                cache_dir=str(capture.workdir / "cache"),
            )
            self._cache = SampleCache(self.config)

        @property
        def tokenizer(self):
            raise ValueError("the callback owns tokenization")

        @property
        def add_special_tokens(self):
            return False

        @property
        def max_length(self):
            raise ValueError("the callback owns its context limit")

        def greedy_until(self, docs):
            responses = []
            for doc in docs:
                ident = doc.specific["id"]
                capture.check_prompt(ident, doc.query)
                responses.append(
                    ModelResponse(input=doc.query, text=[capture.generation(ident)])
                )
            return responses

        def loglikelihood(self, docs):
            raise ValueError(
                "this pipeline captures generated answers with explicit continuation metadata"
            )

        def loglikelihood_rolling(self, docs):
            raise ValueError("rolling likelihood is outside this capture")

    tracker = EvaluationTracker(
        output_dir=str(capture.workdir / "lighteval"), save_details=True
    )
    pipeline = Pipeline(
        tasks="local_live_capture",
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            custom_tasks_directory=custom,
            remove_reasoning_tags=False,
            bootstrap_iters=0,
        ),
        evaluation_tracker=tracker,
        model=CallbackModel(),
    )
    pipeline.evaluate()
    pipeline.save_and_push_results()
    records = []
    for details in tracker.details_logger.details.values():
        for detail in details:
            ident = detail.doc.specific["id"]
            records.append(
                {
                    "id": ident,
                    "doc": _json(detail.doc),
                    "model_response": _json(detail.model_response),
                    "metric_result": detail.metric,
                    "metadata": capture.results[ident]["metadata"],
                }
            )
    return records


def _promptfoo(capture):
    import hmac
    import http.server
    import secrets
    import shutil
    import subprocess
    import threading

    package = Path(os.environ["INVARLOCK_PROMPTFOO_PACKAGE"]).resolve()
    node = shutil.which("node")
    if (
        not node
        or json.loads((package / "package.json").read_text())["version"] != "0.121.19"
    ):
        raise ValueError("live Promptfoo capture requires Node and promptfoo 0.121.19")
    token = secrets.token_hex(32)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/task" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), token
            ):
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                self.send_error(413)
                return
            try:
                body = json.loads(self.rfile.read(length))
                ident = body["id"]
                capture.check_prompt(ident, body["prompt"])
                result = capture.call(ident)
                response = {"output": result["output"], "metadata": result["metadata"]}
                if result.get("error"):
                    response["error"] = result["error"]
                encoded = json.dumps(response, allow_nan=False).encode()
            except Exception as exc:
                encoded = json.dumps({"error": str(exc)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    data = capture.workdir / "promptfoo-cases.json"
    data.write_text(json.dumps(list(capture.cases.values())))
    script = capture.workdir / "promptfoo-capture.mjs"
    script.write_text(
        PROMPTFOO_NETWORK_GUARD
        + """
import fs from 'node:fs';
const api = await import(process.env.LIVE_PROMPTFOO_ENTRY);
const evaluate = api.evaluate ?? api.default?.evaluate;
if (!evaluate) throw Error('Promptfoo evaluate API unavailable');
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const provider = {id:()=> 'local-live-callback', callApi:async (prompt, context)=> {
  const response = await fetch(process.env.LIVE_TASK_URL, {method:'POST',
    headers:{'Authorization':process.env.LIVE_TASK_TOKEN,'Content-Type':'application/json'},
    body:JSON.stringify({id:context.vars.case_id,prompt})});
  if(!response.ok) throw Error(`local task bridge returned ${response.status}`);
  return await response.json();
}};
const result = await evaluate({prompts:['{{prompt}}'],providers:[provider],tests:cases.map(c=>({
  vars:{prompt:c.input,case_id:c.id},metadata:{...c.metadata,invarlock_id:c.id,invarlock_expected:c.expected},
  assert:[{type:'equals',value:c.expected}]
}))}, {maxConcurrency:1,cache:false,writeLatestResults:false});
const rows=(await result.toEvaluateSummary()).results;
fs.writeFileSync(process.argv[3], JSON.stringify(rows));
"""
    )
    native = capture.workdir / "promptfoo-native.json"
    entry = package / json.loads((package / "package.json").read_text())["main"]
    try:
        result = subprocess.run(
            [node, str(script), str(data), str(native)],
            cwd=capture.workdir,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(capture.workdir / "promptfoo-home"),
                "TMPDIR": str(capture.workdir),
                "LANG": "C.UTF-8",
                "PROMPTFOO_DISABLE_TELEMETRY": "1",
                "PROMPTFOO_CONFIG_DIR": str(capture.workdir / "promptfoo"),
                "LIVE_PROMPTFOO_ENTRY": entry.as_uri(),
                "LIVE_TASK_URL": f"http://127.0.0.1:{server.server_port}/task",
                "LIVE_TASK_TOKEN": token,
                "LOG_LEVEL": "error",
            },
            capture_output=True,
            text=True,
            timeout=3600,
        )
        (capture.workdir / "promptfoo-command.json").write_text(
            json.dumps(
                {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        )
        result.check_returncode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    rows = json.loads(native.read_bytes())
    for row in rows:
        ident = row["testCase"]["metadata"]["invarlock_id"]
        if ident in capture.results and capture.results[ident] is not None:
            row["testCase"]["metadata"].update(
                capture.results[ident].get("metadata", {})
            )
    return rows


def run(evaluator, cases, task, workdir):
    """Run a framework-owned callback once per case and return its native payload."""
    if evaluator not in EVALUATORS:
        raise ValueError("unsupported live evaluator")
    capture = Capture(cases, task, workdir)
    driver = {
        "langfuse": _langfuse,
        "inspect-ai": _inspect,
        "lm-evaluation-harness": _harness,
        "promptfoo": _promptfoo,
        "lighteval": _lighteval,
        "garak": _garak,
        "openai-evals": _openai,
    }[evaluator]
    previous = {key: os.environ.get(key) for key in ("OPENAI_API_KEY", "EVALS_THREADS")}
    try:
        if evaluator == "openai-evals":
            os.environ.setdefault("OPENAI_API_KEY", "local-callback-unused")
            os.environ["EVALS_THREADS"] = "1"
        payload = driver(capture)
    finally:
        if evaluator == "openai-evals":
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    capture.complete()
    payload = _json(payload)
    (capture.workdir / "native.json").write_text(
        json.dumps(payload, allow_nan=False) + "\n"
    )
    return payload
