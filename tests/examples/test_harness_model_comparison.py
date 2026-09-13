"""Authored tokenizer and model doubles exercise capture; no model inference runs."""

from __future__ import annotations

import copy
import importlib.util
import json
import signal
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.engine import capture_evaluator_run, case_set_digest, freeze_case_set

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/captured-results/harness_model_comparison.py"
)


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("harness_model_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            return "<bos>"
        # A standalone suffix loses whitespace, like the reviewed tokenizer.
        return "".join(chr(token) for token in ids).lstrip()


class FakeLM:
    loads = []
    calls = []
    fail_at = None
    bad_results = None
    backend = "causal"
    max_length = 512
    logits_cache = False
    enable_thinking = False
    softmax_dtype = "float32"

    def __init__(self, **kwargs):
        self.loads.append(kwargs)
        self.tokenizer = Tokenizer()
        self.cache_hook = SimpleNamespace(dbdict=None)
        self.model = SimpleNamespace(
            parameters=lambda: [
                SimpleNamespace(device=SimpleNamespace(type="mps"), dtype="float16")
            ]
        )

    def tok_encode(self, text):
        return self.tokenizer.encode(text)

    def _encode_pair(self, text, reference):
        left = self.tok_encode(text)
        return left, self.tok_encode(text + reference)[len(left) :]

    def loglikelihood(self, requests, **kwargs):
        self.calls.append(requests)
        if self.fail_at == len(self.calls):
            raise RuntimeError("secret provider exception must not be retained")
        return self.bad_results if self.bad_results is not None else [(-2.5, True)]


@pytest.fixture
def material(tmp_path, helper, monkeypatch):
    FakeLM.loads, FakeLM.calls, FakeLM.fail_at, FakeLM.bad_results = [], [], None, None
    setters = []
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        set_num_threads=lambda n: setters.append(("threads", n)),
        manual_seed=lambda n: setters.append(("seed", n)),
        use_deterministic_algorithms=lambda x: setters.append(("deterministic", x)),
        mps=SimpleNamespace(
            set_per_process_memory_fraction=lambda n: setters.append(("memory", n))
        ),
        float16="float16",
        float32="float32",
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    modules = {
        "lm_eval.models.huggingface": {"HFLM": FakeLM},
        "lm_eval.api.model": {},
        "lm_eval.api.metrics": {},
        "lm_eval.api.instance": {
            "Instance": lambda **kwargs: SimpleNamespace(**kwargs)
        },
        "transformers.models.mistral.modeling_mistral": {},
        "transformers.tokenization_utils_base": {},
        "transformers": {
            "AutoTokenizer": SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()
            )
        },
    }
    facts = {}
    for index, (name, attrs) in enumerate(modules.items()):
        path = tmp_path / f"module{index}.py"
        path.write_text(f"# authored module {index}\n")
        monkeypatch.setitem(
            sys.modules, name, SimpleNamespace(__file__=str(path), **attrs)
        )
        if name in helper.UPSTREAM_HASHES:
            facts[name] = helper.file_fact(path, name)["sha256"].removeprefix("sha256:")
    monkeypatch.setattr(helper, "UPSTREAM_HASHES", facts)
    monkeypatch.setattr(
        helper.importlib.metadata,
        "version",
        lambda name: "0.4.12" if name == "lm-eval" else "authored-test",
    )
    paths, models = {}, {}
    for role, (identity, revision) in helper.MODELS.items():
        root = tmp_path / role
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps(
                {"model_type": "mistral", "architectures": ["MistralForCausalLM"]}
            )
        )
        (root / "model.safetensors").write_bytes(role.encode())
        (root / "tokenizer.json").write_text('{"authored":true}')
        files = [helper.file_fact(path, path.name) for path in sorted(root.iterdir())]
        paths[role] = root
        models[role] = {
            "id": identity,
            "revision": revision,
            "files": files,
            "artifact_digest": helper.digest(files),
            "tokenizer_digest": helper.digest(
                [fact for fact in files if fact["path"] in helper.TOKENIZER_FILES]
            ),
        }
    cases = [
        {
            "id": f"case-{i:03}",
            "input": "A café serves",
            "expected": " café",
            "metadata": {"source": "authored"},
        }
        for i in range(400)
    ]
    config = copy.deepcopy(helper.CONFIGURATION)
    protocol = {
        "format": "invarlock/harness-model-comparison-v1",
        "models": models,
        "cases": cases,
        "configuration": config,
        "source": helper.SOURCE,
        "source_implementations": facts,
        "dataset": {"scope": "authored-test"},
        "capture_script_sha256": helper.file_fact(SCRIPT, "script")["sha256"],
        "policy": {
            "format": "invarlock/comparison-policy-v1",
            "expected_case_set_digest": case_set_digest(freeze_case_set(cases)),
            "metrics": [
                {
                    "name": "reference_nll",
                    "kind": "normalized_nll_per_utf8_byte",
                    "direction": "lower",
                    "unit": "nats_per_utf8_byte",
                    "aggregation": "mean",
                    "minimum_count": 400,
                    "ratio_max": 1.05,
                    "maximum_interval_width": 0.10,
                    "configuration": {
                        "configuration_digest": helper.digest(config),
                        "baseline_tokenizer_digest": models["baseline"][
                            "tokenizer_digest"
                        ],
                        "subject_tokenizer_digest": models["subject"][
                            "tokenizer_digest"
                        ],
                    },
                }
            ],
            "slices": [],
        },
    }
    return SimpleNamespace(protocol=protocol, paths=paths, torch=torch, setters=setters)


def validate(helper, protocol):
    raw = helper.canonical_bytes(protocol)
    helper.validate_protocol(protocol, raw, helper.bytes_digest(raw))
    return raw


def test_complete_distinct_models_preserve_raw_results_and_sdk_facts(
    helper, material, tmp_path
):
    raw = validate(helper, material.protocol)
    for role in helper.MODELS:
        output = tmp_path / "capture" / role
        result = helper.execute_role(
            material.protocol, raw, material.paths[role], output, role
        )
        manifest = json.loads((output / "manifest.json").read_bytes())
        tokens = json.loads((output / "tokenizations.json").read_bytes())
        assert (output / "protocol.json").read_bytes() == raw
        assert len(list((output / "progress").iterdir())) == 800
        assert result["metadata"]["harness_loglikelihood_call_count"] == 400
        assert manifest["configuration_digest"] == helper.digest(helper.CONFIGURATION)
        assert len(manifest["source_implementations"]) == 6
        for index, row in enumerate(result["records"]):
            assert row["id"] == material.protocol["cases"][index]["id"]
            assert row["metadata"] == {"source": "authored"}
            assert row["context"]["model_id"] == helper.MODELS[role][0]
            assert row["context"]["harness"]["result"] == [-2.5, True]
            assert row["likelihood"]["utf8_byte_count"] == 6
            assert tokens[index]["decoded_continuation"] == " café"
            assert Tokenizer().decode(tokens[index]["continuation_token_ids"]) == "café"
            assert (
                json.loads((output / "progress" / f"{index:06}.json").read_bytes())
                == row
            )
        run = capture_evaluator_run(
            result["records"],
            source=helper.SOURCE,
            run_id=role,
            artifact_digest=material.protocol["models"][role]["artifact_digest"],
        )
        assert len(run["records"]) == 400
    assert len(FakeLM.calls) == 800
    assert all(len(call) == 1 for call in FakeLM.calls)
    assert ("memory", 0.42) in material.setters
    assert ("threads", 4) in material.setters
    assert FakeLM.loads[0]["use_safetensors"] is True
    assert FakeLM.loads[0]["attn_implementation"] == "eager"


def test_preflight_writes_manifest_before_tokenizer_and_never_loads_model(
    helper, material, tmp_path, monkeypatch
):
    raw = validate(helper, material.protocol)
    output = tmp_path / "preflight"
    original = helper.tokenizer_only

    def tokenizer(path):
        assert (output / "manifest.json").is_file()
        assert (output / "protocol.json").read_bytes() == raw
        return original(path)

    monkeypatch.setattr(helper, "tokenizer_only", tokenizer)
    result = helper.execute_role(
        material.protocol, raw, material.paths["baseline"], output, "baseline", True
    )
    assert result == {
        "status": "tokenization_only",
        "case_count": 400,
        "model_loaded": False,
        "inference_calls": 0,
    }
    assert not FakeLM.loads and not FakeLM.calls
    assert not (output / "raw-results.json").exists()
    with pytest.raises(FileExistsError):
        helper.execute_role(
            material.protocol, raw, material.paths["baseline"], output, "baseline", True
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(extra=True), "exactly"),
        (lambda p: p.update(format="other"), "unsupported"),
        (lambda p: p["source"].update(version="new"), "unmodified"),
        (lambda p: p["configuration"].update(batch_size=True), "configuration"),
        (
            lambda p: p["configuration"].update(async_weight_loading=True),
            "configuration",
        ),
        (lambda p: p.update(capture_script_sha256="sha256:" + "0" * 64), "script"),
        (lambda p: p.update(dataset={}), "dataset"),
        (lambda p: p.update(policy=[]), "policy"),
        (lambda p: p["models"].pop("subject"), "models"),
        (
            lambda p: p["models"]["subject"].update(id=p["models"]["baseline"]["id"]),
            "identity",
        ),
        (lambda p: p["models"]["subject"].update(revision="a" * 40), "identity"),
        (lambda p: p["models"]["baseline"].update(files=[]), "complete file"),
        (
            lambda p: p["models"]["baseline"]["files"][0].update(path="../config.json"),
            "only pinned",
        ),
        (
            lambda p: p["models"]["baseline"]["files"][0].update(
                path="pytorch_model.bin"
            ),
            "only pinned",
        ),
        (
            lambda p: p["models"]["baseline"]["files"][0].update(byte_size=True),
            "positive integer",
        ),
        (lambda p: p["models"]["baseline"]["files"][0].update(sha256="a"), "SHA-256"),
        (lambda p: p["models"]["baseline"]["files"].reverse(), "sorted"),
        (
            lambda p: p["models"]["baseline"].update(artifact_digest="changed"),
            "digest mismatch",
        ),
        (
            lambda p: p["models"]["baseline"].update(tokenizer_digest="changed"),
            "digest mismatch",
        ),
        (lambda p: p["cases"].pop(), "400-case"),
        (lambda p: p["cases"][1].update(id=p["cases"][0]["id"]), "case identities"),
        (lambda p: p["cases"][0].update(id=""), "case identities"),
        (lambda p: p["cases"][0].update(input="question "), "trailing whitespace"),
        (lambda p: p["cases"][0].update(input=""), "trailing whitespace"),
        (lambda p: p["cases"][0].update(expected=None), "original string"),
        (lambda p: p["cases"][0].update(metadata=[]), "metadata"),
        (lambda p: p["cases"][0].update(input="x" * 65537), "text allowance"),
        (lambda p: p["policy"]["metrics"][0].update(ratio_max=2), "policy differs"),
        (lambda p: p["cases"].reverse(), None),
    ],
)
def test_independent_protocol_rejects_changed_contracts(
    helper, material, mutation, match
):
    protocol = copy.deepcopy(material.protocol)
    mutation(protocol)
    if match:
        with pytest.raises(ValueError, match=match):
            validate(helper, protocol)
    else:
        validate(
            helper, protocol
        )  # Order is independently pinned, membership is ID-sorted.
    assert not FakeLM.calls


def test_protocol_byte_pin_and_distinct_artifacts(helper, material):
    with pytest.raises(ValueError, match="independent byte pin"):
        helper.validate_protocol(
            material.protocol, helper.canonical_bytes(material.protocol), "wrong"
        )
    protocol = copy.deepcopy(material.protocol)
    for key in ("files", "artifact_digest", "tokenizer_digest"):
        protocol["models"]["subject"][key] = protocol["models"]["baseline"][key]
    with pytest.raises(ValueError, match="distinct model artifacts"):
        validate(helper, protocol)


@pytest.mark.parametrize("kind", ["extra", "bytes", "config", "cap"])
def test_snapshot_inventory_refuses_unpinned_or_unsafe_files(helper, material, kind):
    path = material.paths["baseline"]
    if kind == "extra":
        (path / "pytorch_model.bin.index.json").write_text("{}")
    elif kind == "bytes":
        (path / "model.safetensors").write_bytes(b"changed")
    elif kind == "config":
        (path / "config.json").write_text('{"model_type":"other"}')
        material.protocol["models"]["baseline"]["files"][0] = helper.file_fact(
            path / "config.json", "config.json"
        )
    else:
        helper.CONFIGURATION["maximum_model_file_bytes"] = 1
    with pytest.raises(ValueError):
        helper.inventory(path, material.protocol["models"]["baseline"])


@pytest.mark.parametrize("kind", ["version", "source", "protocol"])
def test_dependencies_must_match_installed_source_pins(
    helper, material, monkeypatch, kind
):
    if kind == "version":
        monkeypatch.setattr(helper.importlib.metadata, "version", lambda _: "wrong")
    elif kind == "source":
        name = next(iter(helper.UPSTREAM_HASHES))
        helper.UPSTREAM_HASHES[name] = "0" * 64
    else:
        material.protocol["source_implementations"] = {}
    with pytest.raises(ValueError):
        helper.dependencies(material.protocol)


@pytest.mark.parametrize(
    "kind", ["empty", "negative", "bool", "boundary", "context", "joined", "too_long"]
)
def test_tokenization_refuses_loss_or_truncation(helper, kind):
    lm = FakeLM()
    case = {"input": "Original", "expected": " café"}
    if kind in {"empty", "negative", "bool", "boundary"}:
        lm._encode_pair = lambda *_: (
            [79],
            {"empty": [], "negative": [-1], "bool": [True], "boundary": [2]}[kind],
        )
    elif kind == "context":
        lm.tokenizer.decode = lambda *_args, **_kwargs: "changed"
    elif kind == "joined":
        original = lm.tokenizer.decode
        lm.tokenizer.decode = lambda ids, **kwargs: (
            original(ids, **kwargs) if len(ids) == len(case["input"]) else "changed"
        )
    else:
        case["input"] = "X" * 513
    with pytest.raises(ValueError):
        helper.tokenize(lm, [case])


@pytest.mark.parametrize(
    "result",
    [[], [1], [-1, True, 4], [True, True], [1, True], [float("nan"), True], [-1, 1]],
)
def test_native_result_is_never_repaired(helper, result):
    with pytest.raises(ValueError, match="native"):
        helper.capture_record({}, result, {}, {}, "", 0)


@pytest.mark.parametrize("kind", ["call", "shape", "tokenizer", "timeout"])
def test_failures_retain_admissions_results_and_no_final_output(
    helper, material, tmp_path, monkeypatch, kind
):
    raw = validate(helper, material.protocol)
    output = tmp_path / "failure"
    if kind == "call":
        FakeLM.fail_at = 2
    elif kind == "shape":
        FakeLM.bad_results = []
    elif kind == "tokenizer":
        original = helper.tokenize
        monkeypatch.setattr(
            helper,
            "tokenize",
            lambda lm, cases: original(lm, cases) if not FakeLM.loads else [],
        )
    else:
        monkeypatch.setattr(helper, "load_model", lambda _: helper.deadline(None, None))
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        helper.execute_role(
            material.protocol, raw, material.paths["baseline"], output, "baseline"
        )
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure["completed_cases"] == (1 if kind == "call" else 0)
    assert "secret" not in (output / "failure.json").read_text()
    assert not (output / "raw-results.json").exists()
    if kind == "call":
        assert (output / "progress/000000.json").is_file()
        assert (output / "progress/000001.attempt.json").is_file()
        assert not (output / "progress/000002.attempt.json").exists()
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize(
    "kind", ["unavailable", "device", "dtype", "cache", "length", "softmax"]
)
def test_model_loader_enforces_runtime_and_memory_settings(
    helper, material, monkeypatch, kind
):
    if kind == "unavailable":
        material.torch.backends.mps.is_available = lambda: False
    elif kind in {"device", "dtype", "cache"}:
        original = FakeLM.__init__

        def changed(self, **kwargs):
            original(self, **kwargs)
            if kind == "cache":
                self.cache_hook.dbdict = {}
            else:
                self.model.parameters = lambda: [
                    SimpleNamespace(
                        device=SimpleNamespace(
                            type="cpu" if kind == "device" else "mps"
                        ),
                        dtype="float32" if kind == "dtype" else "float16",
                    )
                ]

        monkeypatch.setattr(FakeLM, "__init__", changed)
    else:
        monkeypatch.setattr(
            FakeLM, "max_length" if kind == "length" else "softmax_dtype", 1
        )
    with pytest.raises(ValueError):
        helper.load_model(material.paths["baseline"])


def test_read_json_duplicate_nonfinite_and_limits(helper, tmp_path):
    path = tmp_path / "input.json"
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', '"' + "x" * 10 + '"'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            helper.read_json(path, maximum=10 if raw.startswith('"') else 100)
    path.write_text('{"a":1}')
    assert helper.read_json(path)[0] == {"a": 1}


def test_network_and_offline_controls(helper, monkeypatch):
    hooks = []
    monkeypatch.setattr(helper.sys, "addaudithook", hooks.append)
    monkeypatch.setattr(helper.sys, "dont_write_bytecode", False)
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "CUDA_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "HF_DEACTIVATE_ASYNC_LOAD",
    ):
        monkeypatch.delenv(name, raising=False)
    helper.offline()
    assert helper.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert helper.os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"
    assert helper.CONFIGURATION["async_weight_loading"] is False
    assert helper.os.environ["HF_HUB_OFFLINE"] == "1"
    assert hooks == [helper.block_network]
    for event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        with pytest.raises(RuntimeError, match="network"):
            helper.block_network(event, ())
    helper.block_network("open", ())


@pytest.mark.parametrize("role", [None, "subject"])
def test_cli_preflight_role_selection(helper, material, tmp_path, monkeypatch, role):
    path = tmp_path / "protocol.json"
    raw = validate(helper, material.protocol)
    path.write_bytes(raw)
    monkeypatch.setattr(helper, "offline", lambda: None)
    args = [
        "--protocol",
        str(path),
        "--protocol-sha256",
        helper.bytes_digest(raw),
        "--baseline-model",
        str(material.paths["baseline"]),
        "--subject-model",
        str(material.paths["subject"]),
        "--output",
        str(tmp_path / "cli"),
        "--preflight",
    ]
    if role:
        args += ["--role", role]
    helper.main(args)
    assert (tmp_path / "cli/subject/preflight.json").is_file()
    assert (tmp_path / "cli/baseline").exists() is (role is None)
    assert not FakeLM.loads


def test_file_hash_rejects_nonregular_descriptor(helper, tmp_path, monkeypatch):
    import stat

    path = tmp_path / "file"
    path.write_bytes(b"authored")
    monkeypatch.setattr(
        helper.os, "fstat", lambda _: SimpleNamespace(st_mode=stat.S_IFDIR)
    )
    with pytest.raises(ValueError, match="regular file"):
        helper.file_fact(path, "file")


def test_changed_model_fails_before_manifest_or_calls(helper, material, tmp_path):
    raw = validate(helper, material.protocol)
    (material.paths["baseline"] / "model.safetensors").write_bytes(b"changed")
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError, match="independent byte pin"):
        helper.execute_role(
            material.protocol, raw, material.paths["baseline"], output, "baseline"
        )
    assert not output.exists()
    assert not FakeLM.loads and not FakeLM.calls
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
