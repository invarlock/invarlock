"""Worker admission/control tests with a fake LM; no model execution claims."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import socket
import stat
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location("live_worker_common", HERE / "common.py")
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)
SPEC = importlib.util.spec_from_file_location(
    "live_model_worker", HERE / "model_worker.py"
)
WORKER = importlib.util.module_from_spec(SPEC)
prior = sys.modules.get("common")
sys.modules["common"] = COMMON
try:
    SPEC.loader.exec_module(WORKER)
finally:
    if prior is None:
        sys.modules.pop("common")
    else:
        sys.modules["common"] = prior


def protocol():
    files = [
        {"path": name, "byte_size": 1, "sha256": "sha256:" + "a" * 64}
        for name in ("config.json", "model.safetensors", "tokenizer.json")
    ]
    model = {
        "files": files,
        "artifact_digest": COMMON.digest(files),
        "tokenizer_digest": COMMON.digest([files[-1]]),
    }
    models = {
        role: {**copy.deepcopy(model), "id": identity, "revision": revision}
        for role, (identity, revision) in WORKER.comparison_helpers().MODELS.items()
    }
    return {
        "models": models,
        "cases": [
            {"id": "case-1", "input": "Question", "expected": " Answer", "metadata": {}}
        ],
        "evaluators": ["deepeval", "ragas"],
        "versions": {name: COMMON.versions()[name] for name in ("deepeval", "ragas")},
        "configuration": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "max_length": 512,
            "max_new_tokens": 32,
            "seed": 0,
        },
        "limits": {"max_requests": 2, "max_seconds": 60},
    }


TOKENS = [
    {
        "context_token_ids": [1],
        "continuation_token_ids": [2, 3],
        "joined_token_ids": [1, 2, 3],
        "decoded_context": "Question",
        "decoded_joined": "Question Answer",
        "decoded_continuation": " Answer",
    }
]


class FakeLM:
    """Synthetic observations solely to test admission and preservation."""

    def __init__(self, output):
        self.output = output
        self.calls = []
        self.fail_likelihood = False

    def generate_until(self, requests, **kwargs):
        assert (
            len(list((self.output / "requests").glob("*.request.json")))
            > len(self.calls) // 2
        )
        self.calls.append(("generation", requests[0], kwargs))
        return ["Synthetic generated answer"]

    def loglikelihood(self, requests, **kwargs):
        self.calls.append(("likelihood", requests[0], kwargs))
        if self.fail_likelihood:
            raise RuntimeError("synthetic likelihood failure")
        return [(-2.5, False)]


def worker(tmp_path, *, clock=lambda: 0.0):
    value = protocol()
    model = FakeLM(tmp_path)
    instance = WORKER.Worker(
        value,
        "baseline",
        model,
        TOKENS,
        tmp_path,
        clock=clock,
        instance=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return instance, model


def request(worker, evaluator="deepeval"):
    return {
        "evaluator": evaluator,
        "case_id": "case-1",
        "protocol_digest": worker.protocol_digest,
    }


def test_durable_admission_fresh_calls_and_exact_measurements(tmp_path):
    instance, model = worker(tmp_path)
    first = instance.handle(request(instance))
    second = instance.handle(request(instance, "ragas"))
    assert [kind for kind, _, _ in model.calls] == ["generation", "likelihood"] * 2
    assert all(call.doc == protocol()["cases"][0] for _, call, _ in model.calls)
    result = first["result"]
    assert result["output"] == "Synthetic generated answer"
    facts = result["metadata"]["invarlock_likelihood"]
    assert facts["logprob_sum"] == -2.5 and facts["token_count"] == 2
    assert facts["utf8_byte_count"] == len(b" Answer")
    assert facts["source"] == {"name": "lm-eval", "version": "0.4.12"}
    assert facts["input_digest"] == COMMON.digest("Question")
    assert result["metadata"]["invarlock_model_execution"]["tokenization"] == TOKENS[0]
    assert first["request"] != second["request"]
    assert len(list((tmp_path / "requests").glob("*.response.json"))) == 2
    with pytest.raises(ValueError, match="already admitted"):
        instance.handle(request(instance))
    assert len(model.calls) == 4


def test_failed_likelihood_retains_actual_generation_without_inventing_facts(tmp_path):
    instance, model = worker(tmp_path)
    model.fail_likelihood = True
    result = instance.handle(request(instance))["result"]
    assert result["output"] == "Synthetic generated answer"
    assert result["error"] == "RuntimeError: synthetic likelihood failure"
    assert "invarlock_likelihood" not in result["metadata"]
    with pytest.raises(ValueError, match="already admitted"):
        instance.handle(request(instance))


@pytest.mark.parametrize("mutation", ["digest", "evaluator", "case", "extra"])
def test_unadmitted_requests_never_invoke_model(tmp_path, mutation):
    instance, model = worker(tmp_path)
    value = request(instance)
    if mutation == "digest":
        value["protocol_digest"] = "sha256:" + "b" * 64
    elif mutation == "evaluator":
        value["evaluator"] = "opik"
    elif mutation == "case":
        value["case_id"] = "unknown"
    else:
        value["input"] = "changed prompt"
    with pytest.raises(ValueError):
        instance.handle(value)
    assert not model.calls and not instance.seen


def test_caps_and_existing_admission_prevent_calls(tmp_path):
    now = [0.0]
    instance, model = worker(tmp_path, clock=lambda: now[0])
    now[0] = 60
    with pytest.raises(TimeoutError):
        instance.handle(request(instance))
    now[0] = 0
    instance.protocol["limits"]["max_requests"] = 0
    with pytest.raises(ValueError, match="request count"):
        instance.handle(request(instance))
    instance.protocol["limits"]["max_requests"] = 2
    value = request(instance)
    path = instance.admissions / (
        COMMON.digest(value).removeprefix("sha256:") + ".request.json"
    )
    COMMON.write(path, value)
    with pytest.raises(FileExistsError):
        instance.handle(value)
    assert not model.calls


def test_time_cap_between_measurements_prevents_second_inference(tmp_path):
    now = [0.0]
    instance, model = worker(tmp_path, clock=lambda: now[0])
    generate = model.generate_until

    def slow_generation(*args, **kwargs):
        result = generate(*args, **kwargs)
        now[0] = 61
        return result

    model.generate_until = slow_generation
    result = instance.handle(request(instance))["result"]
    assert [kind for kind, _, _ in model.calls] == ["generation"]
    assert result["output"] == "Synthetic generated answer"
    assert result["error"].startswith("TimeoutError:")
    assert "invarlock_likelihood" not in result["metadata"]


def test_protocol_requires_complete_schedule_and_immutable_pins():
    value = protocol()
    assert WORKER.validate_protocol(value, "baseline")["local_files_only"] is True
    mutations = [
        ("limits", "max_requests", 3),
        ("limits", "max_seconds", 0),
        ("configuration", "seed", True),
        ("configuration", "max_length", 2048),
        ("configuration", "trust_remote_code", True),
        ("versions", "ragas", "wrong"),
    ]
    for section, key, changed in mutations:
        altered = copy.deepcopy(value)
        altered[section][key] = changed
        with pytest.raises(ValueError):
            WORKER.validate_protocol(altered, "baseline")
    altered = copy.deepcopy(value)
    altered["models"]["baseline"]["files"][0]["path"] = "../config.json"
    with pytest.raises(ValueError):
        WORKER.validate_protocol(altered, "baseline")
    altered = copy.deepcopy(value)
    altered["models"]["baseline"]["id"] = "other/model"
    with pytest.raises(ValueError, match="Mistral model profile"):
        WORKER.validate_protocol(altered, "baseline")


def test_inventory_hashes_exact_local_files(tmp_path):
    model = copy.deepcopy(protocol()["models"]["baseline"])
    for fact in model["files"]:
        path = tmp_path / fact["path"]
        path.write_bytes(
            b'{"model_type":"mistral","architectures":["MistralForCausalLM"]}'
            if fact["path"] == "config.json"
            else b"synthetic"
        )
        fact.update(
            byte_size=path.stat().st_size,
            sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    assert WORKER.inventory(tmp_path, model) == model["files"]
    config_path = tmp_path / "config.json"
    original_config = config_path.read_bytes()
    original_fact = copy.deepcopy(model["files"][0])
    config_path.write_bytes(
        b'{"model_type":"gpt2","architectures":["GPT2LMHeadModel"]}'
    )
    model["files"][0].update(
        byte_size=config_path.stat().st_size,
        sha256="sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="Mistral architecture"):
        WORKER.inventory(tmp_path, model)
    config_path.write_bytes(original_config)
    model["files"][0] = original_fact
    (tmp_path / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="independent pin"):
        WORKER.inventory(tmp_path, model)
    (tmp_path / "extra.py").write_text("unexpected")
    with pytest.raises(ValueError, match="exactly"):
        WORKER.inventory(tmp_path, model)


def test_private_socket_modes_and_no_overwrite(tmp_path):
    # Keep below macOS's short Unix pathname limit, independently of pytest cwd.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="worker-") as directory:
        parent = Path(directory)
        parent.chmod(0o700)
        path = parent / "model.sock"
        with WORKER.private_socket(path) as channel:
            assert channel.family == socket.AF_UNIX
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            with pytest.raises(ValueError, match="already exists"):
                WORKER.private_socket(path)
        path.unlink()
        parent.chmod(0o755)
        with pytest.raises(ValueError, match="0700"):
            WORKER.private_socket(path)


@pytest.mark.parametrize("remote_code", [False, True])
def test_inventory_rejects_pinned_tokenizer_remote_code(tmp_path, remote_code):
    contents = {
        "config.json": {
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
        },
        "tokenizer_config.json": (
            {"auto_map": {"AutoTokenizer": "custom.Tokenizer"}}
            if remote_code
            else {"tokenizer_class": "LlamaTokenizer"}
        ),
    }
    facts = []
    for name, value in contents.items():
        raw = COMMON.encoded(value)
        (tmp_path / name).write_bytes(raw)
        facts.append(
            {
                "path": name,
                "byte_size": len(raw),
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            }
        )
    # A matching file digest authenticates bytes, but does not permit custom code.
    if remote_code:
        with pytest.raises(ValueError, match="tokenizer remote code"):
            WORKER.inventory(tmp_path, {"files": facts})
    else:
        assert WORKER.inventory(tmp_path, {"files": facts}) == facts


def test_private_socket_creates_private_parent_and_rejects_oversized_path():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="worker-new-") as directory:
        parent = Path(directory) / "new"
        path = parent / "model.sock"
        with WORKER.private_socket(path):
            assert stat.S_IMODE(parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        path.unlink()
        oversized = Path(directory) / ("x" * 101) / "model.sock"
        with pytest.raises(ValueError, match="portable byte limit"):
            WORKER.private_socket(oversized)
        assert not oversized.parent.exists()


@pytest.mark.parametrize("failure", ["bind", "listen"])
def test_private_socket_closes_channel_on_setup_failure(monkeypatch, failure):
    import tempfile

    calls = []

    def bind(path):
        calls.append("bind")
        if failure == "bind":
            raise OSError("synthetic bind failure")
        Path(path).touch()

    def listen(backlog):
        assert backlog == 1
        calls.append("listen")
        raise OSError("synthetic listen failure")

    channel = SimpleNamespace(
        bind=bind, listen=listen, close=lambda: calls.append("close")
    )
    monkeypatch.setattr(WORKER.socket, "socket", lambda *args: channel)
    with tempfile.TemporaryDirectory(prefix="worker-fail-") as directory:
        with pytest.raises(OSError, match=f"synthetic {failure} failure"):
            WORKER.private_socket(Path(directory) / "model.sock")
    assert calls == (
        ["bind", "close"] if failure == "bind" else ["bind", "listen", "close"]
    )


def test_preflight_requires_explicit_mode_and_never_loads_model(tmp_path, monkeypatch):
    value = protocol()
    protocol_path = tmp_path / "protocol.json"
    COMMON.write(protocol_path, value)
    output = tmp_path / "preflight"
    args = [
        "--protocol",
        str(protocol_path),
        "--protocol-sha256",
        COMMON.digest(value),
        "--role",
        "baseline",
        "--model-dir",
        str(tmp_path / "model"),
        "--output",
        str(output),
        "--socket",
        str(tmp_path / "socket"),
    ]
    with pytest.raises(SystemExit):
        WORKER.main(args)
    monkeypatch.setattr(WORKER, "offline", lambda: None)

    def prepared(*args):
        output.mkdir()
        return WORKER.validate_protocol(value, "baseline"), TOKENS

    monkeypatch.setattr(WORKER, "prepare", prepared)
    monkeypatch.setattr(
        WORKER, "load_model", lambda *args: pytest.fail("preflight cannot load weights")
    )
    monkeypatch.setattr(
        WORKER, "serve", lambda *args: pytest.fail("preflight cannot serve inference")
    )
    assert WORKER.main([*args, "--preflight"]) == 0
    assert COMMON.read(output / "preflight.json") == {
        "status": "tokenization_only",
        "model_loaded": False,
        "inference_calls": 0,
        "case_count": 1,
    }


def test_socket_serves_frozen_requests_and_rejects_duplicate_pairs(
    tmp_path, monkeypatch
):
    import tempfile

    instance, model = worker(tmp_path, clock=time.monotonic)
    ready = threading.Event()
    original_write = WORKER.durable_write

    def written(path, value):
        original_write(path, value)
        if Path(path).name == "ready.json":
            ready.set()

    monkeypatch.setattr(WORKER, "durable_write", written)
    with tempfile.TemporaryDirectory(prefix="worker-wire-") as directory:
        path = Path(directory) / "model.sock"
        failures = []

        def serve():
            try:
                WORKER.serve(instance, path)
            except Exception as exc:
                failures.append(exc)
                ready.set()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        assert ready.wait(timeout=5)
        assert failures == []

        def call(value):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                channel.settimeout(5)
                channel.connect(str(path))
                channel.sendall(COMMON.encoded(value))
                with channel.makefile("rb") as stream:
                    return json.loads(stream.readline(COMMON.MAX_MESSAGE))

        assert "rejected" in call({"unknown": True})
        assert (
            call(request(instance))["result"]["output"] == "Synthetic generated answer"
        )
        assert "already admitted" in call(request(instance))["rejected"]
        assert (
            call(request(instance, "ragas"))["result"]["output"]
            == "Synthetic generated answer"
        )
        thread.join(timeout=5)
        assert not thread.is_alive() and failures == []
        assert not path.exists()
        assert len(model.calls) == 4


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("configuration", "device", "remote"),
        ("configuration", "dtype", "int8"),
        ("configuration", "unknown_option", True),
        ("configuration", "timeout_seconds", -1),
        ("models", "subject", {}),
        ("model", "revision", "main"),
        ("model", "files", []),
        ("model", "artifact_digest", "wrong"),
        ("file", "byte_size", -1),
        ("file", "sha256", "unpinned"),
    ],
)
def test_worker_rejects_unsupported_or_unpinned_execution(section, key, value):
    candidate = protocol()
    target = (
        candidate["models"]["baseline"]
        if section == "model"
        else (
            candidate["models"]["baseline"]["files"][0]
            if section == "file"
            else candidate[section]
        )
    )
    target[key] = value
    with pytest.raises(ValueError):
        WORKER.validate_protocol(candidate, "baseline")


def test_worker_requires_complete_configuration_roles_and_source_pins():
    for candidate, role in [(protocol(), "unplanned"), ({}, "baseline")]:
        with pytest.raises((ValueError, KeyError)):
            WORKER.validate_protocol(candidate, role)
    for field, changed in [
        ("configuration", {}),
        ("models", {}),
        ("source_implementations", {}),
    ]:
        candidate = protocol()
        candidate[field] = changed
        with pytest.raises(ValueError):
            WORKER.validate_protocol(candidate, "baseline")
    candidate = protocol()
    candidate["models"]["baseline"]["files"][0] = {"path": "config.json"}
    with pytest.raises(ValueError):
        WORKER.validate_protocol(candidate, "baseline")
    candidate = protocol()
    candidate["configuration"].update(
        local_files_only=True, trust_remote_code=False, timeout_seconds=3600
    )
    assert WORKER.validate_protocol(candidate, "baseline")["timeout_seconds"] == 60


def test_offline_policy_blocks_internet_and_allows_private_transport(monkeypatch):
    hooks = []
    monkeypatch.setattr(sys, "addaudithook", hooks.append)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "HF_DEACTIVATE_ASYNC_LOAD",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        monkeypatch.setenv(key, "test-before")
    WORKER.offline()
    hook = hooks[0]
    for event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"):
        with pytest.raises(RuntimeError, match="network access"):
            hook(
                event,
                (SimpleNamespace(family=socket.AF_INET), ("example.invalid", 443)),
            )
        hook(event, (SimpleNamespace(family=socket.AF_UNIX), "/tmp/private.sock"))
    with pytest.raises(RuntimeError):
        hook("socket.getaddrinfo", ())
    hook("open", ())
    assert sys.dont_write_bytecode is True


def test_cuda_workspace_is_fixed_before_runtime_import(monkeypatch, tmp_path):
    import builtins
    import os

    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    monkeypatch.setattr(sys, "addaudithook", lambda hook: None)
    model, calls = _runtime(monkeypatch, "cuda")
    imported = []
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "torch":
            imported.append(name)
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    WORKER.offline()
    value = protocol()
    value["configuration"]["device"] = "cuda"
    assert (
        WORKER.load_model(tmp_path, WORKER.validate_protocol(value, "baseline"))
        is model
    )
    assert imported and ("deterministic", True) in calls


def _runtime(monkeypatch, device="cpu", available=True, change=None):
    calls = []
    torch = SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: available)),
        cuda=SimpleNamespace(is_available=lambda: available),
        manual_seed=lambda n: calls.append(("seed", n)),
        use_deterministic_algorithms=lambda enabled: calls.append(
            ("deterministic", enabled)
        ),
        set_num_threads=lambda n: calls.append(("threads", n)),
        mps=SimpleNamespace(
            set_per_process_memory_fraction=lambda n: calls.append(("memory", n))
        ),
    )
    model = SimpleNamespace(
        backend="causal",
        max_length=512,
        logits_cache=False,
        cache_hook=SimpleNamespace(dbdict=None),
        softmax_dtype="float32",
        enable_thinking=False,
        model=SimpleNamespace(
            parameters=lambda: [
                SimpleNamespace(device=SimpleNamespace(type=device), dtype="float32")
            ]
        ),
    )
    if change is not None:
        setattr(model, *change)

    def constructor(**kwargs):
        calls.append(("load", kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "lm_eval.models.huggingface", SimpleNamespace(HFLM=constructor)
    )
    return model, calls


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_model_loader_applies_offline_precision_and_cache_settings(
    monkeypatch, tmp_path, device
):
    value = protocol()
    value["configuration"]["device"] = device
    configuration = WORKER.validate_protocol(value, "baseline")
    model, calls = _runtime(monkeypatch, device)
    assert WORKER.load_model(tmp_path, configuration) is model
    loaded = next(data for kind, data in calls if kind == "load")
    assert loaded["pretrained"] == str(tmp_path)
    assert loaded["local_files_only"] is True and loaded["trust_remote_code"] is False
    assert loaded["logits_cache"] is False and loaded["truncation"] is False
    assert loaded["batch_size"] == 1 and loaded["dtype"] == "float32"
    assert ("seed", 0) in calls and ("deterministic", True) in calls


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_unavailable_device_never_constructs_model(monkeypatch, tmp_path, device):
    value = protocol()
    value["configuration"]["device"] = device
    _, calls = _runtime(monkeypatch, device, available=False)
    with pytest.raises(ValueError, match="unavailable"):
        WORKER.load_model(tmp_path, WORKER.validate_protocol(value, "baseline"))
    assert not calls


@pytest.mark.parametrize(
    "change",
    [
        ("logits_cache", True),
        ("cache_hook", SimpleNamespace(dbdict={})),
        ("softmax_dtype", "float16"),
        ("enable_thinking", True),
    ],
)
def test_loader_rejects_runtime_configuration_drift(monkeypatch, tmp_path, change):
    _runtime(monkeypatch, change=change)
    with pytest.raises(ValueError, match="differs"):
        WORKER.load_model(tmp_path, WORKER.validate_protocol(protocol(), "baseline"))


def test_tokenization_guard_prevents_generation_truncation(monkeypatch):
    helpers = SimpleNamespace(
        CONFIGURATION={}, tokenize=lambda model, cases: copy.deepcopy(TOKENS)
    )
    monkeypatch.setattr(WORKER, "comparison_helpers", lambda: helpers)
    assert (
        WORKER.tokenizations(object(), [], {"max_length": 1024, "max_new_tokens": 32})
        == TOKENS
    )
    assert helpers.CONFIGURATION["max_length"] == 1024
    helpers.tokenize = lambda model, cases: [{"context_token_ids": [0] * 500}]
    with pytest.raises(ValueError, match="truncate"):
        WORKER.tokenizations(object(), [], {"max_length": 512, "max_new_tokens": 32})


def test_preparation_retains_inventory_tokenization_and_dependency_facts(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    helpers = WORKER.comparison_helpers()
    tokenizer = object()
    monkeypatch.setattr(helpers, "tokenizer_only", lambda path: tokenizer)
    monkeypatch.setattr(
        helpers,
        "dependencies",
        lambda declaration: ({"lm-eval": "synthetic-test"}, [{"source": declaration}]),
    )
    monkeypatch.setattr(WORKER, "comparison_helpers", lambda: helpers)
    observed = [{"path": "synthetic-file"}]
    monkeypatch.setattr(WORKER, "inventory", lambda *args: observed)
    monkeypatch.setattr(
        WORKER,
        "tokenizations",
        lambda actual, *args: (
            TOKENS if actual is tokenizer else pytest.fail("wrong tokenizer")
        ),
    )
    monkeypatch.setattr(
        WORKER,
        "load_model",
        lambda *args: pytest.fail("preparation cannot load weights"),
    )
    output = tmp_path / "prepared"
    configuration, tokens = WORKER.prepare(protocol(), "baseline", tmp_path, output)
    manifest = COMMON.read(output / "manifest.json")
    assert tokens == TOKENS and COMMON.read(output / "tokenizations.json") == TOKENS
    assert manifest["files"] == observed
    assert manifest["effective_configuration"] == configuration
    assert len(manifest["capture_implementations"]) == 3
    assert manifest["packages"] == {"lm-eval": "synthetic-test"}
    assert manifest["runtime_environment"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


@pytest.mark.parametrize("failure", [None, "tokenization", "deadline", "prepare"])
def test_worker_execute_cli_loads_once_or_preserves_failure(
    monkeypatch, tmp_path, failure
):
    value = protocol()
    source = tmp_path / "protocol.json"
    COMMON.write(source, value)
    output = tmp_path / "execution"
    args = [
        "--protocol",
        str(source),
        "--protocol-sha256",
        COMMON.digest(value),
        "--role",
        "baseline",
        "--model-dir",
        str(tmp_path),
        "--output",
        str(output),
        "--socket",
        "unused",
        "--execute",
    ]
    monkeypatch.setattr(WORKER, "offline", lambda: None)
    handlers = []
    monkeypatch.setattr(
        WORKER.signal, "signal", lambda signal, handler: handlers.append(handler)
    )
    monkeypatch.setattr(WORKER.signal, "setitimer", lambda *args: None)

    def prepared(*args):
        output.mkdir()
        if failure == "prepare":
            raise RuntimeError("synthetic preparation failure")
        if failure == "deadline":
            handlers[0](None, None)
        return WORKER.validate_protocol(value, "baseline"), TOKENS

    monkeypatch.setattr(WORKER, "prepare", prepared)
    model = FakeLM(output)
    loads = []
    monkeypatch.setattr(WORKER, "load_model", lambda *args: loads.append(args) or model)
    monkeypatch.setattr(
        WORKER,
        "tokenizations",
        lambda *args: [] if failure == "tokenization" else TOKENS,
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval.api.instance",
        SimpleNamespace(Instance=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setattr(
        WORKER,
        "serve",
        lambda worker, path: [
            worker.handle(request(worker, name)) for name in value["evaluators"]
        ],
    )
    monkeypatch.setattr(
        WORKER, "inventory", lambda *args: value["models"]["baseline"]["files"]
    )
    if failure is None:
        assert WORKER.main(args) == 0
        assert len(loads) == 1 and len(model.calls) == 4
        assert COMMON.read(output / "complete.json")["admitted_requests"] == 2
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            WORKER.main(args)
        assert COMMON.read(output / "failure.json")["status"] == "failed"
        assert not (output / "complete.json").exists()
        assert not model.calls


def test_worker_cli_refuses_changed_protocol_or_existing_output(monkeypatch, tmp_path):
    value = protocol()
    source = tmp_path / "protocol.json"
    COMMON.write(source, value)
    args = [
        "--protocol",
        str(source),
        "--protocol-sha256",
        "wrong",
        "--role",
        "baseline",
        "--model-dir",
        str(tmp_path),
        "--output",
        str(tmp_path),
        "--socket",
        "unused",
        "--preflight",
    ]
    monkeypatch.setattr(
        WORKER,
        "offline",
        lambda: pytest.fail("unadmitted CLI cannot initialize runtime"),
    )
    with pytest.raises(ValueError, match="approved digest"):
        WORKER.main(args)
    args[3] = COMMON.digest(value)
    with pytest.raises(FileExistsError, match="never resumed"):
        WORKER.main(args)


@pytest.mark.parametrize(
    "method,returned",
    [("generate_until", []), ("generate_until", [None]), ("loglikelihood", [])],
)
def test_native_result_cardinality_failures_remain_failed_admitted_attempts(
    tmp_path, method, returned
):
    instance, model = worker(tmp_path)
    setattr(model, method, lambda *args, **kwargs: returned)
    result = instance.handle(request(instance))["result"]
    assert result["error"].startswith("ValueError:")
    assert "invarlock_likelihood" not in result["metadata"]
    with pytest.raises(ValueError, match="already admitted"):
        instance.handle(request(instance))
