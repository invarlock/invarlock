from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
from invarlock_addins.tensorrt_llm import runner


def _runtime_request(tmp_path: Path) -> tuple[bytes, Path]:
    root = tmp_path.resolve() / "run"
    engine = root / "engine"
    engine.mkdir(parents=True)
    engine.joinpath("config.json").write_text(
        json.dumps(
            {
                "build_config": {
                    "max_batch_size": 8,
                    "max_input_len": 64,
                    "max_seq_len": 96,
                },
                "pretrained_config": {
                    "architecture": "LlamaForCausalLM",
                    "dtype": "float16",
                    "mapping": {
                        "cp_size": 1,
                        "pp_size": 1,
                        "tp_size": 1,
                        "world_size": 1,
                    },
                    "num_hidden_layers": 2,
                },
                "version": "1.0.0",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    engine.joinpath("rank0.engine").write_bytes(b"authenticated-engine")
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text(
        json.dumps(
            {
                "add_special_tokens": False,
                "clean_up_tokenization_spaces": False,
                "eos_token_id": 1,
                "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
                "pad_token_id": 0,
                "skip_special_tokens": True,
                "tokenizer_json": {
                    "model": {"type": "test"},
                    "version": "1.0",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    request = {
        "engine_bundle": str(engine),
        "format_version": "invarlock/tensorrt-llm-runner-request-v1",
        "input_text": "hello world",
        "protocol_version": "invarlock/tensorrt-llm-runner-v1",
        "settings": {
            "allow_network": False,
            "batch_size": 4,
            "context_length": 64,
            "max_output_tokens": 16,
            "seed": 17,
            "timeout_seconds": 30,
        },
        "tokenizer_contract": str(tokenizer),
    }
    return (
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode(),
        engine,
    )


def _runtime_batch_request(
    tmp_path: Path,
    *,
    records: list[dict[str, object]] | None = None,
) -> tuple[bytes, Path]:
    payload, engine = _runtime_request(tmp_path)
    request = json.loads(payload)
    del request["input_text"]
    request["format_version"] = "invarlock/tensorrt-llm-runner-batch-request-v1"
    request["records"] = records or [
        {"input_text": "hello world", "record_id": "record/0"},
        {"input_text": "goodbye world", "record_id": "record/1"},
    ]
    return (
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode(),
        engine,
    )


class _RawTokenizer:
    def id_to_token(self, token_id: int) -> str | None:
        return {0: "<pad>", 1: "</s>"}.get(token_id)


class _FastTokenizer:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        self.kwargs = kwargs
        self.eos_token_id = 1
        self.pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


class _FakeSamplingParams:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


class _FakeLLM:
    calls: list[dict[str, object]] = []
    generate_calls: list[tuple[object, dict[str, object]]] = []
    shutdown_calls = 0

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def generate(self, prompt: object, **kwargs: object) -> object:
        self.generate_calls.append((prompt, kwargs))
        if isinstance(prompt, list):
            return [
                SimpleNamespace(
                    finished=True,
                    outputs=[SimpleNamespace(text="OUT:" + str(item))],
                    prompt=item,
                    prompt_token_ids=list(range(len(str(item).split()))),
                )
                for item in prompt
            ]
        return SimpleNamespace(
            finished=True,
            outputs=[SimpleNamespace(text="OUT:" + str(prompt))],
            prompt=prompt,
            prompt_token_ids=list(range(len(str(prompt).split()))),
        )

    def shutdown(self) -> None:
        type(self).shutdown_calls += 1


@pytest.fixture
def fake_backend() -> runner._Backend:  # noqa: SLF001
    _FakeLLM.calls.clear()
    _FakeLLM.generate_calls.clear()
    _FakeLLM.shutdown_calls = 0
    _FakeSamplingParams.calls.clear()
    _FastTokenizer.calls.clear()
    return runner._Backend(  # noqa: SLF001
        llm=_FakeLLM,
        sampling_params=_FakeSamplingParams,
        fast_tokenizer=_FastTokenizer,
        raw_tokenizer_from_str=lambda _value: _RawTokenizer(),
    )


def test_runner_executes_pinned_single_rank_hlapi_contract(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, engine = _runtime_request(tmp_path)
    request = runner._parse_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    output = runner._execute_request(request, backend=fake_backend)  # noqa: SLF001

    assert output == "OUT:hello world"
    assert _FakeLLM.calls == [
        {
            "model": engine,
            "tokenizer": ANY,
            "tokenizer_mode": "auto",
            "skip_tokenizer_init": False,
            "trust_remote_code": False,
            "tensor_parallel_size": 1,
        }
    ]
    assert _FakeSamplingParams.calls == [
        {
            "add_special_tokens": False,
            "best_of": 1,
            "detokenize": True,
            "end_id": 1,
            "exclude_input_from_output": True,
            "max_tokens": 16,
            "n": 1,
            "pad_id": 0,
            "seed": 17,
            "skip_special_tokens": True,
            "temperature": 0.0,
            "top_k": 1,
            "use_beam_search": False,
        }
    ]
    assert _FakeLLM.generate_calls[0][0] == "hello world"
    assert (
        _FakeLLM.generate_calls[0][1]["sampling_params"].kwargs
        == (_FakeSamplingParams.calls[0])
    )
    assert _FakeLLM.generate_calls[0][1]["use_tqdm"] is False
    assert _FakeLLM.shutdown_calls == 1


def test_runner_executes_one_ordered_batch_with_one_loaded_engine(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, engine = _runtime_batch_request(tmp_path)
    request = runner._parse_batch_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    outputs = runner._execute_batch_request(  # noqa: SLF001
        request, backend=fake_backend
    )

    assert outputs == (
        ("record/0", "OUT:hello world"),
        ("record/1", "OUT:goodbye world"),
    )
    assert len(_FakeLLM.calls) == 1
    assert _FakeLLM.calls[0]["model"] == engine
    assert len(_FastTokenizer.calls) == 1
    assert len(_FakeSamplingParams.calls) == 1
    assert _FakeLLM.generate_calls == [
        (
            ["hello world", "goodbye world"],
            {"sampling_params": ANY, "use_tqdm": False},
        )
    ]
    assert _FakeLLM.shutdown_calls == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda request: request.update({"extra": True}), "fields are not closed"),
        (
            lambda request: request["records"][0].update({"extra": True}),
            "record fields are not closed",
        ),
        (
            lambda request: request["records"].append(dict(request["records"][0])),
            "record IDs must be unique",
        ),
        (lambda request: request.update({"records": []}), "records count"),
    ],
)
def test_runner_batch_request_schema_is_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:  # noqa: ANN001
    payload, _engine = _runtime_batch_request(tmp_path)
    request = json.loads(payload)
    mutation(request)

    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._parse_batch_request(  # noqa: SLF001
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        )


def test_runner_batch_rejects_prompt_beyond_authenticated_context_before_engine_load(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _engine = _runtime_batch_request(
        tmp_path,
        records=[{"input_text": " ".join(["token"] * 65), "record_id": "too-long"}],
    )
    request = runner._parse_batch_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    with pytest.raises(
        runner.TensorRTLLMRunnerError,
        match="prompt exceeds the authenticated context length",
    ):
        runner._execute_batch_request(request, backend=fake_backend)  # noqa: SLF001

    assert _FakeLLM.calls == []


def test_runner_batch_rejects_backend_count_mismatch(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _engine = _runtime_batch_request(tmp_path)
    request = runner._parse_batch_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    def one_output(_self, _prompts: object, **_kwargs: object) -> object:
        return [SimpleNamespace(finished=True, outputs=[SimpleNamespace(text="one")])]

    monkeypatch.setattr(_FakeLLM, "generate", one_output)

    with pytest.raises(
        runner.TensorRTLLMRunnerError,
        match="output count does not match",
    ):
        runner._execute_batch_request(request, backend=fake_backend)  # noqa: SLF001


def test_runner_batch_rejects_backend_output_reordering(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _engine = _runtime_batch_request(tmp_path)
    request = runner._parse_batch_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    def swapped_outputs(_self, prompts: object, **_kwargs: object) -> object:
        assert isinstance(prompts, list)
        return [
            SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(text="OUT:" + prompt)],
                prompt=prompt,
                prompt_token_ids=list(range(len(prompt.split()))),
            )
            for prompt in reversed(prompts)
        ]

    monkeypatch.setattr(_FakeLLM, "generate", swapped_outputs)

    with pytest.raises(
        runner.TensorRTLLMRunnerError,
        match="prompt order does not match",
    ):
        runner._execute_batch_request(request, backend=fake_backend)  # noqa: SLF001


def test_runner_batch_rejects_backend_prompt_token_mismatch(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _engine = _runtime_batch_request(
        tmp_path,
        records=[{"input_text": "hello world", "record_id": "record/0"}],
    )
    request = runner._parse_batch_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    def wrong_tokens(_self, _prompts: object, **_kwargs: object) -> object:
        return [
            SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(text="OUT:hello world")],
                prompt="hello world",
                prompt_token_ids=[99],
            )
        ]

    monkeypatch.setattr(_FakeLLM, "generate", wrong_tokens)

    with pytest.raises(
        runner.TensorRTLLMRunnerError,
        match="prompt tokens do not match",
    ):
        runner._execute_batch_request(request, backend=fake_backend)  # noqa: SLF001


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            SimpleNamespace(finished=False, outputs=[SimpleNamespace(text="x")]),
            "did not finish",
        ),
        (
            SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(text=7)],
                prompt="hello",
                prompt_token_ids=[0],
            ),
            "valid user-visible text",
        ),
        (
            SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(text="x" * ((2 * 1024 * 1024) + 1))],
                prompt="hello",
                prompt_token_ids=[0],
            ),
            "byte limit",
        ),
    ],
)
def test_runner_batch_rejects_invalid_or_oversized_outputs(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
    output: object,
    message: str,
) -> None:
    payload, _engine = _runtime_batch_request(
        tmp_path,
        records=[{"input_text": "hello", "record_id": "record/0"}],
    )
    request = runner._parse_batch_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    monkeypatch.setattr(_FakeLLM, "generate", lambda *_args, **_kwargs: [output])

    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._execute_batch_request(request, backend=fake_backend)  # noqa: SLF001


def test_runner_rejects_non_utf8_user_visible_output(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _engine = _runtime_request(tmp_path)
    request = runner._parse_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    def surrogate_output(_self, _prompt: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            finished=True,
            outputs=[SimpleNamespace(text="\ud800")],
        )

    monkeypatch.setattr(_FakeLLM, "generate", surrogate_output)

    with pytest.raises(
        runner.TensorRTLLMRunnerError,
        match="valid user-visible text",
    ):
        runner._execute_request(request, backend=fake_backend)  # noqa: SLF001


def test_runner_rejects_noncanonical_json_request(tmp_path: Path) -> None:
    payload, _engine = _runtime_request(tmp_path)
    noncanonical = json.dumps(json.loads(payload), indent=2).encode()

    with pytest.raises(runner.TensorRTLLMRunnerError, match="not canonical JSON"):
        runner._parse_request(noncanonical)  # noqa: SLF001


def test_runner_fails_closed_without_deterministic_backend_mode(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _engine = _runtime_request(tmp_path)
    request = runner._parse_request(payload)  # noqa: SLF001
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.delenv("FORCE_DETERMINISTIC", raising=False)

    with pytest.raises(runner.TensorRTLLMRunnerError, match="FORCE_DETERMINISTIC=1"):
        runner._execute_request(request, backend=fake_backend)  # noqa: SLF001

    assert _FakeLLM.calls == []


def test_runner_info_is_derived_from_observed_backend_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setattr(runner, "_observed_backend_build_sha256", lambda: "a" * 64)
    monkeypatch.setattr(
        runner,
        "_observe_cuda_device",
        lambda: runner._ObservedDevice(  # noqa: SLF001
            device_name="Observed NVIDIA H200",
            compute_capability="9.0",
            driver_version="570.00",
            cuda_runtime_version="12.8",
        ),
    )

    assert runner._info_payload() == {  # noqa: SLF001
        "backend_build_sha256": "a" * 64,
        "backend_name": "TensorRT-LLM",
        "backend_version": "1.2.1",
        "cuda_compute_capability": "9.0",
        "cuda_device_name": "Observed NVIDIA H200",
        "cuda_driver_version": "570.00",
        "cuda_runtime_version": "12.8",
        "device_kind": "cuda",
        "format_version": "invarlock/tensorrt-llm-runner-info-v1",
        "protocol_version": "invarlock/tensorrt-llm-runner-v1",
    }


def test_cuda_runtime_version_comes_from_live_cudart_api() -> None:
    def get_version(pointer: object) -> int:
        pointer._obj.value = 12_080  # type: ignore[attr-defined]
        return 0

    library = SimpleNamespace(cudaRuntimeGetVersion=get_version)

    assert (
        runner._read_cuda_runtime_version(  # noqa: SLF001
            library_loader=lambda name: (
                library if name == "libcudart.so" else pytest.fail(name)
            )
        )
        == "12.8"
    )


def test_cuda_runtime_version_probe_fails_closed() -> None:
    def get_version(_pointer: object) -> int:
        return 35

    library = SimpleNamespace(cudaRuntimeGetVersion=get_version)

    with pytest.raises(runner.TensorRTLLMRunnerError, match="probe failed"):
        runner._read_cuda_runtime_version(  # noqa: SLF001
            library_loader=lambda _name: library
        )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  "
            "550.54.15  Tue Mar  5 15:41:53 UTC 2024\n",
            "550.54.15",
        ),
        (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
            "580.126.09  Release Build  "
            "(dvs-builder@U22-I3-AM02-24-3)  Wed Jan  7 22:51:36 UTC 2026\n"
            "GCC version: gcc version 13.3.0\n",
            "580.126.09",
        ),
    ],
)
def test_driver_version_accepts_canonical_nvidia_layouts(
    tmp_path: Path, payload: str, expected: str
) -> None:
    version = tmp_path / "version"
    version.write_text(payload, encoding="ascii")

    assert runner._read_driver_version(version_path=version) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    "payload",
    [
        "Kernel Module 580.126.09\n",
        "NVRM version: vendor Kernel Module 580.126.09\n",
        "NVRM version: NVIDIA UNIX Kernel Module  580.126.09  "
        "Wed Jan  7 22:51:36 UTC 2026\n",
        "NVRM version: NVIDIA UNIX Open Kernel Module 580.126.09\n",
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 invalid\n",
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
        "580.126.09 forged suffix\n",
    ],
)
def test_driver_version_rejects_noncanonical_text(tmp_path: Path, payload: str) -> None:
    version = tmp_path / "version"
    version.write_text(payload, encoding="ascii")

    with pytest.raises(runner.TensorRTLLMRunnerError, match="not canonical"):
        runner._read_driver_version(version_path=version)  # noqa: SLF001


def test_driver_version_rejects_unavailable_or_non_ascii_sources(
    tmp_path: Path,
) -> None:
    version = tmp_path / "version"
    with pytest.raises(runner.TensorRTLLMRunnerError, match="unavailable"):
        runner._read_driver_version(version_path=version)  # noqa: SLF001
    version.write_bytes(b"NVRM version: \xff")
    with pytest.raises(runner.TensorRTLLMRunnerError, match="unavailable"):
        runner._read_driver_version(version_path=version)  # noqa: SLF001


def test_runner_backend_build_identity_changes_with_live_module_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [
        *runner._CRITICAL_BACKEND_FILES,  # noqa: SLF001
        "tensorrt_llm/bindings.cpython-312-x86_64-linux-gnu.so",
    ]
    for index, name in enumerate(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"backend-file-{index}".encode())

    class _Distribution:
        files = names

        @staticmethod
        def locate_file(name: object) -> Path:
            return tmp_path / str(name)

    monkeypatch.setattr(runner, "_require_backend_version", lambda: None)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda _name: _Distribution(),
    )
    initial = runner._observed_backend_build_sha256()  # noqa: SLF001
    tmp_path.joinpath(names[-1]).write_bytes(b"changed-native-extension")

    assert initial != runner._observed_backend_build_sha256()  # noqa: SLF001


def test_runner_import_does_not_import_gpu_or_model_frameworks() -> None:
    root = Path.cwd()
    code = """
import json
import sys
before = set(sys.modules)
import invarlock_addins.tensorrt_llm.runner  # noqa: F401
loaded = set(sys.modules) - before
blocked = sorted(name for name in loaded if name == 'torch' or name == 'transformers' or name == 'tensorrt_llm' or name.startswith(('torch.', 'transformers.', 'tensorrt_llm.')))
print(json.dumps(blocked))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(root / "src"),
                str(root / "addins" / "tensorrt_llm" / "src"),
            )
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == []


def test_runner_loads_explicit_tensorrt_engine_hlapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_llm = object()
    engine_llm = object()
    sampling = object()
    fast = object()
    from_str = object()
    modules = {
        "tensorrt_llm": SimpleNamespace(LLM=root_llm, SamplingParams=sampling),
        "tensorrt_llm._tensorrt_engine": SimpleNamespace(LLM=engine_llm),
        "transformers": SimpleNamespace(PreTrainedTokenizerFast=fast),
        "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)),
    }
    monkeypatch.setattr(runner, "_require_backend_version", lambda: None)
    monkeypatch.setattr(runner.importlib, "import_module", modules.__getitem__)
    # Plain object instances are not callable; functions make the distinction clear.
    modules["tensorrt_llm"].SamplingParams = lambda **_kwargs: None
    modules["tensorrt_llm._tensorrt_engine"].LLM = lambda **_kwargs: None
    modules["transformers"].PreTrainedTokenizerFast = lambda **_kwargs: None
    modules["tokenizers"].Tokenizer.from_str = lambda _value: None
    loaded = runner._load_backend()  # noqa: SLF001

    assert loaded.llm is modules["tensorrt_llm._tensorrt_engine"].LLM
    assert loaded.llm is not root_llm


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda request: request["settings"].update({"allow_network": True}),
            "network access",
        ),
        (
            lambda request: request["settings"].update({"extra": 1}),
            "not closed",
        ),
        (
            lambda request: request["settings"].update({"context_length": 65}),
            "context_length exceeds",
        ),
    ],
)
def test_runner_rejects_invalid_closed_requests(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:  # noqa: ANN001
    payload, _engine = _runtime_request(tmp_path)
    request = json.loads(payload)
    mutation(request)

    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._parse_request(  # noqa: SLF001
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        )


def test_runner_rejects_multi_rank_engine(tmp_path: Path) -> None:
    payload, engine = _runtime_request(tmp_path)
    engine.joinpath("rank1.engine").write_bytes(b"second-rank")

    with pytest.raises(runner.TensorRTLLMRunnerError, match="single-rank"):
        runner._parse_request(payload)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("extra", True, "fields are not closed"),
        ("format_version", "unsupported", "version is unsupported"),
        ("tokenizer_json", {}, "non-empty object"),
        ("add_special_tokens", True, "add_special_tokens=false"),
        ("skip_special_tokens", False, "skip_special_tokens=true"),
        (
            "clean_up_tokenization_spaces",
            True,
            "clean_up_tokenization_spaces=false",
        ),
    ],
)
def test_runner_rejects_noncanonical_tokenizer_contract_controls(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload, engine = _runtime_request(tmp_path)
    tokenizer_path = engine.parent / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer[field] = value
    tokenizer_path.write_text(
        json.dumps(tokenizer, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._parse_request(payload)  # noqa: SLF001


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.update({"extra": True}), "fields are not closed"),
        (
            lambda config: config.update({"pretrained_config": []}),
            "sections must be objects",
        ),
        (
            lambda config: config["pretrained_config"].update({"mapping": []}),
            "mapping must be an object",
        ),
        (
            lambda config: config["pretrained_config"]["mapping"].update(
                {"world_size": 2}
            ),
            "single-rank engines",
        ),
        (
            lambda config: config["pretrained_config"]["mapping"].update(
                {"cp_size": 2}
            ),
            "single-rank engines",
        ),
    ],
)
def test_runner_rejects_open_or_multirank_engine_configuration(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:  # noqa: ANN001
    payload, engine = _runtime_request(tmp_path)
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mutation(config)
    config_path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._parse_request(payload)  # noqa: SLF001


def test_runner_network_gate_accepts_only_loopback(tmp_path: Path) -> None:
    ipv4 = tmp_path / "route"
    ipv6 = tmp_path / "ipv6_route"
    ipv4.write_text(
        "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
        "lo 00000000 00000000 0001 0 0 0 00000000 0 0 0\n",
        encoding="ascii",
    )
    ipv6.write_text(
        "00000000000000000000000000000000 00 00000000000000000000000000000000 "
        "00 00000000000000000000000000000000 00000000 00000000 00000000 00000001 lo\n",
        encoding="ascii",
    )
    runner._require_isolated_network_namespace(  # noqa: SLF001
        ipv4_route_path=ipv4,
        ipv6_route_path=ipv6,
    )

    ipv4.write_text(
        "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
        "eth0 00000000 00000000 0001 0 0 0 00000000 0 0 0\n",
        encoding="ascii",
    )
    with pytest.raises(runner.TensorRTLLMRunnerError, match="network-disabled"):
        runner._require_isolated_network_namespace(  # noqa: SLF001
            ipv4_route_path=ipv4,
            ipv6_route_path=ipv6,
        )


def test_runner_cli_accepts_only_exact_protocol_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runner,
        "_info_payload",
        lambda: {
            "backend_build_sha256": "a" * 64,
            "backend_name": "TensorRT-LLM",
        },
    )
    assert runner.main(["--invarlock-runtime-info-v1"]) == 0
    assert json.loads(capsys.readouterr().out)["backend_name"] == "TensorRT-LLM"
    assert runner.main(["--invarlock-runtime-info-v1", "extra"]) == 64


def test_runner_score_cli_emits_strict_response(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    payload, _engine = _runtime_request(tmp_path)

    class _Input:
        buffer = io.BytesIO(payload)

    monkeypatch.setattr(runner.sys, "stdin", _Input())
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")

    def noisy_backend_load() -> runner._Backend:  # noqa: SLF001
        os.write(1, b"vendor import diagnostic\n")
        os.write(2, b"vendor import warning\n")
        return fake_backend

    monkeypatch.setattr(runner, "_load_backend", noisy_backend_load)

    assert runner.main(["--invarlock-score-v1"]) == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": "OUT:hello world",
    }


def test_runner_batch_score_cli_emits_closed_ordered_response(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    payload, _engine = _runtime_batch_request(tmp_path)

    class _Input:
        buffer = io.BytesIO(payload)

    monkeypatch.setattr(runner.sys, "stdin", _Input())
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    monkeypatch.setattr(runner, "_load_backend", lambda: fake_backend)

    assert runner.main(["--invarlock-score-batch-v1"]) == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "format_version": "invarlock/tensorrt-llm-runner-batch-response-v1",
        "outputs": [
            {"output_text": "OUT:hello world", "record_id": "record/0"},
            {"output_text": "OUT:goodbye world", "record_id": "record/1"},
        ],
    }


def test_runner_batch_score_cli_rejects_aggregate_response_beyond_transport_bound(
    tmp_path: Path,
    fake_backend: runner._Backend,  # noqa: SLF001
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    payload, _engine = _runtime_batch_request(tmp_path)

    class _Input:
        buffer = io.BytesIO(payload)

    large_text = "x" * ((1024 * 1024) + 64)

    def aggregate_overflow(_self, prompts: object, **_kwargs: object) -> object:
        assert isinstance(prompts, list)
        return [
            SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(text=large_text)],
                prompt=prompt,
                prompt_token_ids=list(range(len(prompt.split()))),
            )
            for prompt in prompts
        ]

    monkeypatch.setattr(runner.sys, "stdin", _Input())
    monkeypatch.setattr(runner, "_require_runtime_boundary", lambda: None)
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    monkeypatch.setattr(runner, "_load_backend", lambda: fake_backend)
    monkeypatch.setattr(_FakeLLM, "generate", aggregate_overflow)

    assert runner.main(["--invarlock-score-batch-v1"]) == 70
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "batch runner response exceeds the byte limit" in captured.err


def test_runner_cli_maps_known_and_unknown_failures_to_closed_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "_info_payload",
        lambda: (_ for _ in ()).throw(
            runner.TensorRTLLMRunnerError("unavailable runtime")
        ),
    )
    assert runner.main(["--invarlock-runtime-info-v1"]) == 70
    assert "failed closed: unavailable runtime" in capsys.readouterr().err

    monkeypatch.setattr(
        runner,
        "_info_payload",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    assert runner.main(["--invarlock-runtime-info-v1"]) == 70
    assert capsys.readouterr().err == "TensorRT-LLM runner failed closed\n"
