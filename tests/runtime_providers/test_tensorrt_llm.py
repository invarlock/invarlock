from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProvider,
    RuntimeSession,
    artifact_identity_sha256,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_providers import tensorrt_llm as tensorrt_llm_provider
from invarlock.runtime_providers import tensorrt_llm_session
from invarlock.runtime_providers.tensorrt_llm import (
    TensorRTLLMProvider,
    TensorRTLLMRuntimeBindings,
)
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)
from invarlock.runtime_providers.tensorrt_llm_session import (
    TensorRTLLMExecutionError,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64
_BACKEND_BUILD_SHA256 = "b" * 64
_BACKEND_VERSION = "1.2.1"
_REQUIRES_LINUX_FD_EXECUTION = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the pinned TensorRT-LLM image uses Linux descriptor execution",
)


def _bundle(root: Path) -> Path:
    root.mkdir()
    root.joinpath("config.json").write_text(
        json.dumps(
            {
                "build_config": {
                    "max_batch_size": 8,
                    "max_input_len": 128,
                    "max_seq_len": 256,
                },
                "pretrained_config": {
                    "architecture": "LlamaForCausalLM",
                    "dtype": "float16",
                    "mapping": {"pp_size": 1, "tp_size": 1, "world_size": 1},
                    "num_hidden_layers": 2,
                },
                "version": "1.0.0",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    root.joinpath("rank0.engine").write_bytes(b"serialized-test-engine")
    return root


@pytest.fixture(autouse=True)
def _authenticated_container_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", f"invarlock-runtime@{_IMAGE_DIGEST}")
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: True,
    )


def _write_fake_runner(path: Path, *, compute_capability: str = "9.0") -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
import time

INFO = {{
    "backend_build_sha256": "{_BACKEND_BUILD_SHA256}",
    "backend_name": "TensorRT-LLM",
    "backend_version": "{_BACKEND_VERSION}",
    "cuda_compute_capability": "{compute_capability}",
    "cuda_device_name": "Observed NVIDIA H200",
    "cuda_driver_version": "570.00",
    "cuda_runtime_version": "12.8",
    "device_kind": "cuda",
    "format_version": "invarlock/tensorrt-llm-runner-info-v1",
    "protocol_version": "invarlock/tensorrt-llm-runner-v1",
}}

if sys.argv[1:] == ["--invarlock-runtime-info-v1"]:
    print(json.dumps(INFO, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0)
if sys.argv[1:] != ["--invarlock-score-v1"]:
    raise SystemExit(64)

request = json.load(sys.stdin)
prompt = request["input_text"]
if prompt == "__sleep__":
    time.sleep(30)
elif prompt == "__flood__":
    os.write(1, b"x" * (3 * 1024 * 1024))
    raise SystemExit(0)
elif prompt == "__stderr__":
    os.write(2, b"unexpected diagnostic")
    response = {{
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": "x",
    }}
elif prompt == "__fail__":
    raise SystemExit(7)
elif prompt == "__bad_json__":
    os.write(1, b"not-json")
    raise SystemExit(0)
elif prompt == "__duplicate__":
    os.write(1, b'{{"format_version":"a","format_version":"b","output_text":"x"}}')
    raise SystemExit(0)
elif prompt == "__extra__":
    response = {{
        "extra": True,
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": "x",
    }}
elif prompt == "__env__":
    response = {{
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": json.dumps(dict(os.environ), sort_keys=True, separators=(",", ":")),
    }}
elif prompt == "__request__":
    response = {{
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": json.dumps(request, sort_keys=True, separators=(",", ":")),
    }}
else:
    response = {{
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": "OUT:" + prompt,
    }}
print(json.dumps(response, sort_keys=True, separators=(",", ":")))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _runtime_inputs(
    tmp_path: Path,
) -> tuple[ModelRuntimeSpec, TensorRTLLMRuntimeBindings, RuntimeExecutionContext]:
    tokenizer = tmp_path / "private-tokenizer.json"
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
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    bundle = _bundle(tmp_path / "private-engine-name")
    identity = read_tensorrt_llm_artifact_identity(
        bundle,
        target_compute_capability="9.0",
        tokenizer_metadata_sha256=tokenizer_sha256,
    )
    runner = tmp_path / "private-tensorrt-runner"
    _write_fake_runner(runner)
    runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
    spec = ModelRuntimeSpec(
        provider_name="tensorrt_llm",
        model_id=identity.bundle_name,
        settings={
            "backend_build_sha256": _BACKEND_BUILD_SHA256,
            "backend_version": _BACKEND_VERSION,
            "batch_size": 4,
            "builder_config_sha256": identity.builder_config_sha256,
            "context_length": 256,
            "engine_bundle_tree_sha256": identity.engine_bundle_tree_sha256,
            "engine_metadata_sha256": identity.engine_metadata_sha256,
            "file_inventory_sha256": identity.file_inventory_sha256,
            "max_output_tokens": 16,
            "runner_binary_sha256": runner_sha256,
            "seed": 7,
            "target_compute_capability": "9.0",
            "timeout_seconds": 1,
            "tokenizer_metadata_sha256": tokenizer_sha256,
        },
    )
    bindings = TensorRTLLMRuntimeBindings(
        engine_bundle_path=bundle,
        tokenizer_contract_path=tokenizer,
        runner_executable_path=runner,
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cuda",
        artifact_identity_sha256=artifact_identity_sha256(identity),
        native_model=bindings,
    )
    return spec, bindings, context


def _record(record_id: str, text: str) -> EvaluationRecord:
    return EvaluationRecord(
        record_id=record_id,
        input_text=text,
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),
        expected_output=f"OUT:{text}",
    )


def test_tensorrt_llm_closed_environment_pins_vendor_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/ambient-libraries")
    monkeypatch.setenv("OPAL_PREFIX", "/tmp/ambient-opal")
    monkeypatch.setenv("PATH", "/tmp/ambient-bin")
    monkeypatch.setenv("INVARLOCK_TEST_SECRET", "must-not-cross-boundary")
    run_directory = tensorrt_llm_session._RunDirectory(  # noqa: SLF001
        path=tmp_path,
        descriptor=-1,
        initial_stat=tmp_path.stat(),
    )
    monkeypatch.setattr(run_directory, "recheck", lambda: None)

    environment = run_directory.environment()

    rendered = str(tmp_path)
    assert environment == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "DO_NOT_TRACK": "1",
        "FORCE_DETERMINISTIC": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": rendered,
        "INVARLOCK_CONTAINER_EXECUTION": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": "/usr/local/tensorrt/lib",
        "NO_COLOR": "1",
        "NO_PROXY": "*",
        "OPAL_PREFIX": "/opt/hpcx/ompi",
        "PATH": "/opt/hpcx/ompi/bin:/usr/bin:/bin",
        "TELEMETRY_DISABLED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TRTLLM_NO_USAGE_STATS": "1",
        "TMPDIR": rendered,
        "XDG_CACHE_HOME": rendered,
    }
    fixed_paths = [environment["OPAL_PREFIX"]]
    fixed_paths.extend(
        entry
        for variable in ("LD_LIBRARY_PATH", "PATH")
        for entry in environment[variable].split(":")
    )
    assert all(
        Path(entry).is_absolute() and entry not in {rendered, "/tmp", "/var/tmp"}
        for entry in fixed_paths
    )


def _batch(*records: EvaluationRecord) -> EvaluationBatch:
    return EvaluationBatch(schedule_sha256="c" * 64, records=tuple(records))


def test_tensorrt_llm_config_identity_capabilities_and_private_bindings(
    tmp_path: Path,
) -> None:
    spec, bindings, _context = _runtime_inputs(tmp_path)
    provider = TensorRTLLMProvider()

    provider.validate_config(spec)
    identity = provider.identify_artifact(spec)

    assert isinstance(provider, RuntimeProvider)
    assert identity.bundle_name == spec.model_id
    assert (
        identity.engine_bundle_tree_sha256
        == (spec.settings["engine_bundle_tree_sha256"])
    )
    assert provider.capabilities().artifact_formats == ("tensorrt_llm_engine",)
    assert provider.capabilities().metrics == ("exact_match",)
    assert provider.capabilities().execution_modes == ("container",)
    assert provider.capabilities().evidence_surfaces == (
        "behavior",
        "tokenizer",
        "build",
    )
    assert str(bindings.engine_bundle_path) not in repr(bindings)
    assert str(bindings.tokenizer_contract_path) not in repr(bindings)
    assert str(bindings.runner_executable_path) not in repr(bindings)
    assert all("path" not in name for name in spec.settings)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"strict": False}, "strict mode"),
        ({"allow_network": True}, "disable network"),
        ({"container_image_digest": None}, "container image"),
        ({"artifact_identity_sha256": None}, "artifact identity"),
        ({"artifact_identity_sha256": "0" * 64}, "does not match"),
        ({"device_kind": "cpu"}, "CUDA device"),
        ({"native_model": None}, "runtime bindings"),
    ],
)
def test_tensorrt_llm_open_rejects_missing_security_bindings(
    tmp_path: Path, replacement: dict[str, object], message: str
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    with pytest.raises((ValueError, TensorRTLLMExecutionError), match=message):
        TensorRTLLMProvider().open(spec, replace(context, **replacement))


def test_tensorrt_llm_rejects_host_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: False,
    )
    with pytest.raises(ValueError, match="container boundary"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_rejects_marker_only_container_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: False,
    )
    with pytest.raises(ValueError, match="container boundary"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_rejects_unobserved_runtime_image_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", "sha256:" + "f" * 64)

    with pytest.raises(ValueError, match="does not match the container context"):
        TensorRTLLMProvider().open(spec, context)

    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "mutable:tag")
    with pytest.raises(ValueError, match="embed the exact runtime image digest"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_rejects_tokenizer_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    bindings.tokenizer_contract_path.write_bytes(b"changed")
    with pytest.raises(TensorRTLLMExecutionError, match="digest"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_snapshot_rejects_multi_rank_engine(tmp_path: Path) -> None:
    source = _bundle(tmp_path / "source")
    config = json.loads(source.joinpath("config.json").read_text(encoding="utf-8"))
    config["pretrained_config"]["mapping"] = {
        "pp_size": 1,
        "tp_size": 2,
        "world_size": 2,
    }
    source.joinpath("config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    source.joinpath("rank1.engine").write_bytes(b"second rank")

    with pytest.raises(TensorRTLLMExecutionError, match="single-rank"):
        tensorrt_llm_session._snapshot_bundle(  # noqa: SLF001
            source, tmp_path / "destination"
        )


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_scores_in_order_and_emits_bound_receipt(
    tmp_path: Path,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    provider = TensorRTLLMProvider()
    session = provider.open(spec, context)

    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    observation = session.score(_batch(_record("a", "alpha"), _record("b", "beta")))
    receipt = session.runtime_receipt()

    assert isinstance(session, RuntimeSession)
    assert tuple(record.record_id for record in observation.records) == ("a", "b")
    assert tuple(record.output_text for record in observation.records) == (
        "OUT:alpha",
        "OUT:beta",
    )
    assert observation.aggregate_source_sha256 == runtime_scoring_records_sha256(
        [asdict(record) for record in observation.records]
    )
    assert receipt.backend.name == "TensorRT-LLM"
    assert receipt.backend.version == _BACKEND_VERSION
    assert receipt.backend.binary_sha256 == spec.settings["runner_binary_sha256"]
    assert receipt.backend.build_sha256 == _BACKEND_BUILD_SHA256
    assert receipt.backend.source_sha256 is None
    assert receipt.artifact_identity == provider.identify_artifact(spec)
    assert receipt.outer_image_digest == _IMAGE_DIGEST
    assert receipt.device.device_name == "Observed NVIDIA H200"
    assert receipt.device.compute_capability == "9.0"
    assert receipt.device.driver_version == "570.00"
    assert receipt.device.cuda_runtime_version == "12.8"
    assert (
        receipt.scoring_observation_sha256
        == hashlib.sha256(
            json.dumps(
                asdict(observation),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.score(_batch(_record("c", "gamma")))


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_rejects_runner_identity_mismatch(tmp_path: Path) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    wrong_version = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={**spec.settings, "backend_version": "1.2.2"},
    )
    with pytest.raises(TensorRTLLMExecutionError, match="identity"):
        TensorRTLLMProvider().open(wrong_version, context)


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_rejects_observed_compute_capability_mismatch(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    _write_fake_runner(bindings.runner_executable_path, compute_capability="8.9")
    runner_sha256 = hashlib.sha256(
        bindings.runner_executable_path.read_bytes()
    ).hexdigest()
    mismatched_runner = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={**spec.settings, "runner_binary_sha256": runner_sha256},
    )

    with pytest.raises(TensorRTLLMExecutionError, match="observed CUDA"):
        TensorRTLLMProvider().open(mismatched_runner, context)


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_rejects_input_digest_and_runner_path_swap(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    invalid = EvaluationRecord(
        record_id="bad-digest",
        input_text="alpha",
        input_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="does not match input_text"):
        session.score(_batch(invalid))

    bindings.runner_executable_path.unlink()
    _write_fake_runner(bindings.runner_executable_path)
    with pytest.raises(TensorRTLLMExecutionError, match="identity changed"):
        session.score(_batch(_record("a", "alpha")))


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_uses_closed_request_and_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    real_popen = tensorrt_llm_session.subprocess.Popen

    def recording_popen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(dict(kwargs))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(tensorrt_llm_session.subprocess, "Popen", recording_popen)
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setenv("HTTP_PROXY", "http://private-proxy")
    monkeypatch.setenv("INVARLOCK_TEST_SECRET", "private-value")
    session = TensorRTLLMProvider().open(spec, context)

    request_record = session.score(_batch(_record("request", "__request__"))).records[0]
    request = json.loads(request_record.output_text)
    assert request["format_version"] == "invarlock/tensorrt-llm-runner-request-v1"
    assert request["protocol_version"] == "invarlock/tensorrt-llm-runner-v1"
    assert request["settings"] == {
        "allow_network": False,
        "batch_size": 4,
        "context_length": 256,
        "max_output_tokens": 16,
        "seed": 7,
        "timeout_seconds": 1,
    }
    assert request["engine_bundle"].endswith("/engine")
    assert request["tokenizer_contract"].endswith("/tokenizer.json")

    env_record = session.score(_batch(_record("env", "__env__"))).records[0]
    environment = json.loads(env_record.output_text)
    assert environment["DO_NOT_TRACK"] == "1"
    assert environment["FORCE_DETERMINISTIC"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["NO_PROXY"] == "*"
    assert environment["INVARLOCK_CONTAINER_EXECUTION"] == "1"
    assert environment["TELEMETRY_DISABLED"] == "1"
    assert environment["TRTLLM_NO_USAGE_STATS"] == "1"
    assert environment["HOME"] == environment["TMPDIR"]
    assert environment["HOME"] == environment["XDG_CACHE_HOME"]
    assert "HF_TOKEN" not in environment
    assert "HTTP_PROXY" not in environment
    assert "INVARLOCK_TEST_SECRET" not in environment
    assert calls
    assert all(call["shell"] is False for call in calls)
    assert all(call["start_new_session"] is True for call in calls)


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ("__sleep__", "timed out"),
        ("__flood__", "limit exceeded"),
        ("__stderr__", "emitted stderr"),
        ("__fail__", "status 7"),
        ("__bad_json__", "strict JSON"),
        ("__duplicate__", "duplicate key"),
        ("__extra__", "unexpected fields"),
    ],
)
@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_fails_closed_on_runner_errors(
    tmp_path: Path, prompt: str, message: str
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    with pytest.raises(TensorRTLLMExecutionError, match=message):
        session.score(_batch(_record("bad", prompt)))


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_timeout_kills_the_child_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    killed: list[int] = []
    real_kill = tensorrt_llm_session._kill_process_group

    def recording_kill(process):  # noqa: ANN001, ANN202
        killed.append(process.pid)
        real_kill(process)

    monkeypatch.setattr(tensorrt_llm_session, "_kill_process_group", recording_kill)
    session = TensorRTLLMProvider().open(spec, context)
    with pytest.raises(TensorRTLLMExecutionError, match="timed out"):
        session.score(_batch(_record("sleep", "__sleep__")))
    assert killed


@_REQUIRES_LINUX_FD_EXECUTION
def test_tensorrt_llm_uses_private_snapshot_and_detects_mutation(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)

    # The authenticated private snapshot, not this caller-controlled source, is run.
    bindings.engine_bundle_path.joinpath("rank0.engine").write_bytes(b"source changed")
    observation = session.score(_batch(_record("a", "alpha")))
    assert observation.records[0].output_text == "OUT:alpha"

    session._tokenizer_snapshot.write_bytes(b"snapshot changed")  # noqa: SLF001
    with pytest.raises(TensorRTLLMExecutionError, match="tokenizer contract changed"):
        session.score(_batch(_record("b", "beta")))


def test_tensorrt_llm_modules_remain_torch_free() -> None:
    root = Path.cwd() / "src" / "invarlock" / "runtime_providers"
    source = "\n".join(
        root.joinpath(name).read_text(encoding="utf-8")
        for name in (
            "tensorrt_llm.py",
            "tensorrt_llm_runner.py",
            "tensorrt_llm_session.py",
        )
    )
    assert "import torch" not in source
    assert "import tensorrt" not in source
    assert "import transformers" not in source
