from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import sys
from pathlib import Path

import pytest
from invarlock_addins.tensorrt_llm import execution as tensorrt_llm_execution
from invarlock_addins.tensorrt_llm.provider import TensorRTLLMRuntimeBindings

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    artifact_identity_sha256,
    evaluation_input_parts_sha256,
)
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64
_BACKEND_BUILD_SHA256 = "b" * 64
_BACKEND_VERSION = "1.2.1"
_REQUIRES_POSIX_PINNING = pytest.mark.skipif(
    os.name != "posix",
    reason="the pinned TensorRT-LLM runtime requires POSIX nofollow support",
)


def _bundle(root: Path) -> Path:
    root.mkdir()
    root.joinpath("config.json").write_text(
        json.dumps(
            {
                "build_config": {
                    "max_batch_size": 8,
                    "max_input_len": 256,
                    "max_seq_len": 512,
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
elif prompt == "__orphan_pipe__":
    grandchild_pid = os.fork()
    if grandchild_pid == 0:
        time.sleep(30)
        os._exit(0)
    pid_path = os.path.join(os.environ["HOME"], "grandchild.pid")
    with open(pid_path, "w", encoding="ascii") as pid_file:
        pid_file.write(str(grandchild_pid))
        pid_file.flush()
        os.fsync(pid_file.fileno())
    raise SystemExit(0)
elif prompt == "__detached_success__":
    grandchild_pid = os.fork()
    if grandchild_pid == 0:
        os.closerange(0, 3)
        time.sleep(30)
        os._exit(0)
    pid_path = os.path.join(os.environ["HOME"], "detached.pid")
    with open(pid_path, "w", encoding="ascii") as pid_file:
        pid_file.write(str(grandchild_pid))
        pid_file.flush()
        os.fsync(pid_file.fileno())
    response = {{
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": "OUT:" + prompt,
    }}
elif prompt == "__wait_for_release__":
    started_path = os.path.join(os.environ["HOME"], "score.started")
    release_path = os.path.join(os.environ["HOME"], "score.release")
    with open(started_path, "w", encoding="ascii") as started_file:
        started_file.write("ready")
        started_file.flush()
        os.fsync(started_file.fileno())
    while not os.path.exists(release_path):
        time.sleep(0.01)
    response = {{
        "format_version": "invarlock/tensorrt-llm-runner-response-v1",
        "output_text": "released",
    }}
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


def _write_fake_vendor_python(path: Path) -> None:
    path.parent.chmod(0o700)
    parent_stat = path.parent.stat()
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or parent_stat.st_uid != os.geteuid()
    ):
        raise AssertionError("test vendor Python requires a private owned parent")
    path.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
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
    runner = Path(tensorrt_llm_execution._OFFICIAL_RUNNER_PATH)  # noqa: SLF001
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
        provider_state=bindings,
    )
    return spec, bindings, context


def _record(record_id: str, text: str) -> EvaluationRecord:
    parts = (
        EvaluationInputPart(
            kind="text",
            role="prompt",
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        ),
    )
    return EvaluationRecord(
        record_id=record_id,
        input_text=text,
        input_sha256=evaluation_input_parts_sha256(parts),
        expected_output=f"OUT:{text}",
        input_parts=parts,
    )


def _batch(*records: EvaluationRecord) -> EvaluationBatch:
    return EvaluationBatch(schedule_sha256="c" * 64, records=tuple(records))


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status_path = Path(f"/proc/{pid}/stat")
    if Path("/proc").is_dir():
        try:
            process_stat = status_path.read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError as exc:
            raise AssertionError(f"could not inspect Linux process {pid}") from exc
        state, _process_group, _start_time = _parse_linux_process_stat(process_stat)
        return _linux_process_state_is_running(state)
    return True


def _parse_linux_process_stat(process_stat: str) -> tuple[str, int, int]:
    """Parse state, process group, and start time from one procfs stat record."""

    closing_delimiter = process_stat.rfind(") ")
    if closing_delimiter < 0:
        raise ValueError("Linux process stat is missing its command delimiter")
    fields = process_stat[closing_delimiter + 2 :].split()
    if len(fields) < 20:
        raise ValueError("Linux process stat is truncated")
    state = fields[0]
    if len(state) != 1:
        raise ValueError("Linux process stat has an invalid state")
    try:
        process_group = int(fields[2])
        start_time = int(fields[19])
    except ValueError as exc:
        raise ValueError("Linux process stat has invalid numeric fields") from exc
    return state, process_group, start_time


def _linux_process_state_is_running(state: str) -> bool:
    return state not in {"Z", "X", "x"}


def _process_diagnostic(pid: int) -> str:
    status_path = Path(f"/proc/{pid}/stat")
    if not Path("/proc").is_dir():
        return f"pid={pid}; procfs unavailable"
    try:
        process_stat = status_path.read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return f"pid={pid}; process absent"
    except OSError as exc:
        return f"pid={pid}; procfs read failed: {type(exc).__name__}"
    try:
        state, process_group, start_time = _parse_linux_process_stat(process_stat)
    except ValueError as exc:
        return f"pid={pid}; malformed procfs stat: {exc}"
    return (
        f"pid={pid}; state={state}; process_group={process_group}; "
        f"start_time={start_time}"
    )
