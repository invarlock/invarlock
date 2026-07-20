#!/usr/bin/env python3
"""Compare BF16 and calibrated FP8 Qwen3-0.6B TensorRT-LLM engines."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from examples.integrations.launch import _require_committed_checkout, _runtime_image
from invarlock.evidence_pack_contract import canonical_json_bytes

_MODEL = (
    "Qwen/Qwen3-0.6B",
    "c1899de289a04d12100db370d81485cdf75e47ca",
)
_VARIANTS = {"baseline": "none", "subject": "fp8"}


@dataclass(frozen=True)
class Paths:
    repository: Path
    workspace: Path
    models: Path
    work: Path
    resources: Path


def _paths(workspace: Path) -> Paths:
    repository = Path(__file__).resolve().parents[3]
    return Paths(
        repository=repository,
        workspace=workspace,
        models=workspace / "models",
        work=workspace / "engine-build",
        resources=workspace / "tensorrt-inputs",
    )


def _create_workspace(value: Path | None) -> Paths:
    if value is None:
        workspace = Path(tempfile.mkdtemp(prefix="invarlock-tensorrt-llm-")).resolve(
            strict=True
        )
    else:
        requested = value.expanduser().absolute()
        requested.parent.mkdir(parents=True, exist_ok=True)
        workspace = requested.parent.resolve(strict=True) / requested.name
        if os.path.lexists(requested) or os.path.lexists(workspace):
            raise FileExistsError(f"workspace already exists: {workspace}")
        workspace.mkdir()
    paths = _paths(workspace)
    for directory in (paths.models, paths.work, paths.resources):
        directory.mkdir()
        directory.chmod(0o777)
    return paths


def _download(paths: Paths) -> Path:
    repository, revision = _MODEL
    destination = paths.models / "qwen3-0.6b"
    observed = Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=destination,
        )
    ).resolve(strict=True)
    if not destination.is_dir() or observed != destination.resolve(strict=True):
        raise RuntimeError("snapshot download returned an unexpected destination")
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return destination


def _container_build(
    paths: Paths,
    *,
    role: str,
    device: str,
    image: str,
    container_engine: str,
) -> None:
    if not device.isdigit():
        raise ValueError("GPU device indices must be nonnegative integers")
    helper = Path(__file__).with_name("prepare.py").resolve(strict=True)
    role_work = paths.work / role
    role_work.mkdir(mode=0o777)
    role_work.chmod(0o777)
    command = [
        container_engine,
        "run",
        "--rm",
        "--network",
        "none",
        "--gpus",
        f"device={device}",
        "--pull=never",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "4096",
        "--user",
        "65532:65532",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=16g",
        "--env",
        "HOME=/tmp",
        "--env",
        "USER=65532",
        "--env",
        "LOGNAME=65532",
        "--env",
        "LD_LIBRARY_PATH=/usr/local/tensorrt/lib",
        "--mount",
        f"type=bind,src={paths.models / 'qwen3-0.6b'},dst=/model,readonly",
        "--mount",
        f"type=bind,src={role_work},dst=/work",
        "--mount",
        f"type=bind,src={paths.resources},dst=/resources",
        "--mount",
        f"type=bind,src={helper},dst=/example/prepare.py,readonly",
        "--entrypoint",
        "/opt/invarlock/bin/vendor-python",
        image,
        "/example/prepare.py",
        "--model",
        "/model",
        "--checkpoint",
        "/work/checkpoint",
        "--engine",
        f"/resources/{role}-engine",
        "--tokenizer-contract",
        f"/work/{role}.tokenizer-contract.json",
        "--quantization",
        _VARIANTS[role],
    ]
    if role == "subject":
        records = Path(__file__).with_name("records.json").resolve(strict=True)
        command[command.index("--entrypoint") : command.index("--entrypoint")] = [
            "--mount",
            f"type=bind,src={records},dst=/example/records.json,readonly",
        ]
        command.extend(["--calibration-records", "/example/records.json"])
    subprocess.run(command, check=True)


def _prepare_inputs(paths: Paths, *, tokenizer: Any | None = None) -> None:
    contracts = {
        role: (paths.work / role / f"{role}.tokenizer-contract.json").read_bytes()
        for role in _VARIANTS
    }
    if contracts["baseline"] != contracts["subject"]:
        raise RuntimeError("the pinned checkpoints do not share one tokenizer contract")
    (paths.resources / "tokenizer-contract.json").write_bytes(contracts["baseline"])

    source_records = Path(__file__).with_name("records.json")
    records = json.loads(source_records.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 102:
        raise RuntimeError(
            "the maintained causal-cloze schedule must contain 102 records"
        )
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        paths.models / "qwen3-0.6b",
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    for record in records:
        expected = record.get("expected") if isinstance(record, dict) else None
        if not isinstance(expected, str) or not expected.startswith(" "):
            raise RuntimeError(
                "each TensorRT exact-match target must preserve its leading space"
            )
        token_ids = tokenizer(expected, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(token_ids) != 1 or decoded != expected:
            raise RuntimeError(
                "each TensorRT exact-match target must be one losslessly decoded token"
            )
    with (paths.resources / "records.jsonl").open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record))
    (paths.resources / "policy.json").write_bytes(
        canonical_json_bytes(
            {
                "resolved_policy": {
                    "metrics": {
                        "exact_match": {
                            "delta_min_pp": -10.0,
                            "maximum_interval_width_pp": 20.0,
                            "minimum_record_count": 102,
                        }
                    }
                }
            }
        )
    )


def _run_transaction(
    paths: Paths,
    *,
    image: str,
    devices: tuple[str, str],
    container_engine: str,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run.py")),
        "--runtime-image",
        image,
        "--resource-root",
        str(paths.resources),
        "--baseline-locator",
        f"hf://{_MODEL[0]}@{_MODEL[1]}#tensorrt-llm-bf16-engine",
        "--subject-locator",
        f"derived://{_MODEL[0]}@{_MODEL[1]}#modelopt-fp8-tensorrt-llm-engine",
        "--baseline-device",
        f"cuda:{devices[0]}",
        "--subject-device",
        f"cuda:{devices[1]}",
    ]
    environment = dict(os.environ)
    environment["INVARLOCK_CONTAINER_ENGINE"] = container_engine
    subprocess.run(command, check=True, cwd=paths.repository, env=environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--container-engine", choices=("docker",), default="docker")
    parser.add_argument("--baseline-device", default="0")
    parser.add_argument("--subject-device", default="1")
    arguments = parser.parse_args(argv)
    try:
        if arguments.baseline_device == arguments.subject_device:
            raise ValueError("the showcase requires two distinct GPU indices")
        _require_committed_checkout(Path(__file__).resolve().parents[3])
        paths = _create_workspace(arguments.workspace)
        _download(paths)
        runtime_build = paths.workspace / "runtime-build"
        runtime_build.mkdir()
        image, digest = _runtime_image(
            repository=paths.repository,
            build_root=runtime_build,
            container_engine=arguments.container_engine,
            dockerfile="addins/tensorrt_llm/runtime/Dockerfile",
            image_prefix="invarlock-tensorrt-example-runtime",
        )
        if image != digest:
            raise RuntimeError("the local runtime image identity is not immutable")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            builds = [
                executor.submit(
                    _container_build,
                    paths,
                    role=role,
                    device=device,
                    image=image,
                    container_engine=arguments.container_engine,
                )
                for role, device in zip(
                    _VARIANTS,
                    (arguments.baseline_device, arguments.subject_device),
                    strict=True,
                )
            ]
            for future in builds:
                future.result()
        _prepare_inputs(paths)
        _run_transaction(
            paths,
            image=image,
            devices=(arguments.baseline_device, arguments.subject_device),
            container_engine=arguments.container_engine,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"Complete TensorRT-LLM example workspace: {paths.workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
