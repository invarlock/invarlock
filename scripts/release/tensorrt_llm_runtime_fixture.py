#!/usr/bin/env python3
"""Build and qualify a real TensorRT-LLM fixture on two explicit GPUs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Final

from invarlock.core.runtime_provider import artifact_identity_sha256
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)

try:
    from scripts.release import tensorrt_llm_runtime_fixture_boundary as _boundary
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import tensorrt_llm_runtime_fixture_boundary as _boundary  # type: ignore[no-redef]

try:
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        BACKEND_VERSION,
        BUILD_RECIPE,
        MANIFEST_FORMAT,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        QUALIFICATION_FORMAT,
        TARGET_COMPUTE_CAPABILITY,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        IMAGE_DIGEST_RE as _IMAGE_DIGEST,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        SHA256_RE as _SHA256,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        FixtureContractError as TensorRTLLMFixtureError,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        canonical_json as _canonical_json,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        load_manifest as _load_manifest,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        load_qualification_summary as _load_qualification_summary,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        model_inventory_sha256 as _model_inventory_sha256,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        parse_object as _parse_object,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        sha256_file as _sha256_file,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        snapshot_model_tree as _snapshot_model_tree,
    )
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        snapshot_regular_file as _snapshot_regular_file,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tensorrt_llm_runtime_fixture_support import (  # type: ignore[no-redef]
        BACKEND_VERSION,
        BUILD_RECIPE,
        MANIFEST_FORMAT,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        QUALIFICATION_FORMAT,
        TARGET_COMPUTE_CAPABILITY,
    )
    from tensorrt_llm_runtime_fixture_support import (
        IMAGE_DIGEST_RE as _IMAGE_DIGEST,
    )
    from tensorrt_llm_runtime_fixture_support import (
        SHA256_RE as _SHA256,
    )
    from tensorrt_llm_runtime_fixture_support import (
        FixtureContractError as TensorRTLLMFixtureError,
    )
    from tensorrt_llm_runtime_fixture_support import (
        canonical_json as _canonical_json,
    )
    from tensorrt_llm_runtime_fixture_support import (
        load_manifest as _load_manifest,
    )
    from tensorrt_llm_runtime_fixture_support import (
        load_qualification_summary as _load_qualification_summary,
    )
    from tensorrt_llm_runtime_fixture_support import (
        model_inventory_sha256 as _model_inventory_sha256,
    )
    from tensorrt_llm_runtime_fixture_support import (
        parse_object as _parse_object,
    )
    from tensorrt_llm_runtime_fixture_support import (
        sha256_file as _sha256_file,
    )
    from tensorrt_llm_runtime_fixture_support import (
        snapshot_model_tree as _snapshot_model_tree,
    )
    from tensorrt_llm_runtime_fixture_support import (
        snapshot_regular_file as _snapshot_regular_file,
    )

BASE_DIGEST: Final = _boundary.BASE_DIGEST
PROMOTION_FORMAT: Final = "invarlock/tensorrt-llm-runtime-promotion-v1"
BUILD_RESULT_FORMAT: Final = "invarlock/tensorrt-llm-fixture-build-result-v1"
PROBE_RESULT_FORMAT: Final = "invarlock/tensorrt-llm-fixture-probe-result-v1"
CANARY_FORMAT: Final = "invarlock/tensorrt-llm-candidate-qualification-v1"
PREFLIGHT_FORMAT: Final = "invarlock/tensorrt-llm-runtime-preflight-v1"
_WORKER_CONTAINER = "/opt/invarlock-fixture/worker.py"
_MODEL_CONTAINER = "/opt/invarlock-fixture/model"
_OUTPUT_CONTAINER = "/opt/invarlock-fixture/output"
_ENGINE_CONTAINER = "/opt/invarlock-fixture/engine"
_TOKENIZER_CONTAINER = "/opt/invarlock-fixture/tokenizer.json"
_SELECTOR = re.compile(r"^device=(?:[0-9]+|GPU-[A-Fa-f0-9-]{20,80})$")
_MAX_CAPTURE = 4 * 1024 * 1024


def _run_captured(
    command: Sequence[str], *, timeout_seconds: int
) -> tuple[int, bytes, bytes]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(  # noqa: S603 - argv only, no shell
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
            )
            status = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise TensorRTLLMFixtureError("a container command timed out") from exc
        except OSError as exc:
            raise TensorRTLLMFixtureError(
                "the container engine could not be started"
            ) from exc
        stdout.seek(0)
        stderr.seek(0)
        stdout_bytes = stdout.read(_MAX_CAPTURE + 1)
        stderr_bytes = stderr.read(_MAX_CAPTURE + 1)
    if len(stdout_bytes) > _MAX_CAPTURE or len(stderr_bytes) > _MAX_CAPTURE:
        raise TensorRTLLMFixtureError("a container command exceeded its output limit")
    return status, stdout_bytes, stderr_bytes


def _validate_selectors(first: str, second: str) -> tuple[str, str]:
    selectors = (first, second)
    if any(_SELECTOR.fullmatch(value) is None for value in selectors):
        raise TensorRTLLMFixtureError(
            "GPU selectors must be explicit device=N or device=GPU-UUID values"
        )
    if first.casefold() == second.casefold():
        raise TensorRTLLMFixtureError(
            "GPU selectors must identify two distinct devices"
        )
    return selectors


def _validate_hardware(
    *, engine: str, selectors: tuple[str, str]
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Resolve two selectors and bind them to distinct target GPUs."""

    selectors = _validate_selectors(*selectors)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _boundary.probe_base_hardware,
                engine=engine,
                selector=selector,
                run_captured=_run_captured,
            )
            for selector in selectors
        ]
        hardware = (futures[0].result(), futures[1].result())
    if hardware[0][0].casefold() == hardware[1][0].casefold():
        raise TensorRTLLMFixtureError(
            "GPU selectors must resolve to two distinct physical devices"
        )
    if any(item[1] != TARGET_COMPUTE_CAPABILITY for item in hardware):
        raise TensorRTLLMFixtureError(
            "GPU selectors do not match the pinned compute capability"
        )
    return hardware


def _inspect_image(engine: str, image: str) -> str:
    engine = _boundary.validate_container_engine(engine)
    image = _boundary.validate_image_reference(image)
    status, stdout, _stderr = _run_captured(
        (engine, "image", "inspect", image), timeout_seconds=30
    )
    if status != 0:
        raise TensorRTLLMFixtureError("the candidate image cannot be inspected")
    try:
        decoded = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TensorRTLLMFixtureError(
            "the candidate image inspection is invalid"
        ) from exc
    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
        or not isinstance(decoded[0], dict)
    ):
        raise TensorRTLLMFixtureError("the candidate image inspection is ambiguous")
    inspection = decoded[0]
    digest = inspection.get("Id")
    config = inspection.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    expected = {
        "dev.invarlock.runtime-provider": "tensorrt_llm",
        "dev.invarlock.tensorrt-llm.base-digest": BASE_DIGEST,
        "dev.invarlock.tensorrt-llm.version": BACKEND_VERSION,
    }
    if not isinstance(digest, str) or _IMAGE_DIGEST.fullmatch(digest) is None:
        raise TensorRTLLMFixtureError("the candidate image has no canonical digest")
    if not isinstance(labels, Mapping) or any(
        labels.get(k) != v for k, v in expected.items()
    ):
        raise TensorRTLLMFixtureError("the candidate image labels do not match")
    return digest


def build_candidate_image(
    *, engine: str, image: str, source_date_epoch: str
) -> dict[str, object]:
    """Build and inspect the hard-pinned candidate without a command shell."""

    return _boundary.build_candidate_image(
        engine=engine,
        image=image,
        source_date_epoch=source_date_epoch,
        run_captured=_run_captured,
        inspect_image=_inspect_image,
    )


def smoke_candidate_image(
    *, engine: str, image: str, selector: str
) -> dict[str, object]:
    return _boundary.smoke_candidate_image(
        engine=engine,
        image=image,
        selector=selector,
        run_captured=_run_captured,
        inspect_image=_inspect_image,
    )


def preflight_flow(
    *,
    engine: str,
    image: str,
    stable_tag: str,
    source_date_epoch: str,
    smoke_selector: str,
    model: Path,
    output: Path,
    selectors: tuple[str, str],
    expected_model_inventory_sha256: str,
) -> dict[str, object]:
    """Reject invalid full-flow inputs before the candidate image build."""

    engine = _boundary.validate_container_engine(engine)
    image = _boundary.validate_candidate_tag(image)
    _boundary.validate_stable_tag(stable_tag, candidate_image=image)
    _boundary.validate_source_date_epoch(source_date_epoch)
    _boundary.validate_smoke_selector(smoke_selector)
    selectors = _validate_selectors(*selectors)
    if _SHA256.fullmatch(expected_model_inventory_sha256) is None:
        raise TensorRTLLMFixtureError(
            "expected model inventory must be a lowercase sha256 digest"
        )
    model = _boundary.validate_host_path(model, label="model").resolve(strict=True)
    output = _boundary.validate_host_path(
        output.resolve(strict=False), label="fixture output"
    )
    if output.exists():
        raise TensorRTLLMFixtureError("the fixture output directory must be new")
    if output.is_relative_to(model):
        raise TensorRTLLMFixtureError(
            "the fixture output must not be inside the model snapshot"
        )
    if _model_inventory_sha256(model) != expected_model_inventory_sha256:
        raise TensorRTLLMFixtureError(
            "the local model snapshot does not match the reviewed inventory"
        )
    _validate_hardware(engine=engine, selectors=selectors)
    return {
        "format_version": PREFLIGHT_FORMAT,
        "gpu_count": 2,
        "model_inventory_sha256": expected_model_inventory_sha256,
        "ok": True,
        "target_compute_capability": TARGET_COMPUTE_CAPABILITY,
    }


def _docker_prefix(
    *,
    engine: str,
    selector: str,
    worker: Path,
    image: str,
    volumes: Sequence[str] = (),
) -> list[str]:
    command = [
        engine,
        "run",
        "--rm",
        "--gpus",
        selector,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=8g",
        "--env",
        "FORCE_DETERMINISTIC=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "INVARLOCK_ALLOW_NETWORK=0",
        "--env",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        *_boundary.VENDOR_CACHE_ENV_ARGS,
        "--volume",
        f"{worker}:{_WORKER_CONTAINER}:ro",
    ]
    for volume in volumes:
        command.extend(("--volume", volume))
    command.extend(
        (
            *_boundary.VENDOR_BASH_ENTRYPOINT,
            image,
            *_boundary.VENDOR_ARGV_TRAMPOLINE,
            "/opt/invarlock/bin/vendor-python",
        )
    )
    return command


def _validate_build_result(payload: bytes) -> None:
    result = _parse_object(payload, label="fixture build result")
    if result != {
        "backend_version": BACKEND_VERSION,
        "format_version": BUILD_RESULT_FORMAT,
        "ok": True,
    }:
        raise TensorRTLLMFixtureError(
            "the fixture build result has an unexpected schema"
        )


def _build_one(
    *, engine: str, image: str, selector: str, worker: Path, model: Path, output: Path
) -> None:
    command = _docker_prefix(
        engine=engine,
        selector=selector,
        worker=worker,
        image=image,
        volumes=(
            f"{model}:{_MODEL_CONTAINER}:ro",
            f"{output.parent}:{_OUTPUT_CONTAINER}:rw",
        ),
    )
    command.extend(
        (
            _WORKER_CONTAINER,
            "build",
            "--model",
            _MODEL_CONTAINER,
            "--output",
            f"{_OUTPUT_CONTAINER}/{output.name}",
            "--repository",
            MODEL_REPOSITORY,
            "--revision",
            MODEL_REVISION,
        )
    )
    status, stdout, _stderr = _run_captured(command, timeout_seconds=7200)
    if status != 0:
        raise TensorRTLLMFixtureError("a TensorRT-LLM engine build failed")
    _validate_build_result(stdout)


def _copy_new(source: Path, destination: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
        with (
            source.open("rb") as input_stream,
            os.fdopen(descriptor, "wb") as output_stream,
        ):
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    except OSError as exc:
        raise TensorRTLLMFixtureError(
            "the frozen fixture cannot be copied safely"
        ) from exc


def _probe_one(
    *, engine: str, image: str, selector: str, worker: Path, fixture: Path
) -> str:
    command = _docker_prefix(
        engine=engine,
        selector=selector,
        worker=worker,
        image=image,
        volumes=(
            f"{fixture / 'engine'}:{_ENGINE_CONTAINER}:ro",
            f"{fixture / 'tokenizer.json'}:{_TOKENIZER_CONTAINER}:ro",
        ),
    )
    command.extend(
        (
            _WORKER_CONTAINER,
            "probe",
            "--engine",
            _ENGINE_CONTAINER,
            "--tokenizer",
            _TOKENIZER_CONTAINER,
        )
    )
    status, stdout, stderr = _run_captured(command, timeout_seconds=600)
    if status != 0 or stderr:
        raise TensorRTLLMFixtureError("a fixture probe failed")
    result = _parse_object(stdout, label="fixture probe result")
    if (
        set(result) != {"format_version", "ok", "output_text"}
        or result.get("format_version") != PROBE_RESULT_FORMAT
        or result.get("ok") is not True
        or not isinstance(result.get("output_text"), str)
    ):
        raise TensorRTLLMFixtureError(
            "the fixture probe result has an unexpected schema"
        )
    return result["output_text"]


def _write_new_json(path: Path, value: object) -> None:
    payload = _canonical_json(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except OSError as exc:
        raise TensorRTLLMFixtureError(
            "an output manifest cannot be created safely"
        ) from exc


def build_fixture(
    *,
    engine: str,
    image: str,
    model: Path,
    output: Path,
    selectors: tuple[str, str],
    expected_model_inventory_sha256: str,
) -> dict[str, object]:
    """Build on both GPUs concurrently, freeze one fixture, and cross-probe it."""

    selectors = _validate_selectors(*selectors)
    if _SHA256.fullmatch(expected_model_inventory_sha256) is None:
        raise TensorRTLLMFixtureError(
            "expected model inventory must be a lowercase sha256 digest"
        )
    model = _boundary.validate_host_path(model, label="model")
    output = _boundary.validate_host_path(
        output.resolve(strict=False), label="fixture output"
    )
    caller_worker = (
        Path(__file__).with_name("tensorrt_llm_runtime_fixture_worker.py").resolve()
    )
    caller_model = model.resolve(strict=True)
    if output.is_relative_to(caller_model):
        raise TensorRTLLMFixtureError(
            "the fixture output must not be inside the model snapshot"
        )
    _validate_hardware(engine=engine, selectors=selectors)
    image_digest = _inspect_image(engine, image)
    try:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        inputs = output / ".inputs"
        inputs.mkdir(mode=0o700)
        builds = output / ".builds"
        builds.mkdir(mode=0o700)
    except OSError as exc:
        raise TensorRTLLMFixtureError("the output directory must be new") from exc
    worker = inputs / "worker.py"
    model = inputs / "model"
    _snapshot_regular_file(caller_worker, worker)
    _snapshot_model_tree(caller_model, model)
    worker_sha256 = _sha256_file(worker)
    model_inventory = _model_inventory_sha256(model)
    if model_inventory != expected_model_inventory_sha256:
        raise TensorRTLLMFixtureError(
            "the owned model snapshot does not match the reviewed inventory"
        )
    build_outputs = (builds / "gpu-0", builds / "gpu-1")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _build_one,
                engine=engine,
                image=image_digest,
                selector=selector,
                worker=worker,
                model=model,
                output=build_output,
            )
            for selector, build_output in zip(selectors, build_outputs, strict=True)
        ]
        for future in futures:
            future.result()
    tokenizer_digests = tuple(
        _sha256_file(build / "tokenizer.json") for build in build_outputs
    )
    if tokenizer_digests[0] != tokenizer_digests[1]:
        raise TensorRTLLMFixtureError("the tokenizer contract did not reproduce")
    identities = tuple(
        read_tensorrt_llm_artifact_identity(
            build / "engine",
            target_compute_capability=TARGET_COMPUTE_CAPABILITY,
            tokenizer_metadata_sha256=tokenizer_digests[0],
        )
        for build in build_outputs
    )
    fixture = output / "fixture"
    engine_output = fixture / "engine"
    fixture.mkdir(mode=0o700)
    engine_output.mkdir(mode=0o700)
    for name in ("config.json", "rank0.engine"):
        _copy_new(build_outputs[0] / "engine" / name, engine_output / name)
    _copy_new(build_outputs[0] / "tokenizer.json", fixture / "tokenizer.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        probe_futures = [
            pool.submit(
                _probe_one,
                engine=engine,
                image=image_digest,
                selector=selector,
                worker=worker,
                fixture=fixture,
            )
            for selector in selectors
        ]
        outputs = tuple(future.result() for future in probe_futures)
    if outputs[0] != outputs[1]:
        raise TensorRTLLMFixtureError("the frozen fixture output differs across GPUs")
    identities_json = tuple(asdict(identity) for identity in identities)
    manifest: dict[str, object] = {
        "backend_version": BACKEND_VERSION,
        "build_recipe": dict(BUILD_RECIPE),
        "candidate_image_digest": image_digest,
        "engine_builds": {
            "primary": identities_json[0],
            "secondary": identities_json[1],
        },
        "engine_byte_reproduction": (
            "matched"
            if identities[0].engine_bundle_tree_sha256
            == identities[1].engine_bundle_tree_sha256
            else "different"
        ),
        "expected_output_sha256": hashlib.sha256(outputs[0].encode()).hexdigest(),
        "format_version": MANIFEST_FORMAT,
        "model": {
            "inventory_sha256": model_inventory,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "selected_engine_identity": identities_json[0],
        "tokenizer_sha256": tokenizer_digests[0],
        "worker": {"sha256": worker_sha256},
    }
    _write_new_json(output / "fixture-manifest.json", manifest)
    return manifest


def _canary_one(
    *,
    engine: str,
    image: str,
    image_digest: str,
    selector: str,
    fixture: Path,
    manifest: Mapping[str, object],
    expected_artifact_identity_sha256: str,
) -> dict[str, object]:
    identity = manifest["selected_engine_identity"]
    if not isinstance(identity, Mapping):
        raise TensorRTLLMFixtureError("the selected engine identity is invalid")
    engine_tree = identity.get("engine_bundle_tree_sha256")
    if not isinstance(engine_tree, str) or _SHA256.fullmatch(engine_tree) is None:
        raise TensorRTLLMFixtureError("the selected engine digest is invalid")
    command = [
        engine,
        "run",
        "--rm",
        "--gpus",
        selector,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=2g",
        "--env",
        "FORCE_DETERMINISTIC=1",
        "--env",
        "INVARLOCK_ALLOW_NETWORK=0",
        "--env",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "--env",
        f"INVARLOCK_RUNTIME_IMAGE={image_digest}",
        "--env",
        f"INVARLOCK_RUNTIME_IMAGE_DIGEST={image_digest}",
        *_boundary.VENDOR_CACHE_ENV_ARGS,
        "--volume",
        f"{fixture / 'engine'}:/opt/invarlock/canary/engine:ro",
        "--volume",
        f"{fixture / 'tokenizer.json'}:/opt/invarlock/canary/tokenizer.json:ro",
        *_boundary.VENDOR_BASH_ENTRYPOINT,
        image,
        *_boundary.VENDOR_ARGV_TRAMPOLINE,
        "/opt/invarlock/bin/vendor-python",
        "-m",
        "invarlock.runtime_providers.tensorrt_llm_canary",
        "--engine-bundle",
        "/opt/invarlock/canary/engine",
        "--tokenizer-contract",
        "/opt/invarlock/canary/tokenizer.json",
        "--runner",
        "/opt/invarlock/bin/tensorrt-llm-runner",
        "--expected-engine-tree-sha256",
        engine_tree,
        "--expected-tokenizer-sha256",
        str(manifest["tokenizer_sha256"]),
        "--expected-output-sha256",
        str(manifest["expected_output_sha256"]),
    ]
    status, stdout, stderr = _run_captured(command, timeout_seconds=900)
    if status != 0 or stderr:
        raise TensorRTLLMFixtureError("a dual-GPU candidate canary failed")
    result = _parse_object(stdout, label="candidate canary result")
    if set(result) != _boundary.CANARY_KEYS:
        raise TensorRTLLMFixtureError("a candidate canary result has an open schema")
    fixed = {
        "artifact_identity_sha256": expected_artifact_identity_sha256,
        "engine_bundle_tree_sha256": engine_tree,
        "format_version": CANARY_FORMAT,
        "ok": True,
        "output_sha256": manifest["expected_output_sha256"],
        "tokenizer_metadata_sha256": manifest["tokenizer_sha256"],
    }
    if any(result.get(name) != value for name, value in fixed.items()):
        raise TensorRTLLMFixtureError("a candidate canary result is invalid")
    for name in ("scoring_observation_sha256",):
        value = result.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise TensorRTLLMFixtureError("a candidate canary digest is invalid")
    return result


def qualify_two_gpu(
    *,
    engine: str,
    image: str,
    fixture_root: Path,
    output: Path,
    selectors: tuple[str, str],
) -> dict[str, object]:
    """Run the existing real-provider canary independently on both GPUs."""

    selectors = _validate_selectors(*selectors)
    fixture_root = _boundary.validate_host_path(fixture_root, label="fixture")
    output = _boundary.validate_host_path(output, label="qualification output")
    fixture_root = fixture_root.resolve(strict=True)
    _validate_hardware(engine=engine, selectors=selectors)
    manifest = _load_manifest(fixture_root / "fixture-manifest.json")
    image_digest = _inspect_image(engine, image)
    if image_digest != manifest["candidate_image_digest"]:
        raise TensorRTLLMFixtureError(
            "the candidate image differs from the fixture binding"
        )
    tokenizer_digest = _sha256_file(fixture_root / "fixture" / "tokenizer.json")
    if tokenizer_digest != manifest["tokenizer_sha256"]:
        raise TensorRTLLMFixtureError("the fixture tokenizer binding changed")
    identity = read_tensorrt_llm_artifact_identity(
        fixture_root / "fixture" / "engine",
        target_compute_capability=TARGET_COMPUTE_CAPABILITY,
        tokenizer_metadata_sha256=tokenizer_digest,
    )
    if asdict(identity) != manifest["selected_engine_identity"]:
        raise TensorRTLLMFixtureError("the frozen engine identity changed")
    expected_artifact_identity_sha256 = artifact_identity_sha256(identity)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _canary_one,
                engine=engine,
                image=image_digest,
                image_digest=image_digest,
                selector=selector,
                fixture=fixture_root / "fixture",
                manifest=manifest,
                expected_artifact_identity_sha256=expected_artifact_identity_sha256,
            )
            for selector in selectors
        ]
        results = tuple(future.result() for future in futures)
    if results[0] != results[1]:
        raise TensorRTLLMFixtureError(
            "the candidate canary evidence differs across GPUs"
        )
    summary: dict[str, object] = {
        "candidate_image_digest": image_digest,
        "engine_bundle_tree_sha256": identity.engine_bundle_tree_sha256,
        "format_version": QUALIFICATION_FORMAT,
        "gpu_count": 2,
        "ok": True,
        "output_sha256": manifest["expected_output_sha256"],
        "tokenizer_sha256": tokenizer_digest,
    }
    _write_new_json(output, summary)
    return summary


def promote_candidate(
    *, engine: str, image: str, qualification_summary: Path, stable_tag: str
) -> dict[str, object]:
    """Promote only the immutable image bound by a closed qualification summary."""

    qualification_summary = _boundary.validate_host_path(
        qualification_summary, label="qualification summary"
    )
    _boundary.validate_stable_tag(stable_tag, candidate_image=image)
    summary = _load_qualification_summary(qualification_summary)
    bound_digest = str(summary["candidate_image_digest"])
    if _inspect_image(engine, image) != bound_digest:
        raise TensorRTLLMFixtureError(
            "the mutable candidate tag changed after qualification"
        )
    status, _stdout, _stderr = _run_captured(
        (engine, "image", "tag", bound_digest, stable_tag), timeout_seconds=30
    )
    if status != 0:
        raise TensorRTLLMFixtureError("the immutable candidate could not be promoted")
    if _inspect_image(engine, stable_tag) != bound_digest:
        raise TensorRTLLMFixtureError(
            "the stable tag does not bind the qualified image"
        )
    return {
        "candidate_image_digest": bound_digest,
        "format_version": PROMOTION_FORMAT,
        "ok": True,
        "stable_tag": stable_tag,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _boundary.build_argument_parser(description=__doc__)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            if args.gpu_0 is None or args.gpu_1 is None:
                raise TensorRTLLMFixtureError(
                    "full-flow preflight requires two explicit GPU selectors"
                )
            result = preflight_flow(
                engine=args.container_engine,
                image=args.image,
                stable_tag=args.stable_tag,
                source_date_epoch=args.source_date_epoch,
                smoke_selector=args.smoke_gpus,
                model=args.model,
                output=args.output,
                selectors=(args.gpu_0, args.gpu_1),
                expected_model_inventory_sha256=args.expected_model_inventory_sha256,
            )
        elif args.command == "build-image":
            result = build_candidate_image(
                engine=args.container_engine,
                image=args.image,
                source_date_epoch=args.source_date_epoch,
            )
        elif args.command == "smoke-image":
            result = smoke_candidate_image(
                engine=args.container_engine,
                image=args.image,
                selector=args.smoke_gpus,
            )
        elif args.command == "build-fixture":
            if args.gpu_0 is None or args.gpu_1 is None:
                raise TensorRTLLMFixtureError(
                    "fixture builds require two explicit GPU selectors"
                )
            result = build_fixture(
                engine=args.container_engine,
                image=args.image,
                model=args.model,
                output=args.output,
                selectors=(args.gpu_0, args.gpu_1),
                expected_model_inventory_sha256=args.expected_model_inventory_sha256,
            )
        elif args.command == "qualify-two-gpu":
            if args.gpu_0 is None or args.gpu_1 is None:
                raise TensorRTLLMFixtureError(
                    "dual qualification requires two explicit GPU selectors"
                )
            result = qualify_two_gpu(
                engine=args.container_engine,
                image=args.image,
                fixture_root=args.fixture_root,
                output=args.output,
                selectors=(args.gpu_0, args.gpu_1),
            )
        else:
            result = promote_candidate(
                engine=args.container_engine,
                image=args.image,
                qualification_summary=args.qualification_summary,
                stable_tag=args.stable_tag,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"TensorRT-LLM fixture qualification failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entrypoint
    raise SystemExit(main())
