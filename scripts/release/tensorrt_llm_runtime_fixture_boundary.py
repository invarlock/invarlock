"""Validated process boundary for the maintained TensorRT-LLM build flow."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

try:
    from scripts.release.tensorrt_llm_runtime_fixture_support import (
        IMAGE_DIGEST_RE,
        STABLE_TAG_RE,
        FixtureContractError,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tensorrt_llm_runtime_fixture_support import (  # type: ignore[no-redef]
        IMAGE_DIGEST_RE,
        STABLE_TAG_RE,
        FixtureContractError,
    )

BASE_DIGEST: Final = (
    "sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9"
)
BASE_IMAGE: Final = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1@" + BASE_DIGEST
BUILD_FORMAT: Final = "invarlock/tensorrt-llm-runtime-image-build-v1"
SMOKE_FORMAT: Final = "invarlock/tensorrt-llm-runtime-image-smoke-v1"
ENV_CONTAINER_ENGINE: Final = "INVARLOCK_TENSORRT_LLM_CONTAINER_ENGINE"
ENV_IMAGE: Final = "INVARLOCK_TENSORRT_LLM_IMAGE"
ENV_STABLE_TAG: Final = "INVARLOCK_TENSORRT_LLM_STABLE_TAG"
ENV_GPU_0: Final = "INVARLOCK_TENSORRT_LLM_GPU_0"
ENV_GPU_1: Final = "INVARLOCK_TENSORRT_LLM_GPU_1"
ENV_SMOKE_GPU: Final = "INVARLOCK_TENSORRT_LLM_SMOKE_GPU"
ENV_MODEL: Final = "INVARLOCK_TENSORRT_LLM_MODEL"
ENV_FIXTURE_ROOT: Final = "INVARLOCK_TENSORRT_LLM_FIXTURE_ROOT"
ENV_MODEL_INVENTORY: Final = "INVARLOCK_TENSORRT_LLM_MODEL_INVENTORY_SHA256"
ENV_SOURCE_DATE_EPOCH: Final = "INVARLOCK_TENSORRT_LLM_SOURCE_DATE_EPOCH"
_SOURCE_DATE_EPOCH_RE = re.compile(r"^(?:0|[1-9][0-9]{0,11})$")
_SMOKE_SELECTOR_RE = re.compile(r"^(?:all|device=(?:[0-9]+|GPU-[A-Fa-f0-9-]{20,80}))$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Fa-f0-9-]{20,80}$")
_COMPUTE_CAPABILITY_RE = re.compile(r"^(?:0|[1-9][0-9]?)\.(?:0|[1-9][0-9]?)$")
CANARY_KEYS: Final = frozenset(
    {
        "artifact_identity_sha256",
        "engine_bundle_tree_sha256",
        "format_version",
        "ok",
        "output_sha256",
        "scoring_observation_sha256",
        "tokenizer_metadata_sha256",
    }
)

InspectImage = Callable[[str, str], str]


def environment_default(name: str) -> str | None:
    """Return a non-empty Make-exported default without evaluating its contents."""

    value = os.environ.get(name)
    return value if value else None


def environment_path_default(name: str) -> Path | None:
    value = environment_default(name)
    return Path(value) if value is not None else None


def build_argument_parser(*, description: str | None) -> argparse.ArgumentParser:
    """Create the CLI with Make-exported values used only as typed defaults."""

    image = environment_default(ENV_IMAGE)
    fixture = environment_path_default(ENV_FIXTURE_ROOT)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--container-engine",
        default=environment_default(ENV_CONTAINER_ENGINE) or "docker",
    )
    parser.add_argument("--image", default=image, required=image is None)
    parser.add_argument("--gpu-0", default=environment_default(ENV_GPU_0))
    parser.add_argument("--gpu-1", default=environment_default(ENV_GPU_1))
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight_model = environment_path_default(ENV_MODEL)
    preflight.add_argument(
        "--model",
        type=Path,
        default=preflight_model,
        required=preflight_model is None,
    )
    preflight.add_argument(
        "--output", type=Path, default=fixture, required=fixture is None
    )
    preflight_inventory = environment_default(ENV_MODEL_INVENTORY)
    preflight.add_argument(
        "--expected-model-inventory-sha256",
        default=preflight_inventory,
        required=preflight_inventory is None,
    )
    preflight_epoch = environment_default(ENV_SOURCE_DATE_EPOCH)
    preflight.add_argument(
        "--source-date-epoch",
        default=preflight_epoch,
        required=preflight_epoch is None,
    )
    preflight.add_argument(
        "--smoke-gpus", default=environment_default(ENV_SMOKE_GPU) or "all"
    )
    preflight_stable = environment_default(ENV_STABLE_TAG)
    preflight.add_argument(
        "--stable-tag", default=preflight_stable, required=preflight_stable is None
    )
    image_build = commands.add_parser("build-image")
    epoch = environment_default(ENV_SOURCE_DATE_EPOCH)
    image_build.add_argument(
        "--source-date-epoch", default=epoch, required=epoch is None
    )
    smoke = commands.add_parser("smoke-image")
    smoke.add_argument(
        "--smoke-gpus", default=environment_default(ENV_SMOKE_GPU) or "all"
    )
    fixture_build = commands.add_parser("build-fixture")
    model = environment_path_default(ENV_MODEL)
    fixture_build.add_argument(
        "--model", type=Path, default=model, required=model is None
    )
    fixture_build.add_argument(
        "--output", type=Path, default=fixture, required=fixture is None
    )
    inventory = environment_default(ENV_MODEL_INVENTORY)
    fixture_build.add_argument(
        "--expected-model-inventory-sha256",
        default=inventory,
        required=inventory is None,
    )
    qualification = commands.add_parser("qualify-two-gpu")
    qualification.add_argument(
        "--fixture-root", type=Path, default=fixture, required=fixture is None
    )
    qualification.add_argument(
        "--output",
        type=Path,
        default=fixture / "qualification-summary.json" if fixture else None,
        required=fixture is None,
    )
    promotion = commands.add_parser("promote")
    promotion.add_argument(
        "--qualification-summary",
        type=Path,
        default=fixture / "qualification-summary.json" if fixture else None,
        required=fixture is None,
    )
    stable = environment_default(ENV_STABLE_TAG)
    promotion.add_argument("--stable-tag", default=stable, required=stable is None)
    return parser


def validate_container_engine(engine: str) -> str:
    if engine != "docker":
        raise FixtureContractError(
            "the maintained TensorRT-LLM flow requires the docker engine"
        )
    return engine


def validate_image_reference(image: str) -> str:
    if (
        STABLE_TAG_RE.fullmatch(image) is None
        and IMAGE_DIGEST_RE.fullmatch(image) is None
    ):
        raise FixtureContractError("the TensorRT-LLM image reference is invalid")
    return image


def validate_candidate_tag(image: str) -> str:
    if STABLE_TAG_RE.fullmatch(image) is None or image.startswith("sha256:"):
        raise FixtureContractError("the TensorRT-LLM candidate tag is invalid")
    return image


def validate_smoke_selector(selector: str) -> str:
    if _SMOKE_SELECTOR_RE.fullmatch(selector) is None:
        raise FixtureContractError("the TensorRT-LLM smoke GPU selector is invalid")
    return selector


def validate_source_date_epoch(value: str) -> str:
    if _SOURCE_DATE_EPOCH_RE.fullmatch(value) is None:
        raise FixtureContractError(
            "SOURCE_DATE_EPOCH must be a bounded decimal integer"
        )
    return value


def validate_stable_tag(stable_tag: str, *, candidate_image: str) -> str:
    if (
        STABLE_TAG_RE.fullmatch(stable_tag) is None
        or stable_tag.startswith("sha256:")
        or stable_tag == candidate_image
    ):
        raise FixtureContractError("the stable image tag is invalid or unsafe")
    return stable_tag


def validate_host_path(path: Path, *, label: str) -> Path:
    value = os.fspath(path)
    if ":" in value or any(ord(character) < 32 for character in value):
        raise FixtureContractError(f"the {label} path is invalid for a container mount")
    return path


def probe_base_hardware(
    *,
    engine: str,
    selector: str,
    run_captured: Callable[..., tuple[int, bytes, bytes]],
) -> tuple[str, str]:
    """Resolve one selector through the exact base image without a shell."""

    engine = validate_container_engine(engine)
    if _SMOKE_SELECTOR_RE.fullmatch(selector) is None or selector == "all":
        raise FixtureContractError(
            "the hardware preflight requires one explicit GPU selector"
        )
    command = (
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
        "--entrypoint",
        "nvidia-smi",
        BASE_IMAGE,
        "--query-gpu=uuid,compute_cap",
        "--format=csv,noheader,nounits",
    )
    status, stdout, stderr = run_captured(command, timeout_seconds=600)
    if status != 0 or stderr:
        raise FixtureContractError("the exact-base GPU preflight failed")
    try:
        rows = stdout.decode("utf-8", errors="strict").strip().splitlines()
    except UnicodeDecodeError as exc:
        raise FixtureContractError("the exact-base GPU preflight is invalid") from exc
    if len(rows) != 1:
        raise FixtureContractError(
            "each hardware selector must resolve to exactly one GPU"
        )
    fields = tuple(part.strip() for part in rows[0].split(","))
    if (
        len(fields) != 2
        or _GPU_UUID_RE.fullmatch(fields[0]) is None
        or _COMPUTE_CAPABILITY_RE.fullmatch(fields[1]) is None
    ):
        raise FixtureContractError("the exact-base GPU preflight is invalid")
    return fields


def build_candidate_image(
    *,
    engine: str,
    image: str,
    source_date_epoch: str,
    run_captured: Callable[..., tuple[int, bytes, bytes]],
    inspect_image: InspectImage,
) -> dict[str, object]:
    """Build the hard-pinned candidate with an argv-only container invocation."""

    engine = validate_container_engine(engine)
    image = validate_candidate_tag(image)
    source_date_epoch = validate_source_date_epoch(source_date_epoch)
    repository = Path(__file__).resolve().parents[2]
    dockerfile = repository / "runtime" / "Dockerfile.tensorrt-llm"
    command = (
        engine,
        "buildx",
        "build",
        "--load",
        "--provenance=false",
        "--build-arg",
        f"SOURCE_DATE_EPOCH={source_date_epoch}",
        "-f",
        str(dockerfile),
        "-t",
        image,
        str(repository),
    )
    status, _stdout, _stderr = run_captured(command, timeout_seconds=7200)
    if status != 0:
        raise FixtureContractError("the TensorRT-LLM candidate image build failed")
    digest = inspect_image(engine, image)
    return {
        "candidate_image_digest": digest,
        "format_version": BUILD_FORMAT,
        "ok": True,
    }


def smoke_candidate_image(
    *,
    engine: str,
    image: str,
    selector: str,
    run_captured: Callable[..., tuple[int, bytes, bytes]],
    inspect_image: InspectImage,
) -> dict[str, object]:
    """Smoke the inspected immutable candidate through a fixed container argv."""

    selector = validate_smoke_selector(selector)
    digest = inspect_image(engine, image)
    script = (
        "/opt/invarlock/bin/tensorrt-llm-runner --invarlock-runtime-info-v1; "
        "/opt/invarlock/cli-venv/bin/invarlock advanced runtime-behavior --help "
        ">/dev/null; /opt/invarlock/cli-venv/bin/invarlock advanced plugins "
        "runtime-providers --json >/dev/null; /opt/invarlock/bin/vendor-python -c "
        '"import importlib.metadata as m, os, torch, invarlock, tensorrt_llm; '
        "assert os.environ.get('TRT_LLM_VERSION') == '1.2.1'; "
        "assert m.version('tensorrt_llm') == '1.2.1'; "
        "assert os.environ.get('NVIDIA_VISIBLE_DEVICES') not in "
        "{None, '', 'void'}; assert torch.cuda.is_available(), "
        "'CUDA is not visible'; print('TensorRT-LLM runtime contract ok')\""
    )
    command = (
        engine,
        "run",
        "--rm",
        "--gpus",
        selector,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec",
        "--env",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "--entrypoint",
        "/bin/sh",
        digest,
        "-ec",
        script,
    )
    status, _stdout, _stderr = run_captured(command, timeout_seconds=300)
    if status != 0:
        raise FixtureContractError("the TensorRT-LLM candidate smoke failed")
    return {
        "candidate_image_digest": digest,
        "format_version": SMOKE_FORMAT,
        "ok": True,
    }
