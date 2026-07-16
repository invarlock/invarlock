"""Digest-pinned OCI execution for independently isolated evaluation sides."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from invarlock.core.evaluation_request import (
    MAX_EVALUATION_REQUEST_BYTES,
    EvaluationRequest,
    _load_yaml,
    _reject_include_directives,
    _validate_schema,
)
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import RuntimeArtifactResources, RuntimeProvider
from invarlock.evaluation_run import EvaluationRunResult, load_runtime_side_evidence
from invarlock.evaluation_runtime import (
    CallerRuntimeResources,
    ProviderResourceBinding,
    RuntimeSideRole,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import StrictJsonError, read_regular_file_bytes
from invarlock.runtime_provider_evidence import decode_runtime_provider_receipt
from invarlock.runtime_security_helpers import (
    ALLOW_NETWORK_ENV,
    ALLOW_REMOTE_CODE_ENV,
    ALLOW_THIRD_PARTY_PLUGINS_ENV,
    CONTAINER_EXECUTION_ENV,
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
)

type ContainerEngine = Literal["docker", "podman"]
type EntrypointProfile = Literal["auto", "python", "nvidia"]

CONTAINER_ENGINE_ENV = "INVARLOCK_CONTAINER_ENGINE"
RUNTIME_DEVICE_ENV = "INVARLOCK_RUNTIME_DEVICE"
BASELINE_RUNTIME_DEVICE_ENV = "INVARLOCK_BASELINE_RUNTIME_DEVICE"
SUBJECT_RUNTIME_DEVICE_ENV = "INVARLOCK_SUBJECT_RUNTIME_DEVICE"
BASELINE_RUNTIME_IMAGE_ENV = "INVARLOCK_BASELINE_RUNTIME_IMAGE"
SUBJECT_RUNTIME_IMAGE_ENV = "INVARLOCK_SUBJECT_RUNTIME_IMAGE"
BASELINE_RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_BASELINE_RUNTIME_IMAGE_DIGEST"
SUBJECT_RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_SUBJECT_RUNTIME_IMAGE_DIGEST"
RUNTIME_ENTRYPOINT_ENV = "INVARLOCK_RUNTIME_ENTRYPOINT"
BASELINE_RUNTIME_ENTRYPOINT_ENV = "INVARLOCK_BASELINE_RUNTIME_ENTRYPOINT"
SUBJECT_RUNTIME_ENTRYPOINT_ENV = "INVARLOCK_SUBJECT_RUNTIME_ENTRYPOINT"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(0|[1-9][0-9]*))?$")
_MAX_WORKER_DIAGNOSTIC_BYTES = 64 * 1024
_ROLES: tuple[RuntimeSideRole, RuntimeSideRole] = ("baseline", "subject")
_PROVIDER_BINDINGS: Mapping[str, tuple[str, Mapping[str, str]]] = {
    "hf_vision_text": (
        "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT",
        {
            "content_store": "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE",
        },
    ),
    "llama_cpp": (
        "INVARLOCK_GGUF_RESOURCE_ROOT",
        {
            "backend_executable": "INVARLOCK_GGUF_BACKEND_EXECUTABLE",
            "backend_source": "INVARLOCK_GGUF_BACKEND_SOURCE",
        },
    ),
    "tensorrt_llm": (
        "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT",
        {
            "tokenizer_contract": "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT",
            "runner_executable": "INVARLOCK_TENSORRT_LLM_RUNNER_EXECUTABLE",
        },
    ),
}


class OciEvaluationError(ValueError):
    """Raised before or after an unsafe, ambiguous, or failed side launch."""


def evaluation_request_execution_mode(path: Path) -> str | None:
    """Read only the schema-validated execution discriminator for CLI dispatch."""

    try:
        payload = read_regular_file_bytes(
            Path(path),
            label="evaluation request",
            max_bytes=MAX_EVALUATION_REQUEST_BYTES,
        )
        value = _load_yaml(payload)
        _reject_include_directives(value)
        value = _validate_schema(value)
    except (OSError, StrictJsonError, ValueError):
        return None
    execution = value.get("execution")
    if not isinstance(execution, dict):
        return None
    mode = execution.get("mode")
    return mode if mode in {"run", "import"} else None


@dataclass(frozen=True)
class OciSideLaunch:
    """Caller-owned OCI binding for exactly one comparison side."""

    image_ref: str
    image_digest: str
    device: str
    entrypoint: EntrypointProfile = "auto"


@dataclass(frozen=True)
class OciEvaluationLaunch:
    """Closed launch policy for two independently pinned runtime images."""

    engine: ContainerEngine
    baseline: OciSideLaunch
    subject: OciSideLaunch
    engine_path: str | None = None


def _local_image_id(engine_path: str, image: str) -> str:
    """Resolve one local tag without allowing it to remain the execution identity."""

    try:
        completed = subprocess.run(
            [engine_path, "image", "inspect", "--format", "{{.Id}}", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OciEvaluationError(
            "runtime image could not be inspected locally"
        ) from exc
    observed = completed.stdout.strip()
    if completed.returncode != 0 or _DIGEST_RE.fullmatch(observed) is None:
        diagnostic = completed.stderr.strip()[:512] or "image is unavailable"
        raise OciEvaluationError(f"runtime image could not be inspected: {diagnostic}")
    return observed


def _portable_image_ref(image_ref: str, image_digest: str, *, engine_path: str) -> str:
    image = image_ref.strip()
    digest = image_digest.strip()
    if _DIGEST_RE.fullmatch(digest) is None:
        raise OciEvaluationError(
            "runtime image digest: INVARLOCK_RUNTIME_IMAGE_DIGEST or its per-side "
            "override must use lowercase sha256:<64 hex>"
        )
    if (
        not image
        or image != image_ref
        or image.startswith(("-", "/", "\\"))
        or any(ord(character) < 32 or character.isspace() for character in image)
        or image.count("@") > 1
    ):
        raise OciEvaluationError("runtime image must be a portable OCI reference")
    if image == digest:
        return image
    if _DIGEST_RE.fullmatch(image) is not None:
        raise OciEvaluationError(
            "runtime image reference and supplied digest do not agree"
        )
    if "@" in image:
        repository, embedded = image.rsplit("@", 1)
        if not repository or embedded != digest:
            raise OciEvaluationError(
                "runtime image reference and supplied digest do not agree"
            )
        return image
    observed = _local_image_id(engine_path, image)
    if observed != digest:
        raise OciEvaluationError(
            "runtime image reference and supplied digest do not agree"
        )
    return digest


def _embedded_digest(image: str) -> str:
    if _DIGEST_RE.fullmatch(image) is not None:
        return image
    return image.rsplit("@", 1)[1] if "@" in image else ""


def _engine_binary(engine: str) -> tuple[ContainerEngine, str]:
    if engine not in {"docker", "podman"}:
        raise OciEvaluationError("container engine must be docker or podman")
    executable = shutil.which(engine)
    if executable is None:
        raise OciEvaluationError(f"container engine {engine!r} is not available")
    return cast(ContainerEngine, engine), str(Path(executable).resolve())


def _device(value: str, *, label: str) -> str:
    device = value.strip().lower()
    if device == "cpu" or _CUDA_DEVICE_RE.fullmatch(device) is not None:
        return device
    raise OciEvaluationError(f"{label} must be cpu, cuda, or cuda:<index>")


def _entrypoint(value: str, *, label: str) -> EntrypointProfile:
    profile = value.strip().lower()
    if profile not in {"auto", "python", "nvidia"}:
        raise OciEvaluationError(f"{label} must be auto, python, or nvidia")
    return cast(EntrypointProfile, profile)


def launch_from_environment(
    *,
    engine: str | None = None,
    image_ref: str | None = None,
    image_digest: str | None = None,
    baseline_image_ref: str | None = None,
    baseline_image_digest: str | None = None,
    subject_image_ref: str | None = None,
    subject_image_digest: str | None = None,
    default_device: str | None = None,
    baseline_device: str | None = None,
    subject_device: str | None = None,
    runtime_entrypoint: str | None = None,
    baseline_entrypoint: str | None = None,
    subject_entrypoint: str | None = None,
) -> OciEvaluationLaunch:
    """Resolve common defaults and optional per-side OCI bindings."""

    selected_engine, engine_path = _engine_binary(
        engine or os.environ.get(CONTAINER_ENGINE_ENV, "docker")
    )

    common_image = image_ref or os.environ.get(RUNTIME_IMAGE_ENV, "")
    common_digest = image_digest or os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    common_device = _device(
        default_device or os.environ.get(RUNTIME_DEVICE_ENV, "cpu"),
        label="runtime device",
    )
    common_entrypoint = runtime_entrypoint or os.environ.get(
        RUNTIME_ENTRYPOINT_ENV, "auto"
    )

    def side_launch(
        *,
        role: RuntimeSideRole,
        explicit_image: str | None,
        image_variable: str,
        explicit_digest: str | None,
        digest_variable: str,
        explicit_device: str | None,
        device_variable: str,
        explicit_entrypoint: str | None,
        entrypoint_variable: str,
    ) -> OciSideLaunch:
        selected_image = (
            explicit_image or os.environ.get(image_variable) or common_image
        )
        selected_digest = (
            explicit_digest
            or os.environ.get(digest_variable)
            or _embedded_digest(selected_image)
            or common_digest
        )
        return OciSideLaunch(
            image_ref=_portable_image_ref(
                selected_image, selected_digest, engine_path=engine_path
            ),
            image_digest=selected_digest,
            device=_device(
                explicit_device or os.environ.get(device_variable) or common_device,
                label=f"{role} runtime device",
            ),
            entrypoint=_entrypoint(
                explicit_entrypoint
                or os.environ.get(entrypoint_variable)
                or common_entrypoint,
                label=f"{role} runtime entrypoint",
            ),
        )

    return OciEvaluationLaunch(
        engine=selected_engine,
        engine_path=engine_path,
        baseline=side_launch(
            role="baseline",
            explicit_image=baseline_image_ref,
            image_variable=BASELINE_RUNTIME_IMAGE_ENV,
            explicit_digest=baseline_image_digest,
            digest_variable=BASELINE_RUNTIME_IMAGE_DIGEST_ENV,
            explicit_device=baseline_device,
            device_variable=BASELINE_RUNTIME_DEVICE_ENV,
            explicit_entrypoint=baseline_entrypoint,
            entrypoint_variable=BASELINE_RUNTIME_ENTRYPOINT_ENV,
        ),
        subject=side_launch(
            role="subject",
            explicit_image=subject_image_ref,
            image_variable=SUBJECT_RUNTIME_IMAGE_ENV,
            explicit_digest=subject_image_digest,
            digest_variable=SUBJECT_RUNTIME_IMAGE_DIGEST_ENV,
            explicit_device=subject_device,
            device_variable=SUBJECT_RUNTIME_DEVICE_ENV,
            explicit_entrypoint=subject_entrypoint,
            entrypoint_variable=SUBJECT_RUNTIME_ENTRYPOINT_ENV,
        ),
    )


def _directory_mount_source(path: Path, *, label: str) -> Path:
    try:
        entry = path.lstat()
    except OSError as exc:
        raise OciEvaluationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise OciEvaluationError(f"{label} must be a directory, not a symlink")
    resolved = path.resolve(strict=True)
    if "," in str(resolved) or any(ord(character) < 32 for character in str(resolved)):
        raise OciEvaluationError(f"{label} path cannot be represented as an OCI mount")
    return resolved


def _artifact_mount_source(path: Path, *, label: str) -> Path:
    try:
        entry = path.lstat()
    except OSError as exc:
        raise OciEvaluationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(entry.st_mode) or not (
        stat.S_ISREG(entry.st_mode) or stat.S_ISDIR(entry.st_mode)
    ):
        raise OciEvaluationError(
            f"{label} must be a regular file or directory, not a symlink"
        )
    resolved = path.resolve(strict=True)
    if "," in str(resolved) or any(ord(character) < 32 for character in str(resolved)):
        raise OciEvaluationError(f"{label} path cannot be represented as an OCI mount")
    return resolved


def _mount(source: Path, target: str, *, read_only: bool) -> list[str]:
    fields = ["type=bind", f"source={source}", f"target={target}"]
    if read_only:
        fields.append("readonly")
    return ["--mount", ",".join(fields)]


def _provider_bindings(
    environment: Mapping[str, str],
) -> dict[str, ProviderResourceBinding]:
    bindings: dict[str, ProviderResourceBinding] = {}
    for provider_name, (root_variable, support_variables) in _PROVIDER_BINDINGS.items():
        root_value = environment.get(root_variable)
        support = {
            name: value
            for name, variable in support_variables.items()
            if (value := environment.get(variable)) is not None
        }
        if root_value is None:
            if support:
                raise OciEvaluationError(
                    f"{root_variable} is required when provider support is configured"
                )
            continue
        missing = sorted(set(support_variables) - set(support))
        if missing:
            raise OciEvaluationError(
                f"{support_variables[missing[0]]} is required for the configured provider"
            )
        root = _directory_mount_source(
            Path(root_value), label=f"{provider_name} resource root"
        )
        bindings[provider_name] = ProviderResourceBinding(
            root=root, support_resources=support
        )
    return bindings


def _gpu_arguments(engine: ContainerEngine, device: str) -> list[str]:
    if device == "cpu":
        return []
    index = device.split(":", 1)[1] if ":" in device else None
    if engine == "docker":
        return ["--gpus", "all" if index is None else f"device={index}"]
    return [
        "--device",
        "nvidia.com/gpu=all" if index is None else f"nvidia.com/gpu={index}",
    ]


def _inner_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


def _workers_may_run_parallel(launch: OciEvaluationLaunch) -> bool:
    """Allow concurrency only when the two workers cannot contend for one GPU."""

    baseline = launch.baseline.device
    subject = launch.subject.device
    if baseline == subject == "cpu":
        return True
    if baseline.startswith("cuda:") and subject.startswith("cuda:"):
        return baseline.split(":", 1)[1] != subject.split(":", 1)[1]
    return False


def compose_side_worker_command(
    *,
    launch: OciEvaluationLaunch,
    side_launch: OciSideLaunch,
    provider_name: str,
    artifact_source: Path,
    support_sources: Mapping[str, Path],
    job_root: Path,
    output_root: Path | None = None,
) -> list[str]:
    """Compose one argv-only model-worker launch with no signing-key mount."""

    profile = side_launch.entrypoint
    if profile == "auto":
        profile = "nvidia" if provider_name == "tensorrt_llm" else "python"
    output_parent = job_root / "output-root" if output_root is None else output_root
    command = [
        launch.engine_path or launch.engine,
        "run",
        "--rm",
        "--init",
        "--pull=never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "1024",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=4g",
        *_gpu_arguments(launch.engine, side_launch.device),
        *_mount(
            _artifact_mount_source(artifact_source, label="side artifact"),
            "/invarlock-resources/artifact",
            read_only=True,
        ),
        *_mount(job_root, "/invarlock/job", read_only=True),
        *_mount(output_parent, "/invarlock/output-root", read_only=False),
        "-e",
        f"{CONTAINER_EXECUTION_ENV}=1",
        "-e",
        f"{RUNTIME_IMAGE_ENV}={side_launch.image_ref}",
        "-e",
        f"{RUNTIME_IMAGE_DIGEST_ENV}={side_launch.image_digest}",
        "-e",
        f"{ALLOW_NETWORK_ENV}=0",
        "-e",
        f"{ALLOW_REMOTE_CODE_ENV}=0",
        "-e",
        f"{ALLOW_THIRD_PARTY_PLUGINS_ENV}=0",
        "-e",
        "HF_HOME=/tmp/huggingface",
    ]
    for index, (_name, source) in enumerate(sorted(support_sources.items())):
        command.extend(
            _mount(
                _artifact_mount_source(source, label="side support resource"),
                f"/invarlock-resources/support-{index}",
                read_only=True,
            )
        )
    if profile == "nvidia":
        command.extend(
            [
                "--entrypoint",
                "/opt/nvidia/nvidia_entrypoint.sh",
                side_launch.image_ref,
                "/opt/invarlock/cli-venv/bin/python",
            ]
        )
    else:
        command.extend(["--entrypoint", "python", side_launch.image_ref])
    command.extend(
        ["-m", "invarlock.evaluation_side_worker", "/invarlock/job/job.json"]
    )
    return command


def _read_bounded_stream(stream: object, destination: bytearray) -> None:
    """Drain one worker pipe while retaining only bounded diagnostics."""

    reader = getattr(stream, "read", None)
    if not callable(reader):
        return
    while chunk := reader(8192):
        if len(destination) < _MAX_WORKER_DIAGNOSTIC_BYTES:
            destination.extend(chunk[: _MAX_WORKER_DIAGNOSTIC_BYTES - len(destination)])


def run_side_worker(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one precomposed worker without a shell or unbounded log capture."""

    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise OciEvaluationError(
            f"container engine could not be started: {exc}"
        ) from exc
    stdout = bytearray()
    stderr = bytearray()
    assert process.stdout is not None
    assert process.stderr is not None
    drains = (
        threading.Thread(
            target=_read_bounded_stream,
            args=(process.stdout, stdout),
            name="invarlock-worker-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded_stream,
            args=(process.stderr, stderr),
            name="invarlock-worker-stderr",
            daemon=True,
        ),
    )
    for drain in drains:
        drain.start()
    returncode = process.wait()
    for drain in drains:
        drain.join()
    return subprocess.CompletedProcess(
        list(command),
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


@dataclass(frozen=True)
class OciRuntimeExecutor:
    """Host-side coordinator for two isolated model workers."""

    launch: OciEvaluationLaunch
    environment: Mapping[str, str] | None = None

    def execute(
        self,
        request: EvaluationRequest,
        *,
        registry: CoreRegistry,
        schedule_bytes: bytes,
        policy_digest: str,
    ) -> EvaluationRunResult:
        observed_environment = (
            os.environ if self.environment is None else self.environment
        )
        provider_bindings = _provider_bindings(observed_environment)
        sides = {
            "baseline": (request.comparison.baseline, self.launch.baseline),
            "subject": (request.comparison.subject, self.launch.subject),
        }
        with tempfile.TemporaryDirectory(prefix="invarlock-side-workers-") as temporary:
            root = Path(temporary)
            commands: dict[RuntimeSideRole, list[str]] = {}
            outputs: dict[RuntimeSideRole, Path] = {}
            for role, (side, side_launch) in sides.items():
                provider: RuntimeProvider = registry.get_runtime_provider(
                    side.runtime.provider
                )
                resolver = CallerRuntimeResources(
                    container_image_digest=side_launch.image_digest,
                    default_device=_inner_device(side_launch.device),
                    provider_bindings=provider_bindings,
                )
                resources: RuntimeArtifactResources = resolver.resolve(
                    request_root=request.root,
                    role=cast(RuntimeSideRole, role),
                    side=side,
                    provider=provider,
                )
                job_root = root / role
                job_root.mkdir()
                job_root.joinpath("schedule.json").write_bytes(schedule_bytes)
                output_parent = root / f"{role}-output"
                output_parent.mkdir()
                output = output_parent / "side"
                projected_support = {
                    name: f"support-{index}"
                    for index, name in enumerate(sorted(resources.support_resources))
                }
                support_sources = {
                    name: resources.support_path(name)
                    for name in sorted(resources.support_resources)
                }
                job = {
                    "format_version": "invarlock/runtime-side-job-v1",
                    "role": role,
                    "provider": provider.name,
                    "model_id": side.artifact.model_id,
                    "settings": dict(side.runtime.settings),
                    "metric": request.comparison.collection_metric,
                    "policy_digest": policy_digest,
                    "resource_root": "/invarlock-resources",
                    "primary_artifact": "artifact",
                    "support_resources": projected_support,
                    "device_kind": _inner_device(side_launch.device),
                    "image_digest": side_launch.image_digest,
                    "schedule": "/invarlock/job/schedule.json",
                    "output": "/invarlock/output-root/side",
                }
                job_root.joinpath("job.json").write_bytes(canonical_json_bytes(job))
                commands[cast(RuntimeSideRole, role)] = compose_side_worker_command(
                    launch=self.launch,
                    side_launch=side_launch,
                    provider_name=provider.name,
                    artifact_source=resources.primary_path(),
                    support_sources=support_sources,
                    job_root=job_root,
                    output_root=output_parent,
                )
                outputs[cast(RuntimeSideRole, role)] = output

            completed: dict[RuntimeSideRole, subprocess.CompletedProcess[str]] = {}
            if _workers_may_run_parallel(self.launch):
                with ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="invarlock-side"
                ) as pool:
                    futures = {
                        role: pool.submit(run_side_worker, command)
                        for role, command in commands.items()
                    }
                    for role in _ROLES:
                        completed[role] = futures[role].result()
            else:
                for role in _ROLES:
                    completed[role] = run_side_worker(commands[role])
            failures = [
                f"{role} worker exited with status {completed[role].returncode}: "
                + (completed[role].stderr.strip() or "no diagnostic")
                for role in _ROLES
                if completed[role].returncode != 0
            ]
            if failures:
                raise OciEvaluationError("; ".join(failures))
            evidence = {
                role: load_runtime_side_evidence(outputs[role]) for role in _ROLES
            }
            digests = {
                role: decode_runtime_provider_receipt(
                    evidence[role].provider_receipt
                ).outer_image_digest
                for role in _ROLES
            }
            for role in _ROLES:
                expected = getattr(self.launch, role).image_digest
                if digests[role] != expected:
                    raise OciEvaluationError(
                        f"{role} worker receipt does not bind its selected runtime image"
                    )
            return EvaluationRunResult(
                baseline=evidence["baseline"],
                subject=evidence["subject"],
                baseline_runtime_digest=cast(str, digests["baseline"]),
                subject_runtime_digest=cast(str, digests["subject"]),
            )


__all__ = [
    "BASELINE_RUNTIME_DEVICE_ENV",
    "BASELINE_RUNTIME_ENTRYPOINT_ENV",
    "BASELINE_RUNTIME_IMAGE_DIGEST_ENV",
    "BASELINE_RUNTIME_IMAGE_ENV",
    "CONTAINER_ENGINE_ENV",
    "OciEvaluationError",
    "OciEvaluationLaunch",
    "OciRuntimeExecutor",
    "OciSideLaunch",
    "RUNTIME_DEVICE_ENV",
    "RUNTIME_ENTRYPOINT_ENV",
    "SUBJECT_RUNTIME_DEVICE_ENV",
    "SUBJECT_RUNTIME_ENTRYPOINT_ENV",
    "SUBJECT_RUNTIME_IMAGE_DIGEST_ENV",
    "SUBJECT_RUNTIME_IMAGE_ENV",
    "compose_side_worker_command",
    "evaluation_request_execution_mode",
    "launch_from_environment",
    "run_side_worker",
]
