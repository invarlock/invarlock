from __future__ import annotations

import functools
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml

import invarlock.evaluation_oci as evaluation_oci
from invarlock.core.evaluation_request import (
    ArtifactRequest,
    ComparisonRequest,
    ComparisonSideRequest,
    EvaluationRequest,
    ExecutionRequest,
    OutputRequest,
    RuntimeRequest,
)
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import RuntimeProvider
from invarlock.evaluation_oci import (
    OciEvaluationError,
    OciEvaluationLaunch,
    OciRuntimeExecutor,
    OciSideLaunch,
    OciWorkerLimits,
    _workers_may_run_parallel,
    compose_side_worker_command,
    evaluation_request_execution_mode,
    launch_from_environment,
    preflight_oci_launch,
)
from invarlock.evaluation_run import load_runtime_side_evidence
from invarlock.evidence_pack import RuntimeSideEvidence
from invarlock.evidence_pack_contract import schedule_bytes as canonical_schedule_bytes
from tests.evidence_packs.test_evidence_pack import _schedule

_BASELINE_DIGEST = "sha256:" + "a" * 64
_SUBJECT_DIGEST = "sha256:" + "b" * 64
_BASELINE_CONFIG_ID = "sha256:" + "c" * 64
_SUBJECT_CONFIG_ID = "sha256:" + "d" * 64


def _inspection(
    *, config_id: str, repo_digests: tuple[str, ...] = ()
) -> evaluation_oci._LocalImageInspection:  # noqa: SLF001
    return evaluation_oci._LocalImageInspection(  # noqa: SLF001
        config_id=config_id,
        repo_digests=repo_digests,
    )


def _tag_inspection(
    image: str, digest: str, *, config_id: str
) -> evaluation_oci._LocalImageInspection:  # noqa: SLF001
    repository = (
        image.rsplit("@", 1)[0]
        if "@" in image
        else evaluation_oci._tag_repository(image)  # noqa: SLF001
    )
    return _inspection(
        config_id=config_id,
        repo_digests=(f"{repository}@{digest}",),
    )


def _request_file(path: Path, *, mode: str = "run") -> Path:
    side = {
        "artifact": {
            "path": "models/model",
            "model_id": "local/model",
            "locator": "artifact:local-model",
        },
        "runtime": {"provider": "hf_transformers", "settings": {}},
    }
    dataset: object = {
        "path": "inputs/records.jsonl",
        "sha256": "a" * 64,
        "format": "jsonl",
        "name": "local-test",
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "expected",
        "id_field": "id",
    }
    execution: dict[str, object] = {"mode": mode}
    if mode == "import":
        side["artifact"].pop("path")
        dataset = "inputs/schedule.json"
        imported = {
            "identity": "imports/identity.json",
            "receipt": "imports/receipt.json",
            "observation": "imports/observation.json",
            "run_report": "imports/report.json",
            "runtime_manifest": "imports/runtime.manifest.json",
            "runtime_config": "imports/runtime.json",
        }
        execution.update(
            {
                "records": "imports/records.json",
                "schedule": "inputs/schedule.json",
                "baseline": json.loads(json.dumps(imported)),
                "subject": json.loads(json.dumps(imported)),
            }
        )
    payload = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": json.loads(json.dumps(side)),
            "subject": json.loads(json.dumps(side)),
            "dataset": dataset,
            "policy": "inputs/policy.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": execution,
        "output": {"evidence": "artifacts/evidence"},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _side(root: Path, name: str, provider: str) -> ComparisonSideRequest:
    artifact = root / "models" / name
    artifact.mkdir(parents=True)
    return ComparisonSideRequest(
        artifact=ArtifactRequest(
            path=artifact,
            model_id=f"local/{name}",
            locator=f"artifact:{name}",
        ),
        runtime=RuntimeRequest(
            provider=provider,
            settings={"batch_size": 1, "timeout_seconds": 30},
        ),
    )


def _request(root: Path) -> EvaluationRequest:
    return EvaluationRequest(
        format_version="invarlock/evaluation-request-v1",
        root=root,
        comparison=ComparisonRequest(
            baseline=_side(root, "baseline", "hf_transformers"),
            subject=_side(root, "subject", "tensorrt_llm"),
            dataset=root / "dataset.jsonl",
            policy=root / "policy.json",
            task="text_causal",
            metric="exact_match",
        ),
        execution=ExecutionRequest(
            mode="run",
            records=None,
            schedule=None,
            baseline=None,
            subject=None,
        ),
        output=OutputRequest(evidence=root / "evidence"),
    )


def _launch() -> OciEvaluationLaunch:
    return OciEvaluationLaunch(
        engine="docker",
        baseline=OciSideLaunch(
            image_ref=f"registry.example/hf@{_BASELINE_DIGEST}",
            image_digest=_BASELINE_DIGEST,
            device="cpu",
        ),
        subject=OciSideLaunch(
            image_ref=f"registry.example/trt@{_SUBJECT_DIGEST}",
            image_digest=_SUBJECT_DIGEST,
            device="cuda:3",
        ),
    )


def test_preflight_inspects_pinned_images_without_starting_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch()
    observed: list[str] = []

    def inspect_image(_engine: str, image: str) -> evaluation_oci._LocalImageInspection:  # noqa: SLF001
        observed.append(image)
        return _inspection(
            config_id=(_BASELINE_CONFIG_ID if "hf" in image else _SUBJECT_CONFIG_ID),
            repo_digests=(image,),
        )

    monkeypatch.setattr(evaluation_oci, "_inspect_local_image", inspect_image)
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        evaluation_oci.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("container command must not run"),
    )

    assert preflight_oci_launch(launch) == {
        "baseline": _BASELINE_DIGEST,
        "subject": _SUBJECT_DIGEST,
    }
    assert observed == [launch.baseline.image_ref, launch.subject.image_ref]


def test_executor_resolves_each_side_with_its_exact_image_and_device(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    executor = OciRuntimeExecutor(_launch(), environment={})
    registry = CoreRegistry()

    baseline = executor.resolve(
        request_root=request.root,
        role="baseline",
        side=request.comparison.baseline,
        provider=registry.get_runtime_provider("hf_transformers"),
    )
    subject_provider = SimpleNamespace(name="tensorrt_llm")
    subject_root = tmp_path / "subject-resources"
    subject_root.mkdir()
    subject_artifact = request.comparison.subject.artifact.path
    assert subject_artifact is not None
    tokenizer_contract = tmp_path / "tokenizer-contract.json"
    tokenizer_contract.write_text("{}\n", encoding="utf-8")
    environment = {
        "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT": str(tmp_path),
        "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT": tokenizer_contract.name,
    }
    executor = OciRuntimeExecutor(_launch(), environment=environment)
    subject = executor.resolve(
        request_root=request.root,
        role="subject",
        side=request.comparison.subject,
        provider=cast(RuntimeProvider, subject_provider),
    )

    assert baseline.container_image_digest == _BASELINE_DIGEST
    assert baseline.device_kind == "cpu"
    assert subject.container_image_digest == _SUBJECT_DIGEST
    assert subject.device_kind == "cuda"


def test_executor_snapshots_caller_owned_resource_environment(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "model").mkdir(parents=True)
        (root / "images").mkdir()
    environment = {
        "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT": str(first),
        "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE": "images",
    }
    executor = OciRuntimeExecutor(_launch(), environment=environment)
    assert executor.environment is environment
    environment["INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT"] = str(second)
    side = ComparisonSideRequest(
        artifact=ArtifactRequest(
            path=first / "model",
            model_id="local/vision",
            locator="artifact:vision",
        ),
        runtime=RuntimeRequest(provider="hf_vision_text", settings={}),
    )

    resources = executor.resolve(
        request_root=tmp_path,
        role="baseline",
        side=side,
        provider=cast(RuntimeProvider, SimpleNamespace(name="hf_vision_text")),
    )

    assert resources.root == first
    assert resources.support_path("content_store") == first / "images"


def test_execution_mode_dispatch_uses_strict_bounded_yaml(tmp_path: Path) -> None:
    assert (
        evaluation_request_execution_mode(_request_file(tmp_path / "run.yaml")) == "run"
    )
    assert (
        evaluation_request_execution_mode(
            _request_file(tmp_path / "import.yaml", mode="import")
        )
        == "import"
    )
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("execution:\n  mode: run\nexecution:\n  mode: import\n")
    assert evaluation_request_execution_mode(duplicate) is None


def test_launch_has_ergonomic_same_image_default_and_independent_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, image: _tag_inspection(
            image,
            _SUBJECT_DIGEST if "trt" in image else _BASELINE_DIGEST,
            config_id=(_SUBJECT_CONFIG_ID if "trt" in image else _BASELINE_CONFIG_ID),
        ),
    )
    common = launch_from_environment(
        engine="docker",
        image_ref="registry.example/runtime:local",
        image_digest=_BASELINE_DIGEST,
    )
    assert common.baseline.image_ref == common.subject.image_ref
    assert common.baseline.image_digest == common.subject.image_digest

    split = launch_from_environment(
        engine="podman",
        image_ref="registry.example/hf:local",
        image_digest=_BASELINE_DIGEST,
        subject_image_ref="registry.example/trt:local",
        subject_image_digest=_SUBJECT_DIGEST,
        baseline_device="cpu",
        subject_device="cuda:2",
    )
    assert split.baseline.image_digest == _BASELINE_DIGEST
    assert split.subject.image_digest == _SUBJECT_DIGEST
    assert split.baseline.image_ref == f"registry.example/hf@{_BASELINE_DIGEST}"
    assert split.subject.image_ref == f"registry.example/trt@{_SUBJECT_DIGEST}"
    assert split.subject.device == "cuda:2"

    bounded = launch_from_environment(
        engine="docker",
        image_ref="registry.example/hf:local",
        image_digest=_BASELINE_DIGEST,
        runtime_cpus="7.5",
        runtime_memory_mib="16384",
        runtime_user="12001:12001",
    )
    assert bounded.worker_limits == OciWorkerLimits(
        cpus="7.5", memory_mib=16384, user="12001:12001"
    )

    embedded_side = launch_from_environment(
        engine="docker",
        image_ref="registry.example/hf:local",
        image_digest=_BASELINE_DIGEST,
        subject_image_ref=f"registry.example/trt@{_SUBJECT_DIGEST}",
    )
    assert embedded_side.subject.image_digest == _SUBJECT_DIGEST

    with pytest.raises(OciEvaluationError, match="digest"):
        launch_from_environment(
            engine="docker", image_ref="registry.example/runtime:latest"
        )

    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, image: _tag_inspection(
            image, _SUBJECT_DIGEST, config_id=_SUBJECT_CONFIG_ID
        ),
    )
    with pytest.raises(OciEvaluationError, match="supplied repository manifest"):
        launch_from_environment(
            engine="docker",
            image_ref="registry.example/runtime:local",
            image_digest=_BASELINE_DIGEST,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"runtime_cpus": ""}, "runtime CPU limit"),
        ({"runtime_memory_mib": ""}, "runtime memory limit"),
        ({"runtime_user": ""}, "runtime user"),
    ],
)
def test_launch_rejects_explicit_empty_worker_limit_overrides(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, image: _tag_inspection(
            image, _BASELINE_DIGEST, config_id=_BASELINE_CONFIG_ID
        ),
    )

    with pytest.raises(OciEvaluationError, match=message):
        launch_from_environment(
            engine="docker",
            image_ref="registry.example/runtime:local",
            image_digest=_BASELINE_DIGEST,
            **override,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        (evaluation_oci.RUNTIME_CPUS_ENV, "runtime CPU limit"),
        (evaluation_oci.RUNTIME_MEMORY_MIB_ENV, "runtime memory limit"),
        (evaluation_oci.RUNTIME_USER_ENV, "runtime user"),
    ],
)
def test_launch_rejects_empty_worker_limit_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    message: str,
) -> None:
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, image: _tag_inspection(
            image, _BASELINE_DIGEST, config_id=_BASELINE_CONFIG_ID
        ),
    )
    monkeypatch.setenv(variable, "")

    with pytest.raises(OciEvaluationError, match=message):
        launch_from_environment(
            engine="docker",
            image_ref="registry.example/runtime:local",
            image_digest=_BASELINE_DIGEST,
        )


@pytest.mark.parametrize(
    ("payload", "expected_config_id"),
    [
        (
            [
                {
                    "Id": _BASELINE_CONFIG_ID,
                    "RepoDigests": [
                        f"registry.example/other@{_SUBJECT_DIGEST}",
                        f"registry.example/runtime@{_BASELINE_DIGEST}",
                    ],
                }
            ],
            _BASELINE_CONFIG_ID,
        ),
        (
            {
                "Id": _BASELINE_CONFIG_ID.removeprefix("sha256:"),
                "RepoDigests": [f"localhost/runtime@{_BASELINE_DIGEST}"],
            },
            _BASELINE_CONFIG_ID,
        ),
    ],
)
def test_image_inspection_parses_bounded_docker_and_podman_json_with_argv_only(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    expected_config_id: str,
) -> None:
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(evaluation_oci.subprocess, "run", run)
    image = "registry.example:5000/team/runtime:candidate"

    inspected = evaluation_oci._inspect_local_image(  # noqa: SLF001
        "/usr/bin/container-engine",
        image,
    )

    assert inspected.config_id == expected_config_id
    assert observed["command"] == [
        "/usr/bin/container-engine",
        "image",
        "inspect",
        image,
    ]
    keywords = cast(dict[str, object], observed["kwargs"])
    assert keywords["shell"] is False
    assert keywords["timeout"] == 30


def test_tag_resolution_selects_matching_repository_manifest_not_first_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(
        config_id=_BASELINE_CONFIG_ID,
        repo_digests=(
            f"registry.example/unrelated@{_BASELINE_DIGEST}",
            f"registry.example/runtime@{_SUBJECT_DIGEST}",
            f"registry.example/runtime@{_BASELINE_DIGEST}",
        ),
    )
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, _image: inspection,
    )

    assert (
        evaluation_oci._portable_image_ref(  # noqa: SLF001
            "registry.example/runtime:candidate",
            _BASELINE_DIGEST,
            engine_path="/usr/bin/docker",
        )
        == f"registry.example/runtime@{_BASELINE_DIGEST}"
    )


def test_repository_manifest_resolution_rejects_wrong_repo_or_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(
        config_id=_BASELINE_CONFIG_ID,
        repo_digests=(
            f"registry.example/unrelated@{_BASELINE_DIGEST}",
            f"registry.example/runtime@{_SUBJECT_DIGEST}",
        ),
    )
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, _image: inspection,
    )

    with pytest.raises(OciEvaluationError, match="does not resolve"):
        evaluation_oci._portable_image_ref(  # noqa: SLF001
            "registry.example/runtime:candidate",
            _BASELINE_DIGEST,
            engine_path="/usr/bin/docker",
        )
    with pytest.raises(OciEvaluationError, match="manifest.*do not agree"):
        evaluation_oci._portable_image_ref(  # noqa: SLF001
            f"registry.example/runtime@{_BASELINE_DIGEST}",
            _BASELINE_DIGEST,
            engine_path="/usr/bin/docker",
        )


def test_local_config_id_mode_is_explicit_and_preflight_returns_declared_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, _image: _inspection(config_id=_BASELINE_CONFIG_ID),
    )
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")

    launch = launch_from_environment(
        engine="docker",
        image_ref=_BASELINE_CONFIG_ID,
        image_digest=_BASELINE_CONFIG_ID,
    )

    assert launch.baseline.image_ref == _BASELINE_CONFIG_ID
    assert preflight_oci_launch(launch) == {
        "baseline": _BASELINE_CONFIG_ID,
        "subject": _BASELINE_CONFIG_ID,
    }

    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, _image: _inspection(
            config_id=_BASELINE_CONFIG_ID,
            repo_digests=(),
        ),
    )
    with pytest.raises(OciEvaluationError, match="use the config ID as both"):
        evaluation_oci._portable_image_ref(  # noqa: SLF001
            "runtime:local",
            _BASELINE_CONFIG_ID,
            engine_path="/usr/bin/docker",
        )


def test_image_inspection_and_launch_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_oci.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("docker", 30)
        ),
    )
    with pytest.raises(OciEvaluationError, match="could not be inspected locally"):
        evaluation_oci._inspect_local_image(  # noqa: SLF001
            "/usr/bin/docker", "runtime:local"
        )

    monkeypatch.setattr(
        evaluation_oci.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["docker", "image", "inspect"],
            1,
            stdout="",
            stderr="image is missing",
        ),
    )
    with pytest.raises(OciEvaluationError, match="image is missing"):
        evaluation_oci._inspect_local_image(  # noqa: SLF001
            "/usr/bin/docker", "runtime:local"
        )

    monkeypatch.setattr(
        evaluation_oci.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["docker", "image", "inspect"],
            0,
            stdout=b"x" * (evaluation_oci._MAX_IMAGE_INSPECT_BYTES + 1),  # noqa: SLF001
            stderr=b"",
        ),
    )
    with pytest.raises(OciEvaluationError, match="bounded size limit"):
        evaluation_oci._inspect_local_image(  # noqa: SLF001
            "/usr/bin/docker", "runtime:local"
        )

    with pytest.raises(OciEvaluationError, match="lowercase sha256"):
        evaluation_oci._portable_image_ref(
            "runtime:local", "sha256:BAD", engine_path="/usr/bin/docker"
        )
    with pytest.raises(OciEvaluationError, match="portable OCI reference"):
        evaluation_oci._portable_image_ref(
            " runtime:local", _BASELINE_DIGEST, engine_path="/usr/bin/docker"
        )
    with pytest.raises(OciEvaluationError, match="do not agree"):
        evaluation_oci._portable_image_ref(
            _SUBJECT_DIGEST, _BASELINE_DIGEST, engine_path="/usr/bin/docker"
        )
    with pytest.raises(OciEvaluationError, match="do not agree"):
        evaluation_oci._portable_image_ref(
            f"registry.example/runtime@{_SUBJECT_DIGEST}",
            _BASELINE_DIGEST,
            engine_path="/usr/bin/docker",
        )

    with pytest.raises(OciEvaluationError, match="docker or podman"):
        evaluation_oci._engine_binary("containerd")
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda _name: None)
    with pytest.raises(OciEvaluationError, match="not available"):
        evaluation_oci._engine_binary("docker")
    with pytest.raises(OciEvaluationError, match="must be cpu, cuda"):
        evaluation_oci._device("metal", label="baseline runtime device")
    with pytest.raises(OciEvaluationError, match="auto, python, or nvidia"):
        evaluation_oci._entrypoint("shell", label="baseline entrypoint")


@pytest.mark.parametrize(
    "limits",
    [
        {"cpus": "0"},
        {"cpus": "1e3"},
        {"memory_mib": 127},
        {"memory_mib": "08"},
        {"user": "0:0"},
        {"user": "nobody:nogroup"},
    ],
)
def test_worker_limits_reject_ambiguous_or_unbounded_values(
    limits: dict[str, object],
) -> None:
    with pytest.raises(OciEvaluationError):
        OciWorkerLimits(**limits)  # type: ignore[arg-type]


def test_side_commands_are_isolated_argv_only_and_preserve_nvidia_initialization(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    job = tmp_path / "job"
    job.mkdir()
    output_parent = tmp_path / "output-root"
    output_parent.mkdir()
    launch = replace(_launch(), worker_limits=OciWorkerLimits(user="12001:13001"))
    baseline = compose_side_worker_command(
        launch=launch,
        side_launch=launch.baseline,
        provider_name="hf_transformers",
        artifact_source=resources,
        support_sources={},
        job_root=job,
        output_root=output_parent,
    )
    subject = compose_side_worker_command(
        launch=launch,
        side_launch=launch.subject,
        provider_name="tensorrt_llm",
        artifact_source=resources,
        support_sources={},
        job_root=job,
        output_root=output_parent,
    )

    assert baseline[:4] == ["docker", "run", "--rm", "--init"]
    assert baseline[baseline.index("--entrypoint") + 1] == "python"
    assert (
        subject[subject.index("--entrypoint") + 1] == "/opt/nvidia/nvidia_entrypoint.sh"
    )
    assert "/opt/invarlock/cli-venv/bin/python" in subject
    assert ["--gpus", "device=3"] == subject[
        subject.index("--gpus") : subject.index("--gpus") + 2
    ]
    for command in (baseline, subject):
        joined = " ".join(command)
        assert "signing" not in joined.lower()
        assert "--network none" in joined
        assert "--pull=never" in command
        assert command[command.index("--cpus") + 1] == "4"
        assert command[command.index("--memory") + 1] == "65536m"
        assert command[command.index("--user") + 1] == "12001:13001"
        environments = [
            command[index + 1] for index, item in enumerate(command) if item == "-e"
        ]
        assert "HOME=/tmp" in environments
        assert "HF_HOME=/tmp/huggingface" in environments
        assert "LOGNAME=12001" in environments
        assert "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor" in environments
        assert "TRITON_CACHE_DIR=/tmp/triton" in environments
        assert "USER=12001" in environments
        assert command[command.index("--cidfile") + 1] == str(job / "container.cid")
        mounts = [
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount"
        ]
        assert any(
            f"source={job.resolve()}" in mount
            and "target=/invarlock/job" in mount
            and "readonly" in mount
            for mount in mounts
        )
        assert any(
            f"source={output_parent.resolve()}" in mount
            and "target=/invarlock/output-root" in mount
            and "readonly" not in mount
            for mount in mounts
        )
        assert not (output_parent / "side").exists()
        assert command[-3:] == [
            "-m",
            "invarlock.evaluation_side_worker",
            "/invarlock/job/job.json",
        ]


def test_side_commands_reject_artifacts_the_worker_user_cannot_read(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    checkpoint = resources / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    checkpoint.chmod(0o600)
    job = tmp_path / "job"
    job.mkdir()
    output_parent = tmp_path / "output-root"
    output_parent.mkdir()
    launch = replace(_launch(), worker_limits=OciWorkerLimits(user="12001:13001"))

    compose = functools.partial(
        compose_side_worker_command,
        launch=launch,
        side_launch=launch.baseline,
        provider_name="hf_transformers",
        artifact_source=resources,
        support_sources={},
        job_root=job,
        output_root=output_parent,
    )

    with pytest.raises(
        OciEvaluationError,
        match=r"not readable by the runtime worker user 12001:13001.*"
        r"model\.safetensors",
    ):
        compose()

    checkpoint.chmod(0o644)
    assert compose()[:3] == ["docker", "run", "--rm"]

    support = tmp_path / "support"
    support.mkdir()
    tokenizer = support / "tokenizer.json"
    tokenizer.write_bytes(b"{}")
    tokenizer.chmod(0o600)
    with pytest.raises(
        OciEvaluationError,
        match=r"side support resource is not readable by the runtime worker",
    ):
        compose(support_sources={"tokenizer": support})


def test_tensorrt_worker_scratch_scales_with_engine_and_stays_bounded(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    engine.joinpath("config.json").write_text("{}")
    engine.joinpath("rank0.engine").touch()

    assert (
        evaluation_oci._worker_tmpfs_size_gib(  # noqa: SLF001
            provider_name="tensorrt_llm", artifact_source=engine
        )
        == 4
    )
    with engine.joinpath("rank0.engine").open("wb") as stream:
        stream.truncate(3 * 1024**3 + 1)
    assert (
        evaluation_oci._worker_tmpfs_size_gib(  # noqa: SLF001
            provider_name="tensorrt_llm", artifact_source=engine
        )
        == 8
    )
    with engine.joinpath("rank0.engine").open("wb") as stream:
        stream.truncate(32 * 1024**3)
    with pytest.raises(OciEvaluationError, match="bounded worker scratch"):
        evaluation_oci._worker_tmpfs_size_gib(  # noqa: SLF001
            provider_name="tensorrt_llm", artifact_source=engine
        )


def test_tensorrt_worker_scratch_rejects_symbolic_entries(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    engine.joinpath("config.json").write_text("{}")
    engine.joinpath("rank0.engine").symlink_to(engine / "config.json")

    with pytest.raises(OciEvaluationError, match="symbolic links"):
        evaluation_oci._worker_tmpfs_size_gib(  # noqa: SLF001
            provider_name="tensorrt_llm", artifact_source=engine
        )


def test_worker_diagnostics_are_drained_but_bounded() -> None:
    completed = evaluation_oci.run_side_worker(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('o' * 100000); "
            "sys.stderr.write('e' * 100000)",
        ],
        timeout_seconds=10,
    )

    assert completed.returncode == 0
    assert len(completed.stdout.encode("utf-8")) == 64 * 1024
    assert len(completed.stderr.encode("utf-8")) == 64 * 1024


def test_worker_cleanup_is_exception_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cidfile = tmp_path / "container.cid"
    cidfile.write_text("a" * 64, encoding="ascii")

    class Stream:
        closed = False

        @staticmethod
        def read(_size: int) -> bytes:
            return b""

        def close(self) -> None:
            self.closed = True

    class Process:
        stdout = Stream()
        stderr = Stream()

        @staticmethod
        def wait(*, timeout: int) -> int:
            assert timeout == 10
            raise RuntimeError("wait failed")

    process = Process()

    def popen(command: list[str], **kwargs: object) -> Process:
        assert command[0] == "/usr/bin/docker"
        assert kwargs["bufsize"] == 0
        return process

    monkeypatch.setattr(evaluation_oci.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="wait failed"):
        evaluation_oci.run_side_worker(
            ["/usr/bin/docker", "run", "--cidfile", str(cidfile)],
            timeout_seconds=10,
        )

    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert not cidfile.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe inheritance contract")
def test_worker_return_is_bounded_when_a_descendant_inherits_diagnostic_pipes() -> None:
    started = time.monotonic()
    completed = evaluation_oci.run_side_worker(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(3)'], stdout=sys.stdout, stderr=sys.stderr); "
            "print('parent complete')",
        ],
        timeout_seconds=10,
    )

    assert completed.returncode == 0
    assert "parent complete" in completed.stdout
    assert time.monotonic() - started < 2


def test_worker_outer_deadline_terminates_a_hung_process() -> None:
    completed = evaluation_oci.run_side_worker(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_seconds=1,
    )

    assert completed.returncode == 124
    assert "1-second outer deadline" in completed.stderr


def test_timeout_cleanup_stops_then_kills_the_engine_issued_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cidfile = tmp_path / "container.cid"
    container_id = "a" * 64
    cidfile.write_text(container_id + "\n", encoding="ascii")
    controls: list[list[str]] = []

    def control(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        controls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    class Process:
        terminated = False

        @staticmethod
        def poll() -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        @staticmethod
        def wait(*, timeout: int) -> int:
            assert timeout == 5
            return -15

    process = Process()
    monkeypatch.setattr(evaluation_oci.subprocess, "run", control)

    evaluation_oci._terminate_timed_out_worker(  # type: ignore[arg-type]  # noqa: SLF001
        process,
        ["/usr/bin/docker", "run", "--cidfile", str(cidfile)],
        cidfile,
    )

    assert controls == [
        ["/usr/bin/docker", "stop", "--time", "5", container_id],
        ["/usr/bin/docker", "kill", container_id],
    ]
    assert process.terminated is True


def test_worker_deadline_is_derived_from_timeout_and_schedule_size() -> None:
    assert (
        evaluation_oci._worker_outer_timeout_seconds(  # noqa: SLF001
            {"timeout_seconds": 30}, record_count=8
        )
        == 300
    )
    assert (
        evaluation_oci._worker_outer_timeout_seconds(  # noqa: SLF001
            {"timeout_seconds": 3600}, record_count=10_000
        )
        == 24 * 60 * 60
    )
    with pytest.raises(OciEvaluationError, match="positive integer"):
        evaluation_oci._worker_outer_timeout_seconds(  # noqa: SLF001
            {"timeout_seconds": True}, record_count=8
        )


@pytest.mark.parametrize(
    ("baseline", "subject", "expected"),
    [
        ("cpu", "cpu", True),
        ("cuda", "cuda", False),
        ("cuda:0", "cuda:0", False),
        ("cuda:0", "cuda:1", True),
        ("cpu", "cuda:1", False),
    ],
)
def test_worker_parallelism_never_shares_one_cuda_device(
    baseline: str, subject: str, expected: bool
) -> None:
    launch = OciEvaluationLaunch(
        engine="docker",
        baseline=OciSideLaunch(
            image_ref=f"registry.example/baseline@{_BASELINE_DIGEST}",
            image_digest=_BASELINE_DIGEST,
            device=baseline,
        ),
        subject=OciSideLaunch(
            image_ref=f"registry.example/subject@{_SUBJECT_DIGEST}",
            image_digest=_SUBJECT_DIGEST,
            device=subject,
        ),
    )
    assert _workers_may_run_parallel(launch) is expected


class _Registry:
    def get_runtime_provider(self, name: str) -> RuntimeProvider:
        return cast(RuntimeProvider, SimpleNamespace(name=name))


def _evidence(role: str) -> RuntimeSideEvidence:
    return RuntimeSideEvidence(
        run_report=f"{role}-report".encode(),
        runtime_manifest=f"{role}-manifest".encode(),
        runtime_config=f"{role}-config".encode(),
        artifact_identity=f"{role}-identity".encode(),
        provider_receipt=role.encode(),
        scoring_observation=f"{role}-observation".encode(),
    )


def test_executor_launches_different_images_and_collects_only_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    support = {
        "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT": str(tmp_path),
        "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT": "models/subject/tokenizer.json",
    }
    (tmp_path / "models/subject/tokenizer.json").write_text("{}")
    commands: list[list[str]] = []

    observed_timeouts: list[int] = []

    def run(
        command: list[str], *, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        observed_timeouts.append(timeout_seconds)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(evaluation_oci, "run_side_worker", run)
    monkeypatch.setattr(
        evaluation_oci,
        "ThreadPoolExecutor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cpu/cuda workers must run sequentially")
        ),
    )
    monkeypatch.setattr(
        evaluation_oci,
        "load_runtime_side_evidence",
        lambda path: _evidence(path.parent.name.removesuffix("-output")),
    )
    monkeypatch.setattr(
        evaluation_oci,
        "decode_runtime_provider_receipt",
        lambda payload: SimpleNamespace(
            outer_image_digest=(
                _BASELINE_DIGEST if payload == b"baseline" else _SUBJECT_DIGEST
            )
        ),
    )

    result = OciRuntimeExecutor(_launch(), environment=support).execute(
        request,
        registry=cast(CoreRegistry, _Registry()),
        schedule_bytes=canonical_schedule_bytes(_schedule()),
        policy_digest="sha256:" + "c" * 64,
    )

    assert len(commands) == 2
    assert observed_timeouts == [120, 120]
    assert any(_launch().baseline.image_ref in command for command in commands)
    assert any(_launch().subject.image_ref in command for command in commands)
    assert result.baseline.run_report == b"baseline-report"
    assert result.subject.run_report == b"subject-report"
    assert result.baseline_runtime_digest != result.subject_runtime_digest
    baseline_command = next(
        command for command in commands if _launch().baseline.image_ref in command
    )
    subject_command = next(
        command for command in commands if _launch().subject.image_ref in command
    )
    assert str(tmp_path / "models/baseline") in " ".join(baseline_command)
    assert str(tmp_path / "models/subject") not in " ".join(baseline_command)
    assert str(tmp_path / "models/subject") in " ".join(subject_command)
    assert str(tmp_path / "models/baseline") not in " ".join(subject_command)
    assert "/opt/invarlock/bin/tensorrt-llm-runner" not in subject_command


def test_executor_fails_closed_when_one_side_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    support = {
        "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT": str(tmp_path),
        "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT": "models/subject/tokenizer.json",
    }
    (tmp_path / "models/subject/tokenizer.json").write_text("{}")

    def run(
        command: list[str], *, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        assert timeout_seconds == 120
        failed = _launch().subject.image_ref in command
        return subprocess.CompletedProcess(
            command,
            7 if failed else 0,
            stdout="",
            stderr="backend failed" if failed else "",
        )

    monkeypatch.setattr(evaluation_oci, "run_side_worker", run)
    with pytest.raises(OciEvaluationError, match="subject worker exited with status 7"):
        OciRuntimeExecutor(_launch(), environment=support).execute(
            request,
            registry=cast(CoreRegistry, _Registry()),
            schedule_bytes=canonical_schedule_bytes(_schedule()),
            policy_digest="sha256:" + "c" * 64,
        )


def test_six_file_loader_rejects_partial_or_mixed_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    names = {
        "report.json",
        "runtime.manifest.json",
        "run.yaml",
        "model-artifact.identity.json",
        "runtime-provider.receipt.json",
        "runtime-scoring.observation.json",
    }
    for name in names - {"report.json"}:
        output.joinpath(name).write_text("{}")
    with pytest.raises(ValueError, match="missing report.json"):
        load_runtime_side_evidence(output)
    output.joinpath("report.json").write_text("{}")
    output.joinpath("foreign.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected foreign.json"):
        load_runtime_side_evidence(output)
