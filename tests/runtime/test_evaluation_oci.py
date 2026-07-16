from __future__ import annotations

import json
import subprocess
import sys
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
    _workers_may_run_parallel,
    compose_side_worker_command,
    evaluation_request_execution_mode,
    launch_from_environment,
)
from invarlock.evaluation_run import load_runtime_side_evidence
from invarlock.evidence_pack import RuntimeSideEvidence

_BASELINE_DIGEST = "sha256:" + "a" * 64
_SUBJECT_DIGEST = "sha256:" + "b" * 64


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
        runtime=RuntimeRequest(provider=provider, settings={"batch_size": 1}),
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
        "_local_image_id",
        lambda _engine, image: _SUBJECT_DIGEST if "trt" in image else _BASELINE_DIGEST,
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
    assert split.baseline.image_ref == _BASELINE_DIGEST
    assert split.subject.image_ref == _SUBJECT_DIGEST
    assert split.subject.device == "cuda:2"

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
        evaluation_oci, "_local_image_id", lambda _engine, _image: _SUBJECT_DIGEST
    )
    with pytest.raises(OciEvaluationError, match="do not agree"):
        launch_from_environment(
            engine="docker",
            image_ref="registry.example/runtime:local",
            image_digest=_BASELINE_DIGEST,
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
        evaluation_oci._local_image_id("/usr/bin/docker", "runtime:local")

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
        evaluation_oci._local_image_id("/usr/bin/docker", "runtime:local")

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


def test_side_commands_are_isolated_argv_only_and_preserve_nvidia_initialization(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    job = tmp_path / "job"
    job.mkdir()
    output_parent = tmp_path / "output-root"
    output_parent.mkdir()
    launch = _launch()
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


def test_worker_diagnostics_are_drained_but_bounded() -> None:
    completed = evaluation_oci.run_side_worker(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('o' * 100000); "
            "sys.stderr.write('e' * 100000)",
        ]
    )

    assert completed.returncode == 0
    assert len(completed.stdout.encode("utf-8")) == 64 * 1024
    assert len(completed.stderr.encode("utf-8")) == 64 * 1024


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
        "INVARLOCK_TENSORRT_LLM_RUNNER_EXECUTABLE": "models/subject/runner",
    }
    (tmp_path / "models/subject/tokenizer.json").write_text("{}")
    (tmp_path / "models/subject/runner").write_text("runner")
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
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
        schedule_bytes=b"{}\n",
        policy_digest="sha256:" + "c" * 64,
    )

    assert len(commands) == 2
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


def test_executor_fails_closed_when_one_side_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    support = {
        "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT": str(tmp_path),
        "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT": "models/subject/tokenizer.json",
        "INVARLOCK_TENSORRT_LLM_RUNNER_EXECUTABLE": "models/subject/runner",
    }
    (tmp_path / "models/subject/tokenizer.json").write_text("{}")
    (tmp_path / "models/subject/runner").write_text("runner")

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
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
            schedule_bytes=b"{}\n",
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
