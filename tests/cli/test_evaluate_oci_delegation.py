from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import invarlock.evaluation_oci as evaluation_oci
import invarlock.evaluation_transaction as evaluation_transaction
from invarlock.cli.app import app
from invarlock.core.evaluation_request import EvaluationRequest
from invarlock.evaluation_oci import OciEvaluationError, OciRuntimeExecutor
from invarlock.evaluation_transaction import EvaluationTransactionResult

_BASELINE_DIGEST = "sha256:" + "a" * 64
_SUBJECT_DIGEST = "sha256:" + "b" * 64


def _mock_image_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    def inspect(_engine: str, image: str) -> object:
        digest = _SUBJECT_DIGEST if "trt" in image else _BASELINE_DIGEST
        repository = (
            image.rsplit("@", 1)[0]
            if "@" in image
            else evaluation_oci._tag_repository(image)  # noqa: SLF001
        )
        return evaluation_oci._LocalImageInspection(  # noqa: SLF001
            config_id="sha256:" + ("d" if "trt" in image else "c") * 64,
            repo_digests=(f"{repository}@{digest}",),
        )

    monkeypatch.setattr(evaluation_oci, "_inspect_local_image", inspect)


def _mock_preflight_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluation_transaction,
        "preflight_evaluation_request",
        lambda *_args, **_kwargs: object(),
    )


def _request(path: Path) -> Path:
    model = path.parent / "models" / "model"
    model.mkdir(parents=True)
    inputs = path.parent / "inputs"
    inputs.mkdir()
    inputs.joinpath("records.jsonl").write_text(
        '{"expected":"ok","id":"1","prompt":"ready"}\n',
        encoding="utf-8",
    )
    inputs.joinpath("policy.json").write_text("{}\n", encoding="utf-8")
    side = {
        "artifact": {
            "path": "models/model",
            "model_id": "local/model",
            "locator": "artifact:local-model",
        },
        "runtime": {"provider": "hf_transformers", "settings": {}},
    }
    path.write_text(
        yaml.safe_dump(
            {
                "format_version": "invarlock/evaluation-request-v1",
                "comparison": {
                    "baseline": json.loads(json.dumps(side)),
                    "subject": json.loads(json.dumps(side)),
                    "dataset": {
                        "path": "inputs/records.jsonl",
                        "sha256": "a" * 64,
                        "format": "jsonl",
                        "name": "local-test",
                        "split": "validation",
                        "input_field": "prompt",
                        "expected_output_field": "expected",
                        "id_field": "id",
                    },
                    "policy": "inputs/policy.json",
                    "task": "text_causal",
                    "metric": "exact_match",
                },
                "execution": {"mode": "run"},
                "output": {"evidence": "artifacts/evidence"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_internal_side_worker_is_not_a_public_command() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "side-worker" not in result.stdout
    assert "evaluation_side_worker" not in result.stdout


def test_run_request_keeps_host_transaction_and_passes_per_side_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "request.yaml")
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    _mock_image_inspection(monkeypatch)
    _mock_preflight_success(monkeypatch)
    observed: dict[str, object] = {}

    def evaluate(
        path: Path | EvaluationRequest, **kwargs: object
    ) -> EvaluationTransactionResult:
        assert isinstance(path, EvaluationRequest)
        assert path.execution.mode == "run"
        observed.update({"path": path, **kwargs})
        return EvaluationTransactionResult(
            evidence_path=tmp_path / "artifacts/evidence",
            comparison_id="comparison-123",
            pack_manifest_digest="sha256:" + ("a" * 64),
        )

    monkeypatch.setattr(evaluation_transaction, "evaluate_request_file", evaluate)
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--runtime-image",
            "registry.example/hf:local",
            "--runtime-image-digest",
            _BASELINE_DIGEST,
            "--subject-runtime-image",
            "registry.example/trt:local",
            "--subject-runtime-image-digest",
            _SUBJECT_DIGEST,
            "--subject-runtime-device",
            "cuda:1",
            "--runtime-cpus",
            "6.5",
            "--runtime-memory-mib",
            "12288",
            "--runtime-user",
            "12001:12001",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["evidence"] == str(tmp_path / "artifacts/evidence")
    executor = observed["runtime_executor"]
    assert isinstance(executor, OciRuntimeExecutor)
    assert executor.launch.baseline.image_digest == _BASELINE_DIGEST
    assert executor.launch.subject.image_digest == _SUBJECT_DIGEST
    assert executor.launch.subject.device == "cuda:1"
    assert executor.launch.worker_limits.cpus == "6.5"
    assert executor.launch.worker_limits.memory_mib == 12288
    assert executor.launch.worker_limits.user == "12001:12001"
    assert observed["signing_key_path"] is None


def test_worker_failure_is_a_structured_host_transaction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "request.yaml")
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    _mock_image_inspection(monkeypatch)
    _mock_preflight_success(monkeypatch)

    def fail(*_args: object, **_kwargs: object) -> EvaluationTransactionResult:
        raise OciEvaluationError("subject worker failed closed")

    monkeypatch.setattr(evaluation_transaction, "evaluate_request_file", fail)
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--runtime-image",
            f"registry.example/runtime@{_BASELINE_DIGEST}",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["errors"] == ["subject worker failed closed"]


def test_mutable_side_image_is_rejected_before_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "request.yaml")
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    _mock_image_inspection(monkeypatch)
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--runtime-image",
            f"registry.example/runtime@{_BASELINE_DIGEST}",
            "--subject-runtime-image",
            "registry.example/mutable:latest",
            "--subject-runtime-image-digest",
            "not-a-digest",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "digest" in json.loads(result.stdout)["errors"][0]


def _runtime_profile(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "invarlock/runtime-profile-v1",
                "runtime": {
                    "image": f"registry.example/hf@{_BASELINE_DIGEST}",
                    "device": "cpu",
                    "cpus": "2",
                    "memory_mib": 4096,
                },
                "subject": {"device": "cuda:1"},
            }
        )
    )
    return path


def test_profile_preflight_uses_pinned_oci_launch_and_keeps_json_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "request.yaml")
    profile = _runtime_profile(tmp_path / "runtime.json")
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    _mock_image_inspection(monkeypatch)
    # These lower-priority environment values must neither win nor invalidate
    # the fully resolved profile launch.
    monkeypatch.setenv("INVARLOCK_RUNTIME_DEVICE", "invalid-unused-device")
    monkeypatch.setenv("INVARLOCK_SUBJECT_RUNTIME_DEVICE", "cuda:7")
    observed: list[OciRuntimeExecutor] = []
    expected = evaluation_transaction.EvaluationPreflightResult(
        execution_mode="run",
        output="artifacts/evidence",
        schedule_digest="a" * 64,
        policy_digest="b" * 64,
        artifact_digests={},
        evidence_signer_fingerprint="c" * 64,
        request_digest="d" * 64,
        record_count=1,
        providers={},
        checks=("runtime",),
    )

    def preflight(*_args: object, **kwargs: object):
        executor = kwargs["resource_resolver"]
        assert isinstance(executor, OciRuntimeExecutor)
        observed.append(executor)
        return expected

    monkeypatch.setattr(
        evaluation_transaction, "preflight_evaluation_request", preflight
    )
    args = [
        "evaluate",
        str(request),
        "--runtime-profile",
        str(profile),
        "--runtime-device",
        "cuda:2",
        "--baseline-runtime-device",
        "cpu",
        "--preflight",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.stdout
    assert observed[0].launch.baseline.device == "cpu"
    assert observed[0].launch.subject.device == "cuda:2"
    assert observed[0].launch.subject.image_digest == _BASELINE_DIGEST
    assert observed[0].launch.worker_limits.memory_mib == 4096
    assert "profile.runtime.memory_mib" in result.stdout
    assert "--runtime-device" in result.stdout
    assert not (tmp_path / "artifacts").exists()
    result = CliRunner().invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == json.loads(expected.as_json())


@pytest.mark.parametrize("mutation", ["digest_conflict", "root_user", "mutable_image"])
def test_profile_oci_failures_do_not_reach_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    request = _request(tmp_path / "request.yaml")
    profile = _runtime_profile(tmp_path / "runtime.json")
    payload = json.loads(profile.read_text())
    args = []
    if mutation == "digest_conflict":
        payload["subject"]["image_digest"] = _BASELINE_DIGEST
        args = ["--subject-runtime-image", f"registry.example/trt@{_SUBJECT_DIGEST}"]
    elif mutation == "root_user":
        payload["runtime"]["user"] = "0:0"
    else:
        payload["runtime"]["image"] = "registry.example/mutable:latest"
    profile.write_text(json.dumps(payload))
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")
    _mock_image_inspection(monkeypatch)

    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("invalid runtime profile reached execution")

    monkeypatch.setattr(evaluation_transaction, "evaluate_request_file", forbidden)
    result = CliRunner().invoke(
        app,
        ["evaluate", str(request), "--runtime-profile", str(profile), *args, "--json"],
    )
    assert result.exit_code == 2, result.stdout
    assert json.loads(result.stdout)["ok"] is False
    assert not (tmp_path / "artifacts").exists()


def test_profile_file_error_is_a_json_preflight_failure(tmp_path: Path) -> None:
    request = _request(tmp_path / "request.yaml")
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--runtime-profile",
            str(tmp_path / "missing.json"),
            "--preflight",
            "--json",
        ],
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "invarlock/evaluation-preflight-v2"
    assert payload["ok"] is False
    assert "runtime profile" in payload["errors"][0]
