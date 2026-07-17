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
