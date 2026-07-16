from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import invarlock.evaluation_oci as evaluation_oci
import invarlock.evaluation_transaction as evaluation_transaction
from invarlock.cli.app import app
from invarlock.evaluation_oci import OciEvaluationError, OciRuntimeExecutor
from invarlock.evaluation_transaction import EvaluationTransactionResult

_BASELINE_DIGEST = "sha256:" + "a" * 64
_SUBJECT_DIGEST = "sha256:" + "b" * 64


def _request(path: Path) -> Path:
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
    monkeypatch.setattr(
        evaluation_oci,
        "_local_image_id",
        lambda _engine, image: _SUBJECT_DIGEST if "trt" in image else _BASELINE_DIGEST,
    )
    observed: dict[str, object] = {}

    def evaluate(path: Path, **kwargs: object) -> EvaluationTransactionResult:
        observed.update({"path": path, **kwargs})
        return EvaluationTransactionResult(
            evidence_path=tmp_path / "artifacts/evidence",
            comparison_id="comparison-123",
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
    assert observed["signing_key_path"] is None


def test_worker_failure_is_a_structured_host_transaction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "request.yaml")
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda name: f"/bin/{name}")

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
