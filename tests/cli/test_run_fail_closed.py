from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evidence_pack_contract import canonical_json_bytes


def _signing_key(path: Path) -> Path:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path


def test_run_mode_requires_caller_owned_image_before_provider_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", raising=False)
    models = tmp_path / "models"
    models.joinpath("baseline").mkdir(parents=True)
    models.joinpath("subject").mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset_bytes = b'{"id":"one","prompt":"Return A","expected":"A"}\n'
    inputs.joinpath("records.jsonl").write_bytes(dataset_bytes)
    inputs.joinpath("policy.json").write_bytes(
        canonical_json_bytes(
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0.0}}}}
        )
    )

    def side(name: str) -> dict[str, object]:
        return {
            "artifact": {
                "path": f"models/{name}",
                "model_id": f"local/{name}",
                "locator": f"hf://local/{name}@{'b' * 40}",
            },
            "runtime": {
                "provider": "hf_transformers",
                "settings": {
                    "batch_size": 1,
                    "checkpoint_tree_sha256": "c" * 64,
                    "context_length": 8,
                    "max_output_tokens": 1,
                    "offline": True,
                    "seed": 0,
                    "timeout_seconds": 30,
                    "tokenizer_metadata_sha256": "d" * 64,
                },
            },
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline"),
            "subject": side("subject"),
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "format": "jsonl",
                "name": "run-contract",
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
    }
    request_path = tmp_path / "request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request_path),
            "--signing-key",
            str(_signing_key(tmp_path / "evidence.pem")),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "INVARLOCK_RUNTIME_IMAGE_DIGEST" in payload["errors"][0]
    assert not tmp_path.joinpath("artifacts", "evidence").exists()
