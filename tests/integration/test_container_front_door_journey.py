from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint

pytestmark = pytest.mark.skipif(
    os.environ.get("INVARLOCK_RUN_CONTAINER_SMOKE") != "1",
    reason="set INVARLOCK_RUN_CONTAINER_SMOKE=1 to run the container front-door journey",
)


def _module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised by the opt-in target
        pytest.fail(f"container smoke fixture requires {name}: {exc}")


def _private_key(path: Path) -> tuple[Path, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path, public_key_fingerprint(key.public_key())


def _run_child(
    engine: str,
    image: str,
    workspace: Path,
    arguments: list[str],
    *,
    image_digest: str | None,
) -> subprocess.CompletedProcess[str]:
    command = [
        engine,
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        f"type=bind,source={workspace},target=/work",
        "-e",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "-e",
        "INVARLOCK_RUNTIME_DEVICE=cpu",
    ]
    if image_digest is not None:
        command.extend(
            [
                "-e",
                f"INVARLOCK_RUNTIME_IMAGE={image_digest}",
                "-e",
                f"INVARLOCK_RUNTIME_IMAGE_DIGEST={image_digest}",
            ]
        )
    command.extend([image, *arguments])
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_host(
    arguments: list[str],
    *,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    for variable in (
        "INVARLOCK_CONTAINER_EXECUTION",
        "INVARLOCK_RUNTIME_IMAGE_DIGEST",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
    ):
        environment.pop(variable, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return subprocess.run(
        [sys.executable, "-m", "invarlock", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def _tiny_checkpoint(workspace: Path) -> tuple[Path, str, str]:
    torch = _module("torch")
    tokenizers = _module("tokenizers")
    transformers = _module("transformers")
    from invarlock.runtime_providers.hf_transformers import (
        hf_tokenizer_contract_sha256,
    )

    torch.manual_seed(19)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.eval()
    checkpoint = workspace / "models" / "tiny-hf"
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        **{f"token-{index}": index + 4 for index in range(28)},
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(checkpoint)
    return (
        checkpoint,
        checkpoint_tree_sha256(checkpoint),
        hf_tokenizer_contract_sha256(tokenizer),
    )


def _request(
    workspace: Path,
    *,
    checkpoint_digest: str,
    tokenizer_digest: str,
    output: str,
    provider: str = "hf_transformers",
) -> Path:
    inputs = workspace / "inputs"
    inputs.mkdir(exist_ok=True)
    dataset_bytes = (
        b'{"id":"tiny-1","prompt":"token-1 token-2 token-3",'
        b'"expected":"not-the-random-model-output"}\n'
    )
    inputs.joinpath("records.jsonl").write_bytes(dataset_bytes)
    inputs.joinpath("policy.json").write_bytes(
        canonical_json_bytes(
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0.0}}}}
        )
    )

    def side() -> dict[str, object]:
        return {
            "artifact": {
                "path": "models/tiny-hf",
                "model_id": "tiny/reference",
                "locator": f"hf://tiny/reference@{'b' * 40}",
            },
            "runtime": {
                "provider": provider,
                "settings": {
                    "batch_size": 1,
                    "checkpoint_tree_sha256": checkpoint_digest,
                    "context_length": 12,
                    "max_output_tokens": 1,
                    "offline": True,
                    "seed": 19,
                    "timeout_seconds": 30,
                    "tokenizer_metadata_sha256": tokenizer_digest,
                },
            },
        }

    payload = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side(),
            "subject": side(),
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "format": "jsonl",
                "name": "container-front-door-smoke",
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
        "output": {"evidence": output},
    }
    request = workspace / f"request-{output.replace('/', '-')}.yaml"
    request.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return request


def _assert_ok(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    payload = cast(dict[str, object], json.loads(result.stdout))
    assert payload["ok"] is True
    return payload


def test_runtime_image_host_front_door_evaluate_verify_report_and_fail_closed(
    tmp_path: Path,
) -> None:
    engine = os.environ.get("INVARLOCK_CONTAINER_ENGINE", "docker")
    image = os.environ.get("INVARLOCK_RUNTIME_IMAGE", "invarlock-runtime:local")
    inspected = subprocess.run(
        [engine, "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    image_digest = inspected.stdout.strip()
    assert image_digest.startswith("sha256:") and len(image_digest) == 71

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_root = tmp_path / "keys"
    key_root.mkdir()

    help_result = _run_child(
        engine, image, workspace, ["--help"], image_digest=image_digest
    )
    assert help_result.returncode == 0, help_result.stderr
    for command in ("evaluate", "verify", "report"):
        assert command in help_result.stdout
    for retired in ("audit", "catalog", "doctor", "evidence-pack"):
        assert retired not in help_result.stdout

    _checkpoint, checkpoint_digest, tokenizer_digest = _tiny_checkpoint(workspace)
    evidence_signer_key, evidence_signer_fingerprint = _private_key(
        key_root / "evidence-signer.pem"
    )
    verifier_key, _verifier_fingerprint = _private_key(key_root / "verifier.pem")
    request = _request(
        workspace,
        checkpoint_digest=checkpoint_digest,
        tokenizer_digest=tokenizer_digest,
        output="evidence",
    )
    evaluated = _run_host(
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(evidence_signer_key),
            "--container-engine",
            engine,
            "--runtime-image",
            image_digest,
            "--json",
        ],
    )
    evaluated_payload = _assert_ok(evaluated)
    evidence = workspace / "evidence"
    assert evaluated_payload["evidence"] == str(evidence)
    assert evidence.is_dir()
    input_anchors = {
        role: json.loads(
            (evidence / "inputs" / f"{role}.json").read_text(encoding="utf-8")
        )["digest"]
        for role in ("baseline", "subject", "dataset")
    }

    receipt = tmp_path / "verification-receipt.json"
    verified = _run_host(
        [
            "verify",
            str(evidence),
            "--policy",
            str(workspace / "inputs/policy.json"),
            "--expected-baseline-artifact",
            input_anchors["baseline"],
            "--expected-subject-artifact",
            input_anchors["subject"],
            "--expected-schedule",
            input_anchors["dataset"],
            "--expected-baseline-runtime",
            image_digest,
            "--expected-subject-runtime",
            image_digest,
            "--expected-signer",
            evidence_signer_fingerprint,
            "--receipt",
            str(receipt),
            "--verifier-signing-key",
            str(verifier_key),
            "--verifier-identity",
            "container-smoke-verifier",
            "--json",
        ],
    )
    _assert_ok(verified)
    assert receipt.is_file()

    report_path = tmp_path / "report.html"
    report = _run_host(
        ["report", str(evidence), "--html", str(report_path)],
    )
    assert report.returncode == 0, report.stderr or report.stdout
    assert "# InvarLock comparison report" in report.stdout
    assert report_path.is_file()

    missing_digest_request = _request(
        workspace,
        checkpoint_digest=checkpoint_digest,
        tokenizer_digest=tokenizer_digest,
        output="missing-digest-evidence",
    )
    missing_digest = _run_host(
        [
            "evaluate",
            str(missing_digest_request),
            "--signing-key",
            str(evidence_signer_key),
            "--container-engine",
            engine,
            "--runtime-image",
            image,
            "--json",
        ],
    )
    assert missing_digest.returncode != 0
    assert "INVARLOCK_RUNTIME_IMAGE_DIGEST" in missing_digest.stdout
    assert not workspace.joinpath("missing-digest-evidence").exists()

    wrong_receipt = tmp_path / "wrong-runtime-receipt.json"
    wrong_runtime = _run_host(
        [
            "verify",
            str(evidence),
            "--policy",
            str(workspace / "inputs/policy.json"),
            "--expected-baseline-artifact",
            input_anchors["baseline"],
            "--expected-subject-artifact",
            input_anchors["subject"],
            "--expected-schedule",
            input_anchors["dataset"],
            "--expected-baseline-runtime",
            "sha256:" + "0" * 64,
            "--expected-subject-runtime",
            image_digest,
            "--expected-signer",
            evidence_signer_fingerprint,
            "--receipt",
            str(wrong_receipt),
            "--verifier-signing-key",
            str(verifier_key),
            "--verifier-identity",
            "container-smoke-verifier",
            "--json",
        ],
    )
    assert wrong_runtime.returncode != 0
    wrong_payload = json.loads(wrong_runtime.stdout)
    assert wrong_payload["ok"] is False
    assert wrong_receipt.is_file()

    unavailable_request = _request(
        workspace,
        checkpoint_digest=hashlib.sha256(b"unavailable").hexdigest(),
        tokenizer_digest=tokenizer_digest,
        output="unavailable-provider-evidence",
        provider="unavailable_provider",
    )
    unavailable = _run_host(
        [
            "evaluate",
            str(unavailable_request),
            "--signing-key",
            str(evidence_signer_key),
            "--container-engine",
            engine,
            "--runtime-image",
            image_digest,
            "--json",
        ],
    )
    assert unavailable.returncode != 0
    assert "unavailable_provider" in unavailable.stdout
    assert not workspace.joinpath("unavailable-provider-evidence").exists()
