from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import invarlock.runtime_behavior.transaction as runtime_transaction
from invarlock.cli.app import app
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import ModelRuntimeSpec, artifact_identity_sha256
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evaluation_runtime import CallerRuntimeResources
from invarlock.evaluation_transaction import evaluate_request_file
from invarlock.evidence_verification import verify_evidence
from invarlock.runtime_providers import hf_transformers
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "run" / "hf_cpu_decision.py"
IMAGE_DIGEST = "sha256:" + "7" * 64


def _prepare(workspace: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--workspace",
            str(workspace),
            "--prepare-only",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _payload(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_checked_in_hf_run_example_is_distinct_anchored_and_meaningful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")

    workspace = tmp_path / "hf-run"
    prepared = _prepare(workspace)
    assert prepared.returncode == 0, prepared.stderr
    assert "Keys outside request tree" in prepared.stdout

    request_path = workspace / "evaluation" / "request.yaml"
    before = hashlib.sha256(request_path.read_bytes()).hexdigest()
    refused = _prepare(workspace)
    assert refused.returncode == 2
    assert "workspace already exists" in refused.stderr
    assert hashlib.sha256(request_path.read_bytes()).hexdigest() == before

    request = _payload(request_path)
    assert request["execution"] == {"mode": "run"}
    assert request["comparison"]["metric"] == "normalized_nll_per_utf8_byte"
    assert (
        request["comparison"]["baseline"]["artifact"]["path"]
        != request["comparison"]["subject"]["artifact"]["path"]
    )

    evaluation = workspace / "evaluation"
    checkpoints = {
        role: evaluation / request["comparison"][role]["artifact"]["path"]
        for role in ("baseline", "subject")
    }
    assert checkpoint_tree_sha256(checkpoints["baseline"]) != checkpoint_tree_sha256(
        checkpoints["subject"]
    )

    anchors = json.loads(
        (workspace / "verifier" / "trusted-inputs.json").read_text(encoding="utf-8")
    )
    provider = HFTransformersProvider()
    for role in ("baseline", "subject"):
        side = request["comparison"][role]
        observed = "sha256:" + artifact_identity_sha256(
            provider.identify_artifact(
                ModelRuntimeSpec(
                    provider_name="hf_transformers",
                    model_id=side["artifact"]["model_id"],
                    settings=side["runtime"]["settings"],
                )
            )
        )
        assert anchors[f"{role}_artifact"] == observed

    dataset = evaluation / "inputs" / "records.jsonl"
    dataset_bytes = dataset.read_bytes()
    comparison_dataset = request["comparison"]["dataset"]
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset,
            sha256=comparison_dataset["sha256"],
            name=comparison_dataset["name"],
            split=comparison_dataset["split"],
            input_field=comparison_dataset["input_field"],
            expected_output_field=comparison_dataset["expected_output_field"],
            id_field=comparison_dataset["id_field"],
        ),
        dataset_bytes,
    )
    assert anchors["canonical_schedule"] == f"sha256:{schedule.schedule_sha256}"

    request_policy = evaluation / "inputs" / "acceptance.json"
    verifier_policy = workspace / "verifier" / "policy" / "acceptance.json"
    assert request_policy.read_bytes() == verifier_policy.read_bytes()
    assert (
        anchors["policy_sha256"]
        == "sha256:" + hashlib.sha256(verifier_policy.read_bytes()).hexdigest()
    )

    evidence_key = workspace / "keys" / "evidence-signer.pem"
    verifier_key = workspace / "keys" / "verifier.pem"
    assert not evidence_key.is_relative_to(evaluation)
    assert not verifier_key.is_relative_to(evaluation)
    assert stat.S_IMODE(evidence_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(verifier_key.stat().st_mode) == 0o600

    image_ref = f"registry.invalid/invarlock@{IMAGE_DIGEST}"
    for module in (hf_transformers, runtime_transaction):
        monkeypatch.setattr(module, "strict_container_boundary_present", lambda: True)
        monkeypatch.setattr(
            module, "resolve_runtime_image_digest", lambda: IMAGE_DIGEST
        )
        monkeypatch.setattr(module, "resolve_runtime_image", lambda: image_ref)

    evaluated = evaluate_request_file(
        request_path,
        signing_key_path=evidence_key,
        resource_resolver=CallerRuntimeResources(container_image_digest=IMAGE_DIGEST),
    )
    receipt = workspace / "verifier" / "verification.receipt.json"
    verified = verify_evidence(
        evaluated.evidence_path,
        policy_path=verifier_policy,
        expected_baseline_artifact=anchors["baseline_artifact"],
        expected_subject_artifact=anchors["subject_artifact"],
        expected_schedule=anchors["canonical_schedule"],
        expected_baseline_runtime=IMAGE_DIGEST,
        expected_subject_runtime=IMAGE_DIGEST,
        expected_signer=anchors["evidence_signer"],
        receipt_path=receipt,
        verifier_signing_key_path=verifier_key,
        verifier_identity="checked-in-hf-run-test",
    )
    assert verified.payload["ok"] is True
    assert verified.payload["policy_verdict"] == "pass"
    assert receipt.is_file()

    report = json.loads(
        (evaluated.evidence_path / "reports" / "evaluation.report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["verdict"] == "pass"
    assert report["metric"] == "normalized_nll_per_utf8_byte"
    assert 0.0 < report["comparison"]["value"] < 0.1
    assert report["baseline"]["mean_score"] > report["subject"]["mean_score"]
    assert report["derived_measurements"]["perplexity_ratio"]["status"] == "available"
    perplexity = report["derived_measurements"]["perplexity_ratio"]
    assert 0.0 < perplexity["ratio"] < 0.1
    assert perplexity["ratio"] == pytest.approx(
        perplexity["subject_perplexity"] / perplexity["baseline_perplexity"]
    )

    html = workspace / "comparison-report.html"
    rendered = CliRunner().invoke(
        app,
        ["report", str(evaluated.evidence_path), "--html", str(html)],
    )
    assert rendered.exit_code == 0, rendered.stdout
    assert html.is_file()
    assert "PASS" in html.read_text(encoding="utf-8")
