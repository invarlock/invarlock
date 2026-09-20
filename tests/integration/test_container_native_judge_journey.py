"""Actual OCI answer capture with fixed offline judge measurements.

Only judge collection and its credential preflight are replaced by a complete
measurement fixture. Native provider execution, capture binding, publication,
independent verification and report rendering use production code. This tests
container interoperability, not the quality or execution of a live model judge.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evaluation_records.io import run_digest
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.acceptance import (
    verify_signed_judge_verification_receipt,
)
from invarlock.judge_measurements.analysis import (
    analyze_measurements,
    decode_analysis_policy,
)
from invarlock.judge_measurements.evidence import DECISION_SCOPE, object_sha256
from invarlock.judge_measurements.native_capture import validate_native_capture
from invarlock.judge_measurements.native_recipe import finalize_native_plan
from tests.core.test_native_judge_transaction import _completed_measurements, _recipe
from tests.integration.test_container_front_door_journey import (
    _assert_runtime_device,
    _module,
    _normalized_config_id,
    _private_key,
    _request,
    _run_host,
    _runtime_device,
    _tiny_checkpoint,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("INVARLOCK_RUN_CONTAINER_SMOKE") != "1",
    reason="set INVARLOCK_RUN_CONTAINER_SMOKE=1 for actual native judge OCI capture",
)

SIGNER = "offline-container-judge-test"
VERIFIER = "native-judge-container-recipient"


def test_native_judge_container_capture_evaluate_verify_report(tmp_path, monkeypatch):
    if os.environ.get("INVARLOCK_CONTAINER_SMOKE_INSTALLED_WHEEL") == "1":
        package = _module("invarlock")
        assert (
            not Path(package.__file__)
            .resolve()
            .is_relative_to(Path(__file__).resolve().parents[2])
        )
    engine = os.environ.get("INVARLOCK_CONTAINER_ENGINE", "docker")
    image = os.environ.get("INVARLOCK_RUNTIME_IMAGE", "invarlock-runtime:local")
    inspected = subprocess.run(
        [engine, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    image_digest = _normalized_config_id(inspected.stdout.strip())
    device = _runtime_device()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, checkpoint_digest, tokenizer_digest = _tiny_checkpoint(workspace)
    request = _request(
        workspace,
        checkpoint_digest=checkpoint_digest,
        tokenizer_digest=tokenizer_digest,
        output="evidence",
    )
    recipe = _recipe(["tiny-1"])
    # Predeclared single-case bounds exercise plumbing, not statistical quality.
    recipe["analysis"]["allowed_degradation"] = "1"
    (workspace / "inputs/policy.json").write_text(json.dumps(recipe))
    document = yaml.safe_load(request.read_text())
    document["comparison"].update(
        metric="judge",
        judge={"workspace": "judge-workspace", "signer_identity": SIGNER},
    )
    request.write_text(yaml.safe_dump(document))
    signer, fingerprint = _private_key(tmp_path / "signer.pem")
    verifier, verifier_fingerprint = _private_key(tmp_path / "verifier.pem")
    recipient_path = tmp_path / "recipient.json"
    admitted = []

    def fixed_measurements(*, plan, baseline_run, subject_run, **kwargs):
        # Native capture must exist and independently validate before judging.
        frozen = json.loads(
            (workspace / "judge-workspace/native_capture.json").read_bytes()
        )
        assert frozen["normalized_request"]["execution"]["mode"] == "run"
        assert frozen["normalized_request"]["comparison"]["metric"] == "judge"
        assert validate_native_capture(frozen) == (baseline_run, subject_run)
        for role in ("baseline", "subject"):
            _assert_runtime_device(
                base64.b64decode(frozen[role]["runtime_manifest"], validate=True),
                base64.b64decode(frozen[role]["provider_receipt"], validate=True),
                device=device,
                image_digest=image_digest,
            )
        for run in (baseline_run, subject_run):
            assert run["source"]["name"] == "invarlock-native-judge"
            assert len(run["records"]) == 1
            assert run["records"][0]["context"]["runtime_digest"] == image_digest
        measurements = _completed_measurements(
            plan=plan, baseline_run=baseline_run, subject_run=subject_run, **kwargs
        )
        expected_plan, policy = finalize_native_plan(recipe, baseline_run, subject_run)
        assert plan == expected_plan
        analysis = analyze_measurements(
            plan,
            measurements,
            decode_analysis_policy(policy, plan=plan),
            baseline_run=baseline_run,
            subject_run=subject_run,
        ).to_dict()
        assert analysis["decision"] == "pass"
        # Author recipient anchors before publication; do not copy its envelope.
        recipient = {
            "format": "invarlock/judge-measurement-recipient-policy-v1",
            "decision_scope": DECISION_SCOPE,
            "intended_subject": subject_run["artifact_digest"],
            "required_metric_name": policy["metric_name"],
            "trusted_signer": {"identity": SIGNER, "public_key_sha256": fingerprint},
            "bindings": {
                "baseline_run_sha256": run_digest(baseline_run),
                "subject_run_sha256": run_digest(subject_run),
                "case_set_sha256": plan["case_set_sha256"],
                "plan_sha256": object_sha256(plan),
                "measurements_sha256": object_sha256(measurements),
                "analysis_policy_sha256": object_sha256(policy),
                "analysis_result_sha256": object_sha256(analysis),
                "native_capture_sha256": object_sha256(frozen),
            },
            "required_decision": "pass",
        }
        recipient_path.write_text(json.dumps(recipient))
        admitted.append((frozen, recipient))
        return measurements

    # These are test-only collection seams. OCI execution is never substituted.
    monkeypatch.setattr(
        native_workflow,
        "collection_preflight",
        lambda _: {"fixture_measurements": True, "network_calls": 0},
    )
    monkeypatch.setattr(native_workflow, "collect_frozen", fixed_measurements)
    evaluated = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signer),
            "--container-engine",
            engine,
            "--runtime-image",
            image_digest,
            "--runtime-device",
            device,
            "--json",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    value = json.loads(evaluated.stdout)
    assert value["kind"] == "judge" and value["execution_mode"] == "run"
    assert value["authentication"] == "signed" and value["decision"] == "pass"
    assert len(admitted) == 1
    frozen, recipient = admitted[0]
    evidence = workspace / "evidence"
    assert json.loads((evidence / "native_capture.json").read_bytes()) == frozen
    assert (
        object_sha256(json.loads((evidence / "native_capture.json").read_bytes()))
        == recipient["bindings"]["native_capture_sha256"]
    )

    receipt_path = tmp_path / "verification.receipt.json"
    verified = _run_host(
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(recipient_path),
            "--receipt",
            str(receipt_path),
            "--verifier-signing-key",
            str(verifier),
            "--verifier-identity",
            VERIFIER,
            "--json",
        ]
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    verification = json.loads(verified.stdout)
    assert (
        verification["authenticated"]
        and verification["replayed"]
        and verification["accepted"]
    )
    receipt = verify_signed_judge_verification_receipt(
        receipt_path,
        expected_verifier_identity=VERIFIER,
        expected_verifier_fingerprint=verifier_fingerprint,
        expected_recipient_policy_sha256=object_sha256(recipient),
    )
    assert receipt.ok, receipt.errors
    assert receipt.result.to_dict()["bindings"][
        "native_capture_sha256"
    ] == object_sha256(frozen)
    reported = _run_host(["report", str(evidence), "--json"])
    assert reported.returncode == 0, reported.stdout + reported.stderr
    report = json.loads(reported.stdout)
    assert report["analysis"]["decision"] == "pass"
    assert (
        report["assurance"]["authentication"]
        == "signature_present_not_recipient_authorized"
    )
    html_path = tmp_path / "report.html"
    rendered = _run_host(["report", str(evidence), "--html", str(html_path)])
    assert rendered.returncode == 0, rendered.stdout + rendered.stderr
    assert html_path.is_file()

    changed = tmp_path / "tampered-evidence"
    shutil.copytree(evidence, changed)
    tampered = copy.deepcopy(frozen)
    tampered["subject"]["scoring_observation"] = base64.b64encode(b"{}").decode()
    (changed / "native_capture.json").write_text(json.dumps(tampered))
    rejected = _run_host(
        ["verify", str(changed), "--trust-profile", str(recipient_path), "--json"]
    )
    assert rejected.returncode == 4, rejected.stdout + rejected.stderr
    assert not json.loads(rejected.stdout)["accepted"]

    if destination := os.environ.get("INVARLOCK_OCI_ISOLATION_RESULTS"):
        retained = Path(destination) / "native-judge"
        retained.mkdir(parents=True)
        shutil.copytree(evidence, retained / "evidence")
        for path in (recipient_path, receipt_path, html_path):
            shutil.copy2(path, retained / path.name)
        (retained / "verification.json").write_text(verified.stdout)
        (retained / "report.json").write_text(reported.stdout)
        (retained / "scope.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "image_digest": image_digest,
                    "runtime_device": device,
                    "model_scope": "tiny deterministic hardware plumbing fixture",
                    "native_capture_sha256": object_sha256(frozen),
                    "test_source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "judge_measurements": "fixed offline test fixture; no live judge calls",
                    "answer_collection": "actual native OCI provider execution",
                    "tampered_native_capture_rejected": True,
                },
                indent=2,
            )
            + "\n"
        )
