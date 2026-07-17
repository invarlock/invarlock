from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    HFSnapshotArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderPluginIdentity,
    RuntimeScoringRecord,
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.core.scorer_extension import (
    SCORER_EXTENSION_ABI_VERSION,
    ScorerExtensionBinding,
    ScorerExtensionDescriptor,
    ScorerExtensionRegistry,
    ScorerExtensionResult,
    ScorerReplayRequest,
    build_scorer_binding,
    build_scorer_result,
    scorer_binding_payload,
    scorer_configuration_schema_sha256,
)
from invarlock.evaluation_run import EvaluationRunResult, load_runtime_side_evidence
from invarlock.evaluation_transaction import (
    EvaluationTransactionError,
    _prepare_output_parent,
    _revalidate_output_parent,
    evaluate_request_file,
)
from invarlock.evidence_pack_contract import (
    MAX_OBSERVATION_BYTES,
    canonical_json_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.evidence_reporting import render_evidence
from invarlock.evidence_verification import verify_evidence
from invarlock.runtime_import_authoring import (
    RuntimeImportSideEvidence,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

INVARLOCK_SCORER_EXTENSION_ABI = SCORER_EXTENSION_ABI_VERSION


def _key(path: Path) -> tuple[Path, str]:
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


def _input_anchors(evidence: Path) -> dict[str, str]:
    return {
        role: json.loads(
            (evidence / "inputs" / f"{role}.json").read_text(encoding="utf-8")
        )["digest"]
        for role in ("baseline", "subject", "dataset")
    }


def _settings() -> dict[str, object]:
    return {
        "batch_size": 1,
        "checkpoint_tree_sha256": "a" * 64,
        "context_length": 128,
        "immutable_revision": "b" * 40,
        "max_output_tokens": 16,
        "offline": True,
        "seed": 0,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": "c" * 64,
    }


def _side_evidence(
    root: Path,
    *,
    role: str,
    model_id: str,
    schedule,
    records: tuple[RuntimeScoringRecord, ...],
    runtime_digest: str,
    policy_digest: str,
    capability_metrics: tuple[str, ...] | None = None,
    checkpoint_digest: str = "a" * 64,
) -> RuntimeImportSideEvidence:
    identity = HFSnapshotArtifactIdentity(
        model_id=model_id,
        immutable_revision="b" * 40,
        checkpoint_tree_sha256=checkpoint_digest,
        tokenizer_metadata_sha256="c" * 64,
    )
    capabilities = HFTransformersProvider().capabilities()
    if capability_metrics is not None:
        capabilities = replace(
            capabilities,
            metrics=capability_metrics,  # type: ignore[arg-type]
        )
    return write_runtime_import_side(
        root,
        role=role,  # type: ignore[arg-type]
        schedule=schedule,
        policy_digest=policy_digest,
        artifact_identity=identity,
        records=records,
        plugin=RuntimeProviderPluginIdentity(
            name="hf_transformers",
            distribution="invarlock",
            distribution_version="test",
        ),
        backend=RuntimeBackendIdentity(
            name="transformers",
            version="test",
            source_sha256="d" * 64,
            binary_sha256=None,
            build_sha256=None,
        ),
        capabilities=capabilities,
        execution_settings=RuntimeExecutionSettings(
            seed=0,
            context_length=128,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=30,
            allow_network=False,
        ),
        device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
        runtime_image_ref=f"ghcr.io/invarlock/runtime@{runtime_digest}",
        runtime_image_digest=runtime_digest,
        generated_at_utc="2026-07-16T12:00:00+00:00",
    )


def _materialize_request(
    tmp_path: Path,
    *,
    capability_metrics: tuple[str, ...] | None = None,
    scorer_binding: ScorerExtensionBinding | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> dict[str, object]:
    (tmp_path / "imports").mkdir()
    schedule = build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "acceptance",
            "config_name": None,
            "revision": "e" * 40,
            "split": "validation",
        },
        records=[
            {"record_id": "one", "input_text": "Return A", "expected_output": "A"},
            {"record_id": "two", "input_text": "Return B", "expected_output": "B"},
        ],
    )
    schedule_path = tmp_path / "inputs/schedule.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))
    policy = {
        "resolved_policy": {
            "metrics": (
                {
                    "scorer_extension": {
                        "scorer_id": scorer_binding.scorer_id,
                        "scorer_version": scorer_binding.scorer_version,
                        "descriptor_sha256": scorer_binding.descriptor_sha256,
                        "configuration_sha256": scorer_binding.configuration_sha256,
                        "delta_min_pp": -100.0,
                    }
                }
                if scorer_binding is not None
                else {"exact_match": {"delta_min_pp": -100.0}}
            )
        }
    }
    policy_path = tmp_path / "inputs/policy.json"
    policy_path.write_bytes(canonical_json_bytes(policy))
    observation_path = tmp_path / "inputs/subject-variance.json"
    observation_path.write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/diagnostic-observation-v1",
                "input_sha256": "f" * 64,
                "kind": "variance",
                "status": "observation",
            }
        )
    )
    policy_digest = sha256_digest(policy_path.read_bytes())
    runtime_digests = {
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    }

    def records(outputs: tuple[str, str]) -> tuple[RuntimeScoringRecord, ...]:
        return tuple(
            RuntimeScoringRecord(
                record_id=scheduled.record_id,
                input_sha256=scheduled.input_sha256,
                status="ok",
                output_text=output,
                output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            )
            for scheduled, output in zip(schedule.records, outputs, strict=True)
        )

    baseline = _side_evidence(
        tmp_path / "imports/baseline",
        role="baseline",
        model_id="org/baseline",
        schedule=schedule,
        records=records(("A", "B")),
        runtime_digest=runtime_digests["baseline"],
        policy_digest=policy_digest,
        capability_metrics=capability_metrics,
    )
    subject = _side_evidence(
        tmp_path / "imports/subject",
        role="subject",
        model_id="org/subject",
        schedule=schedule,
        records=records(("A", "wrong")),
        runtime_digest=runtime_digests["subject"],
        policy_digest=policy_digest,
        capability_metrics=capability_metrics,
    )
    paired = write_runtime_import_paired_records(
        tmp_path / "imports/paired-records.json",
        schedule=schedule,
        metric=(
            scorer_binding.scorer_id if scorer_binding is not None else "exact_match"
        ),
        baseline=baseline,
        subject=subject,
        scorer_binding=scorer_binding,
        scorer_registry=scorer_registry,
    )
    records_path = paired.path

    def side(model_id: str) -> dict[str, object]:
        return {
            "artifact": {
                "model_id": model_id,
                "locator": f"hf://{model_id}@{'b' * 40}",
            },
            "runtime": {
                "provider": "hf_transformers",
                "settings": _settings(),
            },
        }

    def imported(side_name: str) -> dict[str, str]:
        prefix = f"imports/{side_name}"
        return {
            "identity": f"{prefix}/model-artifact.identity.json",
            "receipt": f"{prefix}/runtime-provider.receipt.json",
            "observation": f"{prefix}/runtime-scoring.observation.json",
            "run_report": f"{prefix}/report.json",
            "runtime_manifest": f"{prefix}/runtime.manifest.json",
            "runtime_config": f"{prefix}/run.yaml",
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("org/baseline"),
            "subject": side("org/subject"),
            "dataset": "inputs/schedule.json",
            "policy": "inputs/policy.json",
            "task": "text_causal",
        },
        "execution": {
            "mode": "import",
            "records": "imports/paired-records.json",
            "schedule": "inputs/schedule.json",
            "baseline": imported("baseline"),
            "subject": imported("subject"),
        },
        "observations": [
            {
                "id": "subject-variance",
                "kind": "variance",
                "scope": "subject",
                "path": "inputs/subject-variance.json",
            }
        ],
        "output": {"evidence": "artifacts/evidence"},
    }
    comparison = request["comparison"]
    assert isinstance(comparison, dict)
    if scorer_binding is None:
        comparison["metric"] = "exact_match"
    else:
        comparison["scorer_extension"] = scorer_binding_payload(scorer_binding)
    request_path = tmp_path / "request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    return {
        "request": request_path,
        "policy": policy_path,
        "runtime_digests": runtime_digests,
        "records": records_path,
    }


_TEXT_SCORER_CONFIGURATION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
}


class _CasefoldTextScorer:
    abi_version = SCORER_EXTENSION_ABI_VERSION

    def descriptor(self) -> ScorerExtensionDescriptor:
        return ScorerExtensionDescriptor(
            scorer_id="example.casefold_exact",
            scorer_version="1.0.0",
            supported_tasks=("text_causal",),
            supported_input_kinds=("text",),
            supported_output_kinds=("text",),
            required_facts=("expected_output", "output_text", "output_sha256"),
            configuration_schema_sha256=scorer_configuration_schema_sha256(
                _TEXT_SCORER_CONFIGURATION_SCHEMA
            ),
        )

    def configuration_schema(self) -> dict[str, object]:
        return dict(_TEXT_SCORER_CONFIGURATION_SCHEMA)

    def replay(self, request: ScorerReplayRequest) -> ScorerExtensionResult:
        return build_scorer_result(
            request,
            [
                float(
                    str(record.facts["output_text"]).casefold()
                    == str(record.facts["expected_output"]).casefold()
                )
                for record in request.records
            ],
        )


class _PhaseDriftTextScorer(_CasefoldTextScorer):
    def __init__(self) -> None:
        self._calls = 0

    def replay(self, request: ScorerReplayRequest) -> ScorerExtensionResult:
        self._calls += 1
        if self._calls <= 4:
            return super().replay(request)
        return build_scorer_result(
            request,
            [0.0 for _record in request.records],
        )


def _text_scorer_registry_and_binding() -> tuple[
    ScorerExtensionRegistry, ScorerExtensionBinding
]:
    scorer = _CasefoldTextScorer()
    return (
        ScorerExtensionRegistry(allow_installed=False, authorized=(scorer,)),
        build_scorer_binding(scorer.descriptor(), {}),
    )


def _install_casefold_scorer_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "installed-scorer"
    distribution = site / "invarlock_casefold_scorer-1.0.0.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: invarlock-casefold-scorer\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (distribution / "entry_points.txt").write_text(
        "[invarlock.scorers]\n"
        f"example.casefold_exact = {__name__}:_CasefoldTextScorer\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()


def test_text_scorer_extension_evaluate_verify_report_transaction(
    tmp_path: Path,
) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=registry
    )
    evidence_key, evidence_fingerprint = _key(tmp_path / "evidence.pem")
    verifier_key, _verifier_fingerprint = _key(tmp_path / "verifier.pem")

    evaluated = evaluate_request_file(
        material["request"],  # type: ignore[arg-type]
        signing_key_path=evidence_key,
        scorer_registry=registry,
    )
    evidence = evaluated.evidence_path
    normalized = json.loads((evidence / "request.json").read_bytes())
    assert "metric" not in normalized["comparison"]
    assert normalized["comparison"]["scorer_extension"] == scorer_binding_payload(
        binding
    )
    paired = json.loads((evidence / "records/paired-records.json").read_bytes())
    assert paired["metric"] == binding.scorer_id
    assert set(paired["scorer_replay"]) == {"baseline", "subject"}
    report_payload = json.loads(
        (evidence / "reports/evaluation.report.json").read_bytes()
    )
    assert report_payload["baseline"]["mean_score"] == 1.0
    assert report_payload["subject"]["mean_score"] == 0.5
    assert report_payload["comparison"] == {
        "kind": "scorer_extension_delta_pp",
        "minimum": -100.0,
        "value": -50.0,
    }
    assert report_payload["verdict"] == "pass"

    anchors = _input_anchors(evidence)
    runtimes = material["runtime_digests"]
    assert isinstance(runtimes, dict)
    missing_registry = verify_comparison_evidence(
        evidence,
        policy_path=material["policy"],  # type: ignore[arg-type]
        expected_artifact_digests={
            "baseline": anchors["baseline"],
            "subject": anchors["subject"],
        },
        expected_schedule_digest=anchors["dataset"],
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=evidence_fingerprint,
    )
    assert missing_registry.payload["ok"] is False
    assert any(
        "explicitly authorized scorer registry" in str(error)
        for error in missing_registry.payload["errors"]
    )
    verified = verify_evidence(
        evidence,
        policy_path=material["policy"],  # type: ignore[arg-type]
        expected_baseline_artifact=anchors["baseline"],
        expected_subject_artifact=anchors["subject"],
        expected_schedule=anchors["dataset"],
        expected_baseline_runtime=str(runtimes["baseline"]),
        expected_subject_runtime=str(runtimes["subject"]),
        expected_signer=evidence_fingerprint,
        receipt_path=tmp_path / "extension.receipt.json",
        verifier_signing_key_path=verifier_key,
        verifier_identity="invarlock-verifier/extension-fixture",
        scorer_registry=registry,
    )
    assert verified.payload["ok"] is True
    rendered = render_evidence(evidence)
    assert "Extension scorer delta (pp)" in rendered.text
    assert binding.scorer_id in rendered.text


def test_text_scorer_extension_fails_closed_without_authorized_registry(
    tmp_path: Path,
) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=registry
    )
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")
    with pytest.raises(
        EvaluationTransactionError, match="explicitly authorized scorer registry"
    ):
        evaluate_request_file(
            material["request"],  # type: ignore[arg-type]
            signing_key_path=evidence_key,
        )


@pytest.mark.parametrize("pin", ["descriptor_sha256", "configuration_sha256"])
def test_text_scorer_extension_rejects_policy_pin_mismatch(
    tmp_path: Path, pin: str
) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=registry
    )
    policy_path = material["policy"]
    assert isinstance(policy_path, Path)
    policy = json.loads(policy_path.read_bytes())
    policy["resolved_policy"]["metrics"]["scorer_extension"][pin] = "f" * 64
    policy_path.write_bytes(canonical_json_bytes(policy))
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")
    with pytest.raises(EvaluationTransactionError, match="pin does not match"):
        evaluate_request_file(
            material["request"],  # type: ignore[arg-type]
            signing_key_path=evidence_key,
            scorer_registry=registry,
        )


def test_text_scorer_extension_rejects_cross_phase_state_drift(
    tmp_path: Path,
) -> None:
    fixture_registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=fixture_registry
    )
    drift_registry = ScorerExtensionRegistry(
        allow_installed=False,
        authorized=(_PhaseDriftTextScorer(),),
    )
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")

    with pytest.raises(
        EvaluationTransactionError,
        match="publication paired records do not match transaction-derived records",
    ):
        evaluate_request_file(
            material["request"],  # type: ignore[arg-type]
            signing_key_path=evidence_key,
            scorer_registry=drift_registry,
        )


def test_installed_text_scorer_runs_through_public_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=fixture_registry
    )
    _install_casefold_scorer_entry_point(tmp_path, monkeypatch)
    evidence_key, evidence_fingerprint = _key(tmp_path / "evidence.pem")
    verifier_key, _verifier_fingerprint = _key(tmp_path / "verifier.pem")
    runner = CliRunner()

    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            str(material["request"]),
            "--signing-key",
            str(evidence_key),
            "--allow-installed-scorers",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.stdout
    evidence = tmp_path / "artifacts/evidence"
    anchors = _input_anchors(evidence)
    runtimes = material["runtime_digests"]
    assert isinstance(runtimes, dict)
    receipt = tmp_path / "installed-scorer.receipt.json"

    verified = runner.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--policy",
            str(material["policy"]),
            "--expected-baseline-artifact",
            anchors["baseline"],
            "--expected-subject-artifact",
            anchors["subject"],
            "--expected-schedule",
            anchors["dataset"],
            "--expected-baseline-runtime",
            str(runtimes["baseline"]),
            "--expected-subject-runtime",
            str(runtimes["subject"]),
            "--expected-signer",
            evidence_fingerprint,
            "--receipt",
            str(receipt),
            "--verifier-signing-key",
            str(verifier_key),
            "--verifier-identity",
            "invarlock-verifier/installed-scorer",
            "--allow-installed-scorers",
        ],
    )
    assert verified.exit_code == 0, verified.stdout
    assert receipt.is_file()

    rendered = runner.invoke(app, ["report", str(evidence)])
    assert rendered.exit_code == 0, rendered.stdout
    assert "Extension scorer delta (pp)" in rendered.stdout


@pytest.mark.parametrize(
    "relative",
    ["records/paired-records.json", "reports/evaluation.report.json"],
)
def test_text_scorer_extension_rejects_stored_result_tampering(
    tmp_path: Path, relative: str
) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=registry
    )
    evidence_key, evidence_fingerprint = _key(tmp_path / "evidence.pem")
    evidence = evaluate_request_file(
        material["request"],  # type: ignore[arg-type]
        signing_key_path=evidence_key,
        scorer_registry=registry,
    ).evidence_path
    target = evidence / relative
    payload = json.loads(target.read_bytes())
    if relative.startswith("records/"):
        payload["records"][0]["subject"]["score"] = 0.25
    else:
        payload["comparison"]["value"] = 25.0
    target.chmod(0o644)
    target.write_bytes(canonical_json_bytes(payload))

    anchors = _input_anchors(evidence)
    runtimes = material["runtime_digests"]
    assert isinstance(runtimes, dict)
    verified = verify_comparison_evidence(
        evidence,
        policy_path=material["policy"],  # type: ignore[arg-type]
        expected_artifact_digests={
            "baseline": anchors["baseline"],
            "subject": anchors["subject"],
        },
        expected_schedule_digest=anchors["dataset"],
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=evidence_fingerprint,
        scorer_registry=registry,
    )
    assert verified.payload["ok"] is False
    assert verified.payload["errors"]


def test_cli_import_verify_report_is_a_real_signed_transaction(tmp_path: Path) -> None:
    material = _materialize_request(tmp_path)
    evidence_key, evidence_fingerprint = _key(tmp_path / "evidence.pem")
    verifier_key, verifier_fingerprint = _key(tmp_path / "verifier.pem")
    runner = CliRunner()

    evaluated = runner.invoke(
        app,
        ["evaluate", str(material["request"]), "--signing-key", str(evidence_key)],
    )
    assert evaluated.exit_code == 0, evaluated.stdout
    evidence = tmp_path / "artifacts/evidence"
    normalized_request = json.loads(
        (evidence / "request.json").read_text(encoding="utf-8")
    )
    assert normalized_request["comparison"]["baseline"]["artifact"]["locator"] == (
        f"hf://org/baseline@{'b' * 40}"
    )
    assert str(tmp_path) not in json.dumps(normalized_request)
    assert normalized_request["execution"]["baseline"]["runtime_manifest"] == (
        "providers/baseline/runtime.manifest.json"
    )
    assert normalized_request["observations"] == [
        {
            "id": "subject-variance",
            "kind": "variance",
            "scope": "subject",
            "payload_digest": sha256_digest(
                (tmp_path / "inputs/subject-variance.json").read_bytes()
            ),
        }
    ]
    report_payload = json.loads(
        (evidence / "reports/evaluation.report.json").read_bytes()
    )
    assert report_payload["comparison"] == {
        "kind": "exact_match_delta_pp",
        "minimum": -100.0,
        "value": -50.0,
    }
    paired_binary = report_payload["paired_binary"]
    assert paired_binary["baseline_pass_subject_fail"] == 1
    assert paired_binary["baseline_fail_subject_pass"] == 0
    assert paired_binary["both_pass"] == 1
    assert paired_binary["both_fail"] == 0
    assert paired_binary["discordant_pairs"] == 1
    assert paired_binary["mcnemar_exact_two_sided_p_value"] == 1.0
    assert paired_binary["effect_size_pp"] == -50.0
    paired_interval = paired_binary["effect_size_confidence_interval"]
    assert paired_interval["lower_pp"] <= -50.0 <= paired_interval["upper_pp"]
    assert report_payload["uncertainty"]["lower"] == paired_interval["lower_pp"]
    assert report_payload["uncertainty"]["upper"] == paired_interval["upper_pp"]
    assert report_payload["verdict"] == "pass"
    receipt = tmp_path / "verification.receipt.json"
    runtime_digests = material["runtime_digests"]
    assert isinstance(runtime_digests, dict)
    input_anchors = _input_anchors(evidence)

    verified = runner.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--policy",
            str(material["policy"]),
            "--expected-baseline-artifact",
            input_anchors["baseline"],
            "--expected-subject-artifact",
            input_anchors["subject"],
            "--expected-schedule",
            input_anchors["dataset"],
            "--expected-baseline-runtime",
            str(runtime_digests["baseline"]),
            "--expected-subject-runtime",
            str(runtime_digests["subject"]),
            "--expected-signer",
            evidence_fingerprint,
            "--receipt",
            str(receipt),
            "--verifier-signing-key",
            str(verifier_key),
            "--verifier-identity",
            "invarlock-verifier/fixture",
        ],
    )
    assert verified.exit_code == 0, verified.stdout
    independent = verify_signed_verification_receipt(
        receipt,
        evidence,
        policy_path=material["policy"],
        expected_artifact_digests={
            "baseline": input_anchors["baseline"],
            "subject": input_anchors["subject"],
        },
        expected_schedule_digest=input_anchors["dataset"],
        expected_runtime_digests=runtime_digests,
        expected_pack_signer_fingerprint=evidence_fingerprint,
        expected_verifier_identity="invarlock-verifier/fixture",
        expected_verifier_fingerprint=verifier_fingerprint,
    )
    assert independent.ok is True

    rendered = runner.invoke(app, ["report", str(evidence)])
    assert rendered.exit_code == 0, rendered.stdout
    assert "PASS" in rendered.stdout
    assert "subject-variance" in rendered.stdout
    assert "complete acceptance calculation" in rendered.stdout


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"status": "observation"}\n', "canonical JSON"),
        (b'{"status":"observation","status":"changed"}\n', "duplicate key"),
        (b'["observation"]\n', "JSON object"),
        (
            b'{"blob":"' + (b"x" * MAX_OBSERVATION_BYTES) + b'"}\n',
            "size limit",
        ),
    ],
)
def test_evaluate_rejects_invalid_observation_before_import_replay(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    material = _materialize_request(tmp_path)
    observation = tmp_path / "inputs/subject-variance.json"
    observation.write_bytes(payload)
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")

    with pytest.raises(EvaluationTransactionError, match=message):
        evaluate_request_file(
            material["request"],  # type: ignore[arg-type]
            signing_key_path=evidence_key,
        )

    assert not (tmp_path / "artifacts/evidence").exists()


@pytest.mark.parametrize("runtime_binding_matches", [True, False])
def test_run_executor_converges_through_the_same_host_verifier_and_publication(
    tmp_path: Path,
    runtime_binding_matches: bool,
) -> None:
    """Live side workers cannot bypass the import-grade host convergence path."""

    dataset_bytes = (
        b'{"id":"one","prompt":"Return A","expected":"A"}\n'
        b'{"id":"two","prompt":"Return B","expected":"B"}\n'
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset_path = inputs / "records.jsonl"
    dataset_path.write_bytes(dataset_bytes)
    dataset_request = LocalDatasetRequest(
        path=dataset_path,
        sha256=hashlib.sha256(dataset_bytes).hexdigest(),
        format="jsonl",
        name="host-convergence",
        split="validation",
        input_field="prompt",
        expected_output_field="expected",
        id_field="id",
    )
    schedule = prepare_local_evaluation_schedule_bytes(dataset_request, dataset_bytes)
    policy_path = inputs / "policy.json"
    policy_path.write_bytes(
        canonical_json_bytes(
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -100.0}}}}
        )
    )
    policy_digest = sha256_digest(policy_path.read_bytes())
    runtime_digests = {
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    }

    def records(outputs: tuple[str, str]) -> tuple[RuntimeScoringRecord, ...]:
        return tuple(
            RuntimeScoringRecord(
                record_id=scheduled.record_id,
                input_sha256=scheduled.input_sha256,
                status="ok",
                output_text=output,
                output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            )
            for scheduled, output in zip(schedule.records, outputs, strict=True)
        )

    worker_root = tmp_path / "worker-output"
    worker_root.mkdir()
    checkpoint_digests: dict[str, str] = {}
    for role in ("baseline", "subject"):
        checkpoint = tmp_path / "models" / role
        checkpoint.mkdir(parents=True)
        checkpoint.joinpath("config.json").write_text(
            json.dumps({"model_type": "gpt2", "role": role}),
            encoding="utf-8",
        )
        checkpoint_digests[role] = checkpoint_tree_sha256(checkpoint).removeprefix(
            "sha256:"
        )
    baseline = _side_evidence(
        worker_root / "baseline",
        role="baseline",
        model_id="org/baseline",
        schedule=schedule,
        records=records(("A", "B")),
        runtime_digest=runtime_digests["baseline"],
        policy_digest=policy_digest,
        checkpoint_digest=checkpoint_digests["baseline"],
    )
    subject = _side_evidence(
        worker_root / "subject",
        role="subject",
        model_id="org/subject",
        schedule=schedule,
        records=records(("A", "wrong")),
        runtime_digest=runtime_digests["subject"],
        policy_digest=policy_digest,
        checkpoint_digest=checkpoint_digests["subject"],
    )

    def side(role: str) -> dict[str, object]:
        settings = _settings()
        settings["checkpoint_tree_sha256"] = checkpoint_digests[role]
        return {
            "artifact": {
                "path": f"models/{role}",
                "model_id": f"org/{role}",
                "locator": f"hf://org/{role}@{'b' * 40}",
            },
            "runtime": {"provider": "hf_transformers", "settings": settings},
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline"),
            "subject": side("subject"),
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": dataset_request.sha256,
                "format": "jsonl",
                "name": dataset_request.name,
                "split": dataset_request.split,
                "input_field": dataset_request.input_field,
                "expected_output_field": dataset_request.expected_output_field,
                "id_field": dataset_request.id_field,
            },
            "policy": "inputs/policy.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {"mode": "run"},
        "output": {"evidence": "artifacts/evidence"},
    }
    request_path = tmp_path / "run-request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")

    class Executor:
        def execute(self, _request, *, registry, schedule_bytes, policy_digest):
            del registry
            assert (tmp_path / "artifacts").is_dir()
            assert not (tmp_path / "artifacts/evidence").exists()
            assert schedule_bytes == canonical_runtime_behavioral_schedule_json(
                schedule
            )
            assert policy_digest == sha256_digest(policy_path.read_bytes())
            return EvaluationRunResult(
                baseline=load_runtime_side_evidence(baseline.directory),
                subject=load_runtime_side_evidence(subject.directory),
                baseline_runtime_digest=runtime_digests["baseline"],
                subject_runtime_digest=runtime_digests["subject"],
            )

    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")
    expected_runtime_digests = (
        runtime_digests
        if runtime_binding_matches
        else {
            "baseline": "sha256:" + "8" * 64,
            "subject": "sha256:" + "9" * 64,
        }
    )
    if not runtime_binding_matches:
        with pytest.raises(
            EvaluationTransactionError,
            match="validated runtime digest does not match preflight",
        ):
            evaluate_request_file(
                request_path,
                signing_key_path=evidence_key,
                runtime_executor=Executor(),  # type: ignore[arg-type]
                runtime_image_digests=expected_runtime_digests,
            )
        assert not (tmp_path / "artifacts/evidence").exists()
        return

    result = evaluate_request_file(
        request_path,
        signing_key_path=evidence_key,
        runtime_executor=Executor(),  # type: ignore[arg-type]
        runtime_image_digests=expected_runtime_digests,
    )

    assert result.evidence_path == (tmp_path / "artifacts/evidence").resolve()
    assert result.pack_manifest_digest == (
        "sha256:"
        + hashlib.sha256(
            result.evidence_path.joinpath("manifest.json").read_bytes()
        ).hexdigest()
    )
    assert (
        result.evidence_path / "providers/baseline/runtime-provider.receipt.json"
    ).is_file()
    assert (
        result.evidence_path / "providers/subject/runtime-provider.receipt.json"
    ).is_file()
    normalized = json.loads((result.evidence_path / "request.json").read_bytes())
    preparation = normalized["comparison"]["dataset"]
    assert preparation == {
        "format_version": "invarlock/local-jsonl-preparation-v1",
        "source_sha256": dataset_request.sha256,
        "source_format": "jsonl",
        "name": "host-convergence",
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "expected",
        "id_field": "id",
        "content_role": None,
        "content_id_field": None,
        "content_sha256_field": None,
        "content_byte_length_field": None,
        "content_media_type_field": None,
        "limit": None,
        "selected_record_count": 2,
    }
    assert "path" not in preparation
    assert str(tmp_path) not in json.dumps(normalized)


def test_import_rejects_attacker_supplied_paired_scores(tmp_path: Path) -> None:
    material = _materialize_request(tmp_path)
    records_path = material["records"]
    assert isinstance(records_path, Path)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["records"][0]["subject"]["score"] = 0.0
    records_path.write_bytes(canonical_json_bytes(records))
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")

    result = CliRunner().invoke(
        app,
        ["evaluate", str(material["request"]), "--signing-key", str(evidence_key)],
    )

    assert result.exit_code == 2
    assert "do not equal verifier-derived pairs" in result.stdout
    assert not (tmp_path / "artifacts/evidence").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "subject"),
        ("policy_digest", "sha256:" + "9" * 64),
    ],
)
def test_import_rejects_semantically_false_runtime_config(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    material = _materialize_request(tmp_path)
    config_path = tmp_path / "imports/baseline/run.yaml"
    config = json.loads(config_path.read_bytes())
    config[field] = value
    config_path.write_bytes(canonical_json_bytes(config))
    manifest_path = tmp_path / "imports/baseline/runtime.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["config"]["sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")

    result = CliRunner().invoke(
        app,
        ["evaluate", str(material["request"]), "--signing-key", str(evidence_key)],
    )

    assert result.exit_code == 2
    assert "runtime config does not bind" in result.stdout
    assert not (tmp_path / "artifacts/evidence").exists()


def test_import_rejects_metric_not_declared_by_provider_receipts(
    tmp_path: Path,
) -> None:
    material = _materialize_request(
        tmp_path,
        capability_metrics=("exact_match",),
    )
    request_path = material["request"]
    assert isinstance(request_path, Path)
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    request["comparison"]["metric"] = "normalized_nll_per_utf8_byte"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    policy_path = material["policy"]
    assert isinstance(policy_path, Path)
    policy_path.write_bytes(
        canonical_json_bytes(
            {
                "resolved_policy": {
                    "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.05}}
                }
            }
        )
    )
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")

    result = CliRunner().invoke(
        app,
        ["evaluate", str(request_path), "--signing-key", str(evidence_key)],
    )

    assert result.exit_code == 2
    assert "does not declare requested metric" in result.stdout
    assert not (tmp_path / "artifacts/evidence").exists()


def test_import_repeats_no_follow_reads_after_request_load(tmp_path: Path) -> None:
    material = _materialize_request(tmp_path)
    records_path = material["records"]
    assert isinstance(records_path, Path)
    original = records_path.with_suffix(".real.json")
    records_path.rename(original)
    records_path.symlink_to(original.name)
    evidence_key, _fingerprint = _key(tmp_path / "evidence.pem")

    result = CliRunner().invoke(
        app,
        ["evaluate", str(material["request"]), "--signing-key", str(evidence_key)],
    )

    assert result.exit_code == 2
    assert "symlink" in result.stdout or "without following links" in result.stdout
    assert not (tmp_path / "artifacts/evidence").exists()


def test_output_parent_inode_anchor_rejects_parent_swap(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts/evidence"
    anchor = _prepare_output_parent(tmp_path, destination)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts").rename(tmp_path / "moved-artifacts")
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(EvaluationTransactionError, match="parent changed"):
            _revalidate_output_parent(anchor, destination, published=False)
    finally:
        anchor.close()


def test_failed_verdict_still_discloses_its_signed_receipt(tmp_path: Path) -> None:
    material = _materialize_request(tmp_path)
    evidence_key, evidence_fingerprint = _key(tmp_path / "evidence.pem")
    verifier_key, _verifier_fingerprint = _key(tmp_path / "verifier.pem")
    runner = CliRunner()
    evaluated = runner.invoke(
        app,
        ["evaluate", str(material["request"]), "--signing-key", str(evidence_key)],
    )
    assert evaluated.exit_code == 0, evaluated.stdout
    policy_path = material["policy"]
    assert isinstance(policy_path, Path)
    policy_path.write_bytes(
        canonical_json_bytes(
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0.0}}}}
        )
    )
    runtimes = material["runtime_digests"]
    assert isinstance(runtimes, dict)
    receipt = tmp_path / "failed.verification.receipt.json"
    evidence = tmp_path / "artifacts/evidence"
    input_anchors = _input_anchors(evidence)

    result = runner.invoke(
        app,
        [
            "verify",
            str(tmp_path / "artifacts/evidence"),
            "--policy",
            str(policy_path),
            "--expected-baseline-artifact",
            input_anchors["baseline"],
            "--expected-subject-artifact",
            input_anchors["subject"],
            "--expected-schedule",
            input_anchors["dataset"],
            "--expected-baseline-runtime",
            str(runtimes["baseline"]),
            "--expected-subject-runtime",
            str(runtimes["subject"]),
            "--expected-signer",
            evidence_fingerprint,
            "--receipt",
            str(receipt),
            "--verifier-signing-key",
            str(verifier_key),
            "--verifier-identity",
            "invarlock-verifier/fixture",
        ],
    )

    assert result.exit_code != 0
    assert receipt.is_file()
    assert "Receipt" in result.stdout
    assert receipt.name in result.stdout
