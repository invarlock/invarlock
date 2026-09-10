"""Public dispatch and native failure contracts around captured transactions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from invarlock import engine, evaluation_transaction
from invarlock.core import evaluation_request
from invarlock.evaluation_record_contracts.contracts import MAX_RECORDS
from tests.cli.test_evaluation_preflight import _materialize_run_request
from tests.cli.test_import_journey import (
    _materialize_request,
    _settings,
    _side_evidence,
    _text_scorer_registry_and_binding,
)
from tests.core.test_captured_sdk_omissions import _bytes, _inputs, _key
from tests.core.test_evaluation_request_contract import _valid_request


@pytest.mark.parametrize("count", [0, MAX_RECORDS + 1])
def test_facade_rejects_empty_and_oversized_declared_collections(count: int) -> None:
    records: list[dict[str, Any]] = [{}] * count
    with pytest.raises(engine.EvaluationRecordsError, match="run must contain"):
        engine.make_run(
            records,
            source={"name": "test", "version": "1"},
            run_id="baseline",
            artifact_digest="sha256:" + "a" * 64,
        )
    with pytest.raises(engine.EvaluationRecordsError, match="case set must contain"):
        engine.freeze_case_set(records)


def test_facade_validates_exact_planned_membership_without_mutation(tmp_path) -> None:
    _, baseline, _, _ = _inputs(tmp_path)
    cases = engine.freeze_case_set(
        [
            {k: row[k] for k in ("id", "input", "expected", "metadata")}
            for row in baseline["records"]
        ]
    )
    expected = engine.case_set_digest(cases)
    original = _bytes(baseline)
    engine.validate_run_case_set(MappingProxyType(baseline), expected)
    assert _bytes(baseline) == original
    missing = {**baseline, "records": baseline["records"][:-1]}
    with pytest.raises(engine.EvaluationRecordsError, match="case.set"):
        engine.validate_run_case_set(MappingProxyType(missing), expected)


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"\xff", "UTF-8"),
        (b"value: \tbad", "could not be scanned"),
        (b"[]", "YAML object"),
        (b'{"format_version":"future"}', "unsupported evaluation request"),
    ],
)
def test_public_request_readers_reject_unreadable_or_ambiguous_documents(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "request.yaml"
    path.write_bytes(payload)
    for reader in (
        engine.load_evaluation_request,
        evaluation_request.evaluation_request_mode,
    ):
        with pytest.raises(engine.EvaluationRequestError, match=message):
            reader(path)


@pytest.mark.parametrize("mode", ["captured", "run"])
@pytest.mark.parametrize("execution", [None, [], {}, {"mode": "future"}])
def test_discriminator_rejects_invalid_execution_before_discovery(
    tmp_path: Path, mode: str, execution: object
) -> None:
    path = tmp_path / "request.json"
    path.write_bytes(
        _bytes(
            {
                "format_version": "invarlock/evaluation-request-v2"
                if mode == "captured"
                else "invarlock/evaluation-request-v1",
                "execution": execution,
            }
        )
    )
    with pytest.raises(
        engine.EvaluationRequestError, match="execution mode is invalid"
    ):
        evaluation_request.evaluation_request_mode(path)


def test_discriminator_reports_unavailable_request(tmp_path: Path) -> None:
    with pytest.raises(engine.EvaluationRequestError, match="request is unavailable"):
        evaluation_request.evaluation_request_mode(tmp_path / "missing.yaml")


def test_loader_rejects_fifo_artifact_without_blocking(tmp_path: Path) -> None:
    path = _valid_request(tmp_path)
    fifo = tmp_path / "model.fifo"
    os.mkfifo(fifo)
    value = yaml.safe_load(path.read_text())
    value["comparison"]["baseline"]["artifact"]["path"] = fifo.name
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(
        engine.EvaluationRequestError, match="regular file or directory"
    ):
        engine.load_evaluation_request(path)


def test_loader_translates_missing_authorized_provider(tmp_path: Path) -> None:
    path = _valid_request(tmp_path)
    resolver = Mock(side_effect=LookupError("provider removed"))
    with pytest.raises(
        engine.EvaluationRequestError, match="not installed or authorized"
    ):
        engine.load_evaluation_request(path, provider_resolver=resolver)
    resolver.assert_called_once_with("hf_transformers")


def test_loader_reports_destination_inspection_failure(tmp_path, monkeypatch) -> None:
    path = _valid_request(tmp_path)
    (tmp_path / "artifacts").mkdir()
    original = os.stat

    def denied(path, *args, **kwargs):
        if path == "evidence" and kwargs.get("dir_fd") is not None:
            raise PermissionError("destination lookup denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", denied)
    with pytest.raises(engine.EvaluationRequestError, match="cannot be inspected"):
        engine.load_evaluation_request(path)
    assert list((tmp_path / "artifacts").iterdir()) == []


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize(
    "controls",
    [
        {"unsigned": True},
        {"max_bootstrap_draws": None},
        {"max_bootstrap_draws": True},
        {"max_bootstrap_draws": 0},
    ],
)
def test_native_typed_requests_reject_captured_controls_before_key_or_runtime_work(
    tmp_path, monkeypatch, preflight, controls
) -> None:
    request = engine.load_evaluation_request(_valid_request(tmp_path))
    assert isinstance(request, engine.EvaluationRequest)
    forbidden = Mock(side_effect=AssertionError("must reject before preparation"))
    monkeypatch.setattr(evaluation_transaction, "_prepare_evaluation_inputs", forbidden)
    operation = (
        engine.preflight_evaluation_request
        if preflight
        else engine.evaluate_request_file
    )
    error = (
        engine.EvaluationPreflightError
        if preflight
        else engine.EvaluationTransactionError
    )
    with pytest.raises(error, match="captured controls") as caught:
        operation(request, signing_key_path=None, **controls)
    assert isinstance(
        caught.value,
        engine.EvaluationPreflightError | engine.EvaluationTransactionError,
    )
    assert json.loads(caught.value.as_json())["format_version"] == (
        "invarlock/evaluation-preflight-v2"
        if preflight
        else "invarlock/evaluation-result-v1"
    )
    forbidden.assert_not_called()
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("preflight", [False, True])
def test_invalid_captured_shape_retains_captured_error_contract(
    tmp_path, preflight
) -> None:
    request, _, _, _ = _inputs(tmp_path)
    request["unexpected"] = True
    (tmp_path / "request.json").write_bytes(_bytes(request))
    operation = (
        engine.preflight_evaluation_request
        if preflight
        else engine.evaluate_request_file
    )
    error = (
        engine.EvaluationPreflightError
        if preflight
        else engine.EvaluationTransactionError
    )
    with pytest.raises(error, match="evaluation_request_v2") as caught:
        operation(tmp_path / "request.json", signing_key_path=None, unsigned=True)
    assert isinstance(
        caught.value,
        engine.EvaluationPreflightError | engine.EvaluationTransactionError,
    )
    payload = json.loads(caught.value.as_json())
    assert payload["kind"] == "captured"
    assert payload["ok"] is False
    if preflight:
        assert payload["requested_authentication"] == "unsigned_local"
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("initial", ["captured", "native"])
def test_transaction_rejects_request_mode_replacement_during_loading(
    tmp_path, monkeypatch, initial
) -> None:
    native = _valid_request(tmp_path)
    native_bytes = native.read_bytes()
    _inputs(tmp_path)
    captured_bytes = (tmp_path / "request.json").read_bytes()
    native.write_bytes(captured_bytes if initial == "captured" else native_bytes)
    _key(tmp_path / "key.pem")
    discriminator = evaluation_transaction.evaluation_request_mode

    def replace_after_discrimination(path):
        mode = discriminator(path)
        path.write_bytes(native_bytes if initial == "captured" else captured_bytes)
        return mode

    monkeypatch.setattr(
        evaluation_transaction, "evaluation_request_mode", replace_after_discrimination
    )
    with pytest.raises(
        engine.EvaluationTransactionError, match="mode changed|captured evaluation path"
    ):
        engine.evaluate_request_file(native, signing_key_path=tmp_path / "key.pem")
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize("failure", ["identity_json", "identity_mismatch", "settings"])
def test_import_preflight_rejects_unreproducible_provider_evidence(
    tmp_path, side, failure
) -> None:
    _materialize_request(tmp_path)
    _key(tmp_path / "key.pem")
    request_path = tmp_path / "request.yaml"
    value = yaml.safe_load(request_path.read_text())
    if failure == "identity_json":
        (tmp_path / f"imports/{side}/model-artifact.identity.json").write_bytes(b"{}")
        message = "provider evidence is invalid"
    elif failure == "identity_mismatch":
        value["comparison"][side]["artifact"]["model_id"] = "org/other-model"
        message = "artifact identity does not match"
    else:
        value["comparison"][side]["runtime"]["settings"]["batch_size"] = 2
        message = "batch_size"
    request_path.write_text(yaml.safe_dump(value))
    with pytest.raises(engine.EvaluationPreflightError, match=message):
        engine.preflight_evaluation_request(
            request_path, signing_key_path=tmp_path / "key.pem"
        )
    assert not (tmp_path / "artifacts").exists()


def test_import_preflight_wraps_provider_identity_failure(
    tmp_path, monkeypatch
) -> None:
    _materialize_request(tmp_path)
    _key(tmp_path / "key.pem")
    provider = "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.identify_artifact"
    monkeypatch.setattr(provider, Mock(side_effect=ValueError("identity unavailable")))
    with pytest.raises(
        engine.EvaluationPreflightError, match="cannot reproduce an artifact identity"
    ):
        engine.preflight_evaluation_request(
            tmp_path / "request.yaml", signing_key_path=tmp_path / "key.pem"
        )
    assert not (tmp_path / "artifacts").exists()


def test_run_preflight_wraps_resource_resolution_failure(tmp_path) -> None:
    path, key = _materialize_run_request(tmp_path)
    resolver = Mock()
    resolver.resolve.side_effect = OSError("resource mount disappeared")
    with pytest.raises(
        engine.EvaluationPreflightError,
        match="baseline caller-owned runtime resources are invalid",
    ):
        engine.preflight_evaluation_request(
            path,
            signing_key_path=key,
            runtime_image_digests=dict.fromkeys(
                ("baseline", "subject"), "sha256:" + "a" * 64
            ),
            resource_resolver=resolver,
        )
    resolver.resolve.assert_called_once()
    assert not (tmp_path / "artifacts").exists()


def test_import_records_are_rechecked_after_preflight(tmp_path, monkeypatch) -> None:
    _materialize_request(tmp_path)
    _key(tmp_path / "key.pem")
    original = evaluation_transaction.preflight_evaluation_request

    def replace_after_preflight(*args, **kwargs):
        result = original(*args, **kwargs)
        records = tmp_path / "imports/paired-records.json"
        records.write_bytes(records.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        evaluation_transaction, "preflight_evaluation_request", replace_after_preflight
    )
    with pytest.raises(
        engine.EvaluationTransactionError, match="must use canonical JSON"
    ):
        engine.evaluate_request_file(
            tmp_path / "request.yaml", signing_key_path=tmp_path / "key.pem"
        )
    assert not (tmp_path / "artifacts/evidence").exists()


@pytest.mark.parametrize(
    "failure", [None, "subject_preflight", "baseline_worker", "subject_worker"]
)
def test_native_execution_validates_both_receipts_and_worker_digest_claims(
    tmp_path, monkeypatch, failure
) -> None:
    from dataclasses import replace

    from invarlock.core.runtime_provider import (
        RuntimeScoringRecord,
        build_runtime_behavioral_schedule,
    )
    from invarlock.evaluation_run import EvaluationRunResult, load_runtime_side_evidence
    from invarlock.evaluation_runtime import CallerRuntimeResources
    from invarlock.evidence_pack_contract import sha256_digest

    path, key = _materialize_run_request(tmp_path)
    value = yaml.safe_load(path.read_text())
    for side in ("baseline", "subject"):
        settings = value["comparison"][side]["runtime"]["settings"]
        value["comparison"][side]["runtime"]["settings"] = {
            **_settings(),
            "checkpoint_tree_sha256": settings["checkpoint_tree_sha256"],
        }
    path.write_text(yaml.safe_dump(value))
    request = engine.load_evaluation_request(path)
    assert isinstance(request, engine.EvaluationRequest)
    runtime = "sha256:" + "a" * 64
    inspected = dict.fromkeys(("baseline", "subject"), runtime)
    executions = []
    (tmp_path / "workers").mkdir()

    class Executor:
        def resolve(self, **kwargs):
            return CallerRuntimeResources(container_image_digest=runtime).resolve(
                **kwargs
            )

        def execute(self, loaded, *, registry, schedule_bytes, policy_digest):
            assert loaded == request
            executions.append(loaded)
            schedule = build_runtime_behavioral_schedule(json.loads(schedule_bytes))
            sides = {}
            runtime_digests = dict(inspected)
            if failure == "subject_preflight":
                runtime_digests["subject"] = "sha256:" + "b" * 64
            for role in ("baseline", "subject"):
                side = getattr(loaded.comparison, role)
                records = []
                for row in schedule.records:
                    assert row.expected_output is not None
                    records.append(
                        RuntimeScoringRecord(
                            record_id=row.record_id,
                            input_sha256=row.input_sha256,
                            status="ok",
                            output_text=row.expected_output,
                            output_sha256=sha256_digest(
                                row.expected_output.encode()
                            ).removeprefix("sha256:"),
                        )
                    )
                authored = _side_evidence(
                    tmp_path / "workers" / role,
                    role=role,
                    model_id=side.artifact.model_id,
                    schedule=schedule,
                    records=tuple(records),
                    runtime_digest=runtime_digests[role],
                    policy_digest=policy_digest,
                    checkpoint_digest=side.runtime.settings["checkpoint_tree_sha256"],
                )
                sides[role] = load_runtime_side_evidence(authored.directory)
            result = EvaluationRunResult(
                baseline=sides["baseline"],
                subject=sides["subject"],
                baseline_runtime_digest=runtime_digests["baseline"],
                subject_runtime_digest=runtime_digests["subject"],
            )
            if failure == "baseline_worker":
                result = replace(result, baseline_runtime_digest="sha256:" + "c" * 64)
            elif failure == "subject_worker":
                result = replace(result, subject_runtime_digest="sha256:" + "c" * 64)
            return result

    executor = Executor()
    if failure is None:
        monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", runtime)

        def execute_direct(loaded, *, resource_resolver, **kwargs):
            assert isinstance(resource_resolver, CallerRuntimeResources)
            assert resource_resolver.container_image_digest == runtime
            return executor.execute(loaded, **kwargs)

        monkeypatch.setattr(
            evaluation_transaction, "execute_runtime_comparison", execute_direct
        )
        result = engine.evaluate_request_file(
            request, signing_key_path=key, runtime_image_digests=inspected
        )
        assert isinstance(result, engine.EvaluationTransactionResult)
        report = json.loads(
            (result.evidence_path / "reports/evaluation.report.json").read_bytes()
        )
        assert result.policy_verdict == report["verdict"] == "fail"
        assert (result.evidence_path / "manifest.json").is_file()
        before = (result.evidence_path / "manifest.json").read_bytes()
        with pytest.raises(engine.EvaluationTransactionError, match="already exists"):
            engine.evaluate_request_file(
                request, signing_key_path=key, runtime_image_digests=inspected
            )
        assert (result.evidence_path / "manifest.json").read_bytes() == before
    else:
        message = (
            "subject validated runtime digest does not match preflight"
            if failure == "subject_preflight"
            else f"{failure.removesuffix('_worker')} worker runtime digest does not match its validated receipt"
        )
        with pytest.raises(engine.EvaluationTransactionError, match=message):
            engine.evaluate_request_file(
                request,
                signing_key_path=key,
                runtime_image_digests=inspected,
                runtime_executor=executor,
            )
        assert not request.output.evidence.exists()
    assert executions == [request]


def test_import_preflight_wraps_scorer_inventory_failure(tmp_path, monkeypatch) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    _materialize_request(tmp_path, scorer_binding=binding, scorer_registry=registry)
    _key(tmp_path / "key.pem")
    failure = engine.ScorerExtensionError("installed scorer inventory changed")
    monkeypatch.setattr(registry, "list_scorers", Mock(side_effect=failure))
    with pytest.raises(
        engine.EvaluationPreflightError, match="scorer inventory changed"
    ) as caught:
        engine.preflight_evaluation_request(
            tmp_path / "request.yaml",
            signing_key_path=tmp_path / "key.pem",
            scorer_registry=registry,
        )
    assert caught.value.__cause__ is not None
    assert caught.value.__cause__.__cause__ is failure
    assert not (tmp_path / "artifacts").exists()


def test_publication_revalidates_path_after_descriptor_check(
    tmp_path, monkeypatch
) -> None:
    _materialize_request(tmp_path)
    _key(tmp_path / "key.pem")
    original = Path.lstat
    destination = tmp_path / "artifacts/evidence"

    def disappeared(path, *args, **kwargs):
        if path == destination and path.is_dir():
            raise FileNotFoundError("published path disappeared")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", disappeared)
    with pytest.raises(
        engine.EvaluationTransactionError, match="published evidence path changed"
    ):
        engine.evaluate_request_file(
            tmp_path / "request.yaml", signing_key_path=tmp_path / "key.pem"
        )


def test_loader_rejects_missing_top_level_captured_source(tmp_path) -> None:
    _inputs(tmp_path)
    (tmp_path / "baseline.json").unlink()
    with pytest.raises(
        engine.EvaluationRequestError, match="comparison.baseline.path.*root-confined"
    ):
        engine.load_evaluation_request(tmp_path / "request.json")
    assert not (tmp_path / "artifacts").exists()


def test_loader_rejects_scorer_configuration_digest_drift(tmp_path) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    _materialize_request(tmp_path, scorer_binding=binding, scorer_registry=registry)
    path = tmp_path / "request.yaml"
    value = yaml.safe_load(path.read_text())
    value["comparison"]["scorer_extension"]["configuration_sha256"] = "a" * 64
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(
        engine.EvaluationRequestError, match="configuration_sha256 does not match"
    ) as caught:
        engine.load_evaluation_request(path)
    assert isinstance(caught.value.__cause__, engine.ScorerExtensionError)
