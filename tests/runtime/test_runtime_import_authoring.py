from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeScoringRecord,
    ScoringObservation,
    build_runtime_behavioral_schedule_from_material,
)
from invarlock.runtime_import_authoring import (
    MAX_EXTERNAL_RECORDS,
    RuntimeImportAuthoringError,
    build_runtime_import_observation,
    build_runtime_import_receipt,
    load_external_scoring_records_jsonl,
    load_runtime_import_side,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)

_POLICY_DIGEST = "sha256:" + "f" * 64
_GENERATED_AT = "2026-07-16T12:00:00+00:00"


def _schedule(expected_outputs: tuple[str, str] = ("A", "B")):
    return build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "external-record-fixture",
            "config_name": None,
            "revision": "a" * 40,
            "split": "validation",
        },
        records=[
            {
                "record_id": "one",
                "input_text": "Return A",
                "expected_output": expected_outputs[0],
            },
            {
                "record_id": "two",
                "input_text": "Return B",
                "expected_output": expected_outputs[1],
            },
        ],
    )


def _records(outputs: tuple[str, str]):
    schedule = _schedule()
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


def _record_payload(record: RuntimeScoringRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "input_sha256": record.input_sha256,
        "status": record.status,
        "output_text": record.output_text,
        "output_sha256": record.output_sha256,
        "logprob_sum": record.logprob_sum,
        "token_count": record.token_count,
        "utf8_byte_count": record.utf8_byte_count,
        "error_code": record.error_code,
    }


def _write_record_objects(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _reload_side(side, *, schedule=None):
    return load_runtime_import_side(
        side.directory,
        role=side.role,
        schedule=schedule or _schedule(),
        policy_digest=_POLICY_DIGEST,
        expected_runtime_image_digest=side.runtime_image_digest,
    )


def _artifact(name: str, marker: str) -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name=name,
        sha256=marker * 64,
        byte_length=100,
        gguf_metadata_sha256="2" * 64,
        tensor_inventory_sha256="3" * 64,
        tokenizer_metadata_sha256="4" * 64,
    )


def _plugin(name: str = "neutral_records") -> RuntimeProviderPluginIdentity:
    return RuntimeProviderPluginIdentity(
        name=name,
        distribution="pinned-evaluator-adapter",
        distribution_version="1.0.0",
    )


def _backend() -> RuntimeBackendIdentity:
    return RuntimeBackendIdentity(
        name="pinned-evaluator",
        version="1.0.0",
        source_sha256="5" * 64,
        binary_sha256=None,
        build_sha256=None,
    )


def _capabilities(
    *,
    provider_name: str = "neutral_records",
    artifact_formats: tuple[str, ...] = ("gguf",),
) -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name=provider_name,
        artifact_formats=artifact_formats,  # type: ignore[arg-type]
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("container",),
        required_extra=None,
        required_image=None,
    )


def _execution_settings(*, allow_network: bool = False) -> RuntimeExecutionSettings:
    return RuntimeExecutionSettings(
        seed=0,
        context_length=128,
        batch_size=1,
        max_output_tokens=16,
        timeout_seconds=30,
        allow_network=allow_network,
    )


def _observation(
    artifact: GGUFArtifactIdentity, *, provider_name: str = "neutral_records"
) -> ScoringObservation:
    return build_runtime_import_observation(
        provider_name=provider_name,
        artifact_identity=artifact,
        schedule=_schedule(),
        records=_records(("A", "B")),
    )


def _write_side(
    path: Path,
    *,
    role: str,
    artifact: GGUFArtifactIdentity,
    outputs: tuple[str, str],
    image_marker: str,
):
    return write_runtime_import_side(
        path,
        role=role,  # type: ignore[arg-type]
        schedule=_schedule(),
        policy_digest=_POLICY_DIGEST,
        artifact_identity=artifact,
        records=_records(outputs),
        plugin=_plugin(),
        backend=_backend(),
        capabilities=_capabilities(),
        execution_settings=_execution_settings(),
        device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
        runtime_image_ref="registry.example/pinned-evaluator",
        runtime_image_digest="sha256:" + image_marker * 64,
        generated_at_utc=_GENERATED_AT,
    )


def test_jsonl_adapter_loads_only_schedule_bound_per_record_facts(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": record.record_id,
                    "input_sha256": record.input_sha256,
                    "status": record.status,
                    "output_text": record.output_text,
                    "output_sha256": record.output_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in _records(("A", "B"))
        ),
        encoding="utf-8",
    )

    loaded = load_external_scoring_records_jsonl(
        records_path,
        schedule=_schedule(),
    )

    assert loaded == _records(("A", "B"))


def test_jsonl_adapter_rejects_aggregate_or_unknown_fields(tmp_path: Path) -> None:
    scheduled = _schedule().records[0]
    path = tmp_path / "summary.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_id": scheduled.record_id,
                "input_sha256": scheduled.input_sha256,
                "status": "ok",
                "output_text": "A",
                "output_sha256": hashlib.sha256(b"A").hexdigest(),
                "accuracy": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="aggregate summaries are not accepted",
    ):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "non-empty newline-terminated JSONL"),
        (b"{}", "non-empty newline-terminated JSONL"),
        (b"{\n", "external record line 1"),
        (b"[]\n", "must be a JSON object"),
    ],
)
def test_jsonl_adapter_rejects_empty_unterminated_malformed_or_non_object_input(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)

    with pytest.raises(RuntimeImportAuthoringError, match=message):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_missing_required_record_identity(tmp_path: Path) -> None:
    path = tmp_path / "missing-identity.jsonl"
    value = _record_payload(_records(("A", "B"))[0])
    del value["input_sha256"]
    _write_record_objects(path, [value])

    with pytest.raises(RuntimeImportAuthoringError, match="missing 'input_sha256'"):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_invalid_typed_record_facts(tmp_path: Path) -> None:
    path = tmp_path / "invalid-record.jsonl"
    value = _record_payload(_records(("A", "B"))[0])
    value["output_text"] = True
    _write_record_objects(path, [value])

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="external record line 1 is invalid: output_text must be a string or null",
    ):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_blank_records(tmp_path: Path) -> None:
    path = tmp_path / "blank-record.jsonl"
    first, second = (_record_payload(record) for record in _records(("A", "B")))
    path.write_text(
        json.dumps(first, sort_keys=True, separators=(",", ":"))
        + "\n\n"
        + json.dumps(second, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeImportAuthoringError, match="line 2 must not be blank"):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_reordered_schedule_records(tmp_path: Path) -> None:
    path = tmp_path / "reordered.jsonl"
    records = _records(("A", "B"))
    _write_record_objects(
        path, [_record_payload(record) for record in reversed(records)]
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="order and input identities must exactly match the schedule",
    ):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_backend_error_records(tmp_path: Path) -> None:
    path = tmp_path / "backend-error.jsonl"
    first, second = _records(("A", "B"))
    failed = RuntimeScoringRecord(
        record_id=first.record_id,
        input_sha256=first.input_sha256,
        status="error",
        error_code="backend_failure",
    )
    _write_record_objects(path, [_record_payload(failed), _record_payload(second)])

    with pytest.raises(RuntimeImportAuthoringError, match="'one' is not successful"):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_output_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "wrong-output-digest.jsonl"
    first, second = _records(("A", "B"))
    forged = replace(first, output_sha256="0" * 64)
    _write_record_objects(path, [_record_payload(forged), _record_payload(second)])

    with pytest.raises(RuntimeImportAuthoringError, match="output digest is invalid"):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_jsonl_adapter_rejects_missing_or_non_regular_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeImportAuthoringError, match="external scoring records"):
        load_external_scoring_records_jsonl(
            tmp_path / "missing.jsonl", schedule=_schedule()
        )

    directory = tmp_path / "records.jsonl"
    directory.mkdir()
    with pytest.raises(RuntimeImportAuthoringError, match="regular file"):
        load_external_scoring_records_jsonl(directory, schedule=_schedule())


def test_jsonl_adapter_rejects_record_count_above_closed_limit(tmp_path: Path) -> None:
    path = tmp_path / "too-many.jsonl"
    path.write_bytes(b"{}\n" * (MAX_EXTERNAL_RECORDS + 1))

    with pytest.raises(
        RuntimeImportAuthoringError,
        match=f"exceed the {MAX_EXTERNAL_RECORDS}-record limit",
    ):
        load_external_scoring_records_jsonl(path, schedule=_schedule())


def test_import_observation_rejects_non_schedule_and_empty_record_sets() -> None:
    artifact = _artifact("baseline.gguf", "1")

    with pytest.raises(TypeError, match="schedule must be RuntimeBehavioralSchedule"):
        build_runtime_import_observation(
            provider_name="neutral_records",
            artifact_identity=artifact,
            schedule=object(),  # type: ignore[arg-type]
            records=_records(("A", "B")),
        )

    with pytest.raises(RuntimeImportAuthoringError, match="non-empty tuple"):
        build_runtime_import_observation(
            provider_name="neutral_records",
            artifact_identity=artifact,
            schedule=_schedule(),
            records=(),
        )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match=f"exceed the {MAX_EXTERNAL_RECORDS}-record limit",
    ):
        build_runtime_import_observation(
            provider_name="neutral_records",
            artifact_identity=artifact,
            schedule=_schedule(),
            records=(_records(("A", "B"))[0],) * (MAX_EXTERNAL_RECORDS + 1),
        )


def test_import_receipt_rejects_provider_identity_mismatch() -> None:
    artifact = _artifact("baseline.gguf", "1")

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="plugin, capabilities, and observation provider names must agree",
    ):
        build_runtime_import_receipt(
            plugin=_plugin("different_provider"),
            backend=_backend(),
            capabilities=_capabilities(),
            artifact_identity=artifact,
            execution_settings=_execution_settings(),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
            runtime_image_digest="sha256:" + "a" * 64,
            observation=_observation(artifact),
        )


def test_import_receipt_rejects_undeclared_artifact_format() -> None:
    artifact = _artifact("baseline.gguf", "1")

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="artifact format is not declared by provider capabilities",
    ):
        build_runtime_import_receipt(
            plugin=_plugin(),
            backend=_backend(),
            capabilities=_capabilities(artifact_formats=("hf_snapshot",)),
            artifact_identity=artifact,
            execution_settings=_execution_settings(),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
            runtime_image_digest="sha256:" + "a" * 64,
            observation=_observation(artifact),
        )


def test_import_receipt_rejects_network_enabled_execution() -> None:
    artifact = _artifact("baseline.gguf", "1")

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="requires offline execution settings",
    ):
        build_runtime_import_receipt(
            plugin=_plugin(),
            backend=_backend(),
            capabilities=_capabilities(),
            artifact_identity=artifact,
            execution_settings=_execution_settings(allow_network=True),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
            runtime_image_digest="sha256:" + "a" * 64,
            observation=_observation(artifact),
        )


def test_import_receipt_rejects_artifact_identity_substitution() -> None:
    supplied_artifact = _artifact("baseline.gguf", "1")
    observed_artifact = _artifact("other.gguf", "6")

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="observation does not bind the supplied artifact identity",
    ):
        build_runtime_import_receipt(
            plugin=_plugin(),
            backend=_backend(),
            capabilities=_capabilities(),
            artifact_identity=supplied_artifact,
            execution_settings=_execution_settings(),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
            runtime_image_digest="sha256:" + "a" * 64,
            observation=_observation(observed_artifact),
        )


def test_import_receipt_wraps_invalid_runtime_image_identity() -> None:
    artifact = _artifact("baseline.gguf", "1")

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="runtime import receipt is invalid: outer_image_digest",
    ):
        build_runtime_import_receipt(
            plugin=_plugin(),
            backend=_backend(),
            capabilities=_capabilities(),
            artifact_identity=artifact,
            execution_settings=_execution_settings(),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu"),
            runtime_image_digest="mutable-image-tag",
            observation=_observation(artifact),
        )


def test_complete_import_side_is_deterministic_and_strictly_reloadable(
    tmp_path: Path,
) -> None:
    artifact = _artifact("baseline.gguf", "1")
    first = _write_side(
        tmp_path / "first",
        role="baseline",
        artifact=artifact,
        outputs=("A", "B"),
        image_marker="a",
    )
    second = _write_side(
        tmp_path / "second",
        role="baseline",
        artifact=artifact,
        outputs=("A", "B"),
        image_marker="a",
    )

    filenames = (
        "model-artifact.identity.json",
        "runtime-scoring.observation.json",
        "runtime-provider.receipt.json",
        "report.json",
        "run.yaml",
        "runtime.manifest.json",
    )
    assert all(
        (first.directory / filename).read_bytes()
        == (second.directory / filename).read_bytes()
        for filename in filenames
    )
    assert (
        json.loads(first.manifest_path.read_bytes())["generated_at_utc"]
        == _GENERATED_AT
    )
    assert (
        load_runtime_import_side(
            first.directory,
            role="baseline",
            schedule=_schedule(),
            policy_digest=_POLICY_DIGEST,
            expected_runtime_image_digest="sha256:" + "a" * 64,
        )
        == first
    )


def test_paired_records_are_rederived_from_verified_side_evidence(
    tmp_path: Path,
) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "wrong"),
        image_marker="b",
    )

    paired = write_runtime_import_paired_records(
        tmp_path / "paired-records.json",
        schedule=_schedule(),
        metric="exact_match",
        baseline=baseline,
        subject=subject,
    )

    assert paired.payload["metric"] == "exact_match"
    records = paired.payload["records"]
    assert isinstance(records, list)
    assert [record["baseline"]["score"] for record in records] == [1.0, 1.0]
    assert [record["subject"]["score"] for record in records] == [1.0, 0.0]
    assert hashlib.sha256(paired.path.read_bytes()).hexdigest() == paired.sha256


def test_paired_records_reject_metric_outside_receipt_capabilities(
    tmp_path: Path,
) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="does not declare metric 'normalized_nll_per_utf8_byte'",
    ):
        write_runtime_import_paired_records(
            tmp_path / "paired-records.json",
            schedule=_schedule(),
            metric="normalized_nll_per_utf8_byte",
            baseline=baseline,
            subject=subject,
        )


def test_paired_records_publish_only_after_complete_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )
    destination = tmp_path / "paired-records.json"

    def interrupt_before_publish(*_args, **_kwargs) -> None:
        assert not destination.exists()
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(
        "invarlock.runtime_import_authoring.os.link", interrupt_before_publish
    )

    with pytest.raises(RuntimeImportAuthoringError, match="new and writable"):
        write_runtime_import_paired_records(
            destination,
            schedule=_schedule(),
            metric="exact_match",
            baseline=baseline,
            subject=subject,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".paired-records.json.staging.*"))


def test_reload_rejects_tampered_report_binding(tmp_path: Path) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    report = json.loads(side.report_path.read_bytes())
    report["record_count"] = 1
    side.report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="report does not bind its observation",
    ):
        load_runtime_import_side(
            side.directory,
            role="baseline",
            schedule=_schedule(),
            policy_digest=_POLICY_DIGEST,
            expected_runtime_image_digest="sha256:" + "a" * 64,
        )


@pytest.mark.parametrize(
    ("role", "policy_digest", "message"),
    [
        ("control", _POLICY_DIGEST, "role must be baseline or subject"),
        ("baseline", "not-a-digest", "policy_digest must be a sha256 digest"),
        (
            "baseline",
            "sha256:" + "F" * 64,
            "policy_digest must be a sha256 digest",
        ),
    ],
)
def test_reload_rejects_invalid_caller_side_bindings(
    tmp_path: Path,
    role: str,
    policy_digest: str,
    message: str,
) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )

    with pytest.raises(RuntimeImportAuthoringError, match=message):
        load_runtime_import_side(
            side.directory,
            role=role,  # type: ignore[arg-type]
            schedule=_schedule(),
            policy_digest=policy_digest,
            expected_runtime_image_digest="sha256:" + "a" * 64,
        )


@pytest.mark.parametrize(
    "destination_kind", ["unnamed", "missing_parent", "file_parent"]
)
def test_side_writer_rejects_unsafe_output_locations(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    if destination_kind == "unnamed":
        destination = Path()
        message = "output must name a directory"
    elif destination_kind == "missing_parent":
        destination = tmp_path / "missing" / "side"
        message = "output parent must be an existing directory"
    else:
        parent = tmp_path / "not-a-directory"
        parent.write_text("occupied\n", encoding="utf-8")
        destination = parent / "side"
        message = "output parent must be an existing directory"

    with pytest.raises(RuntimeImportAuthoringError, match=message):
        _write_side(
            destination,
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )


def test_side_writer_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "baseline"
    sentinel = destination / "owner.txt"
    destination.mkdir()
    sentinel.write_text("original owner\n", encoding="utf-8")

    with pytest.raises(RuntimeImportAuthoringError, match="output already exists"):
        _write_side(
            destination,
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )

    assert sentinel.read_text(encoding="utf-8") == "original owner\n"


def test_side_writer_removes_staging_after_atomic_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "baseline"

    def fail_publication(_staging: Path, _output: Path) -> None:
        raise OSError("simulated no-replace publication failure")

    monkeypatch.setattr(
        "invarlock.runtime_import_authoring.publish_directory_no_replace",
        fail_publication,
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="simulated no-replace publication failure",
    ):
        _write_side(
            destination,
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".baseline.staging.*"))


def test_side_writer_does_not_publish_manifest_rejected_by_independent_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "baseline"
    monkeypatch.setattr(
        "invarlock.runtime_import_authoring.verify_runtime_manifest_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(errors=("network enabled",)),
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="runtime import manifest is invalid: network enabled",
    ):
        _write_side(
            destination,
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".baseline.staging.*"))


@pytest.mark.parametrize("root_kind", ["missing", "file", "symlink"])
def test_reload_rejects_non_directory_or_indirect_side_roots(
    tmp_path: Path,
    root_kind: str,
) -> None:
    if root_kind == "missing":
        root = tmp_path / "missing"
    elif root_kind == "file":
        root = tmp_path / "side-file"
        root.write_text("not a side\n", encoding="utf-8")
    else:
        side = _write_side(
            tmp_path / "baseline",
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )
        root = tmp_path / "side-link"
        root.symlink_to(side.directory, target_is_directory=True)

    with pytest.raises(RuntimeImportAuthoringError, match="must be a real directory"):
        load_runtime_import_side(
            root,
            role="baseline",
            schedule=_schedule(),
            policy_digest=_POLICY_DIGEST,
            expected_runtime_image_digest="sha256:" + "a" * 64,
        )


def test_reload_rejects_missing_provider_receipt(tmp_path: Path) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    side.provider_evidence.paths.receipt.unlink()

    with pytest.raises(RuntimeImportAuthoringError, match=r"runtime-provider\.receipt"):
        _reload_side(side)


def test_reload_rejects_observation_from_another_schedule(tmp_path: Path) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="observation does not bind the supplied schedule",
    ):
        _reload_side(side, schedule=_schedule(("different", "B")))


@pytest.mark.parametrize("payload_kind", ["malformed", "array", "noncanonical"])
def test_reload_rejects_malformed_or_noncanonical_report(
    tmp_path: Path,
    payload_kind: str,
) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    if payload_kind == "malformed":
        side.report_path.write_bytes(b"{\n")
        message = "runtime import report"
    elif payload_kind == "array":
        side.report_path.write_bytes(b"[]\n")
        message = "must be a JSON object"
    else:
        report = json.loads(side.report_path.read_bytes())
        side.report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        message = "must use canonical JSON"

    with pytest.raises(RuntimeImportAuthoringError, match=message):
        _reload_side(side)


def test_reload_rejects_config_for_another_role_or_policy(tmp_path: Path) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    config = json.loads(side.config_path.read_bytes())
    config["role"] = "subject"
    side.config_path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="config does not bind role, schedule, artifact, and policy",
    ):
        _reload_side(side)


@pytest.mark.parametrize("payload", [b"{\n", b"[]\n"])
def test_reload_rejects_malformed_or_non_object_manifest(
    tmp_path: Path,
    payload: bytes,
) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    side.manifest_path.write_bytes(payload)

    with pytest.raises(RuntimeImportAuthoringError, match="runtime import manifest"):
        _reload_side(side)


def test_reload_rejects_manifest_that_enables_network(tmp_path: Path) -> None:
    side = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    manifest = json.loads(side.manifest_path.read_bytes())
    manifest["outer_container"]["allow_network"] = True
    side.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeImportAuthoringError, match="manifest is invalid"):
        _reload_side(side)


def test_paired_records_reject_swapped_side_roles(tmp_path: Path) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="require baseline and subject side roles",
    ):
        write_runtime_import_paired_records(
            tmp_path / "paired.json",
            schedule=_schedule(),
            metric="exact_match",
            baseline=replace(baseline, role="subject"),
            subject=replace(subject, role="baseline"),
        )


def test_paired_records_reject_sides_from_another_schedule(tmp_path: Path) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="paired records are invalid",
    ):
        write_runtime_import_paired_records(
            tmp_path / "paired.json",
            schedule=_schedule(("different", "B")),
            metric="exact_match",
            baseline=baseline,
            subject=subject,
        )


@pytest.mark.parametrize(
    "destination_kind", ["unnamed", "missing_parent", "file_parent"]
)
def test_paired_records_reject_unsafe_destinations(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )
    if destination_kind == "unnamed":
        destination = Path()
        message = "destination must name a file"
    elif destination_kind == "missing_parent":
        destination = tmp_path / "missing" / "paired.json"
        message = "destination parent must exist"
    else:
        parent = tmp_path / "not-a-directory"
        parent.write_text("occupied\n", encoding="utf-8")
        destination = parent / "paired.json"
        message = "destination parent must exist"

    with pytest.raises(RuntimeImportAuthoringError, match=message):
        write_runtime_import_paired_records(
            destination,
            schedule=_schedule(),
            metric="exact_match",
            baseline=baseline,
            subject=subject,
        )


def test_paired_records_no_clobber_preserves_existing_evidence(tmp_path: Path) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )
    destination = tmp_path / "paired.json"
    destination.write_text('{"owner":"existing"}\n', encoding="utf-8")

    with pytest.raises(RuntimeImportAuthoringError, match="new and writable"):
        write_runtime_import_paired_records(
            destination,
            schedule=_schedule(),
            metric="exact_match",
            baseline=baseline,
            subject=subject,
        )

    assert destination.read_text(encoding="utf-8") == '{"owner":"existing"}\n'


def test_paired_records_detect_post_link_publication_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _write_side(
        tmp_path / "baseline",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    subject = _write_side(
        tmp_path / "subject",
        role="subject",
        artifact=_artifact("subject.gguf", "6"),
        outputs=("A", "B"),
        image_marker="b",
    )
    destination = tmp_path / "paired.json"
    original_link = os.link

    def link_then_tamper(*args, **kwargs) -> None:
        original_link(*args, **kwargs)
        destination.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("invarlock.runtime_import_authoring.os.link", link_then_tamper)

    with pytest.raises(
        RuntimeImportAuthoringError,
        match="paired records changed during publication",
    ):
        write_runtime_import_paired_records(
            destination,
            schedule=_schedule(),
            metric="exact_match",
            baseline=baseline,
            subject=subject,
        )

    assert not destination.exists()
