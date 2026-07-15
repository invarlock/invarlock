from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.runtime_behavior.side as runtime_behavior_side
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeScoringRecord,
    ScoringObservation,
    artifact_identity_sha256,
)
from invarlock.policy_pack import build_behavioral_policy_pack
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_behavior import (
    RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME,
    RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
    RuntimeBehaviorError,
    run_side,
)
from invarlock.runtime_providers import hf_transformers
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider
from invarlock.runtime_security_helpers import RUNTIME_MANIFEST_FILENAME
from invarlock.runtime_verify import verify_runtime_manifest
from tests.runtime._runtime_behavior_support import (
    _IMAGE_DIGEST,
    _IMAGE_REF,
    _baseline_identity,
    _binding,
    _context,
    _execution_settings,
    _FakeProvider,
    _record_payload,
    _rewrite_policy,
    _run,
    _sha256,
    _strict_container_environment,  # noqa: F401
    _subject_identity,
    _write_inputs,
)


def test_run_side_publishes_only_a_strict_portable_bundle(tmp_path: Path) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    bundle = _run(
        tmp_path,
        provider=provider,
        role="baseline",
        directory_name="baseline",
        schedule_path=schedule_path,
        policy_path=policy_path,
    )

    assert provider.session is not None and provider.session.closed
    assert bundle.role == "baseline"
    assert bundle.metric_result.value == 1.0
    assert {path.name for path in bundle.directory.iterdir()} == {
        RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME,
        RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
        RUNTIME_MANIFEST_FILENAME,
        "model-artifact.identity.json",
        "runtime-provider.receipt.json",
        "runtime-scoring.observation.json",
    }
    result = verify_runtime_manifest(
        bundle.report_path,
        bundle.manifest_path,
        expected_image_digest=_IMAGE_DIGEST,
        require_strict_runtime=True,
    )
    assert result.ok, result.errors
    assert json.loads(bundle.config_path.read_text(encoding="utf-8"))["role"] == (
        "baseline"
    )
    assert json.loads(bundle.report_path.read_text(encoding="utf-8"))["role"] == (
        "baseline"
    )
    portable_bytes = bundle.report_path.read_bytes() + bundle.manifest_path.read_bytes()
    assert str(tmp_path).encode() not in portable_bytes


def test_first_party_hf_provider_receipt_runs_through_side_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    identity = _baseline_identity()
    monkeypatch.setattr(
        hf_transformers, "strict_container_boundary_present", lambda: True
    )
    monkeypatch.setattr(hf_transformers, "network_allowed", lambda: False)
    monkeypatch.setattr(hf_transformers, "remote_code_allowed", lambda: False)
    monkeypatch.setattr(hf_transformers, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        hf_transformers, "resolve_runtime_image_digest", lambda: _IMAGE_DIGEST
    )
    monkeypatch.setattr(hf_transformers, "resolve_runtime_image", lambda: _IMAGE_REF)
    monkeypatch.setattr(
        hf_transformers,
        "_installed_backend_identity",
        lambda _model: RuntimeBackendIdentity(
            name="transformers+torch",
            version="transformers=5.12.0;torch=2.11.0",
            source_sha256="7" * 64,
            binary_sha256="8" * 64,
            build_sha256="9" * 64,
        ),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_observed_device_facts",
        lambda _model, *, expected_device_kind: RuntimeDeviceFacts(
            device_kind=expected_device_kind,
            device_name="authenticated test cpu",
        ),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        lambda *, spec, identity, context: context.scorer,
    )
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="portable-model",
        adapter_name="hf_causal",
        settings={
            "batch_size": 1,
            "context_length": 128,
            "immutable_revision": "1" * 40,
            "max_output_tokens": 8,
            "offline": True,
            "seed": 7,
            "timeout_seconds": 30,
            "tokenizer_metadata_sha256": "2" * 64,
        },
    )

    def scorer(
        batch: EvaluationBatch, settings: RuntimeExecutionSettings
    ) -> ScoringObservation:
        assert settings == _execution_settings()
        records = tuple(
            RuntimeScoringRecord(
                record_id=record.record_id,
                input_sha256=record.input_sha256,
                status="ok",
                output_text=output,
                output_sha256=_sha256(output.encode("utf-8")),
            )
            for record, output in zip(batch.records, ("A", "B"), strict=True)
        )
        return ScoringObservation(
            provider_name="hf_transformers",
            artifact_identity_sha256=artifact_identity_sha256(identity),
            schedule_sha256=batch.schedule_sha256,
            records=records,
            aggregate_source_sha256=runtime_scoring_records_sha256(
                [_record_payload(record) for record in records]
            ),
        )

    bundle = run_side(
        role="baseline",
        provider=HFTransformersProvider(),
        spec=spec,
        context=RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            artifact_identity_sha256=artifact_identity_sha256(identity),
            model_adapter=object(),
            native_model=object(),
            scorer=scorer,
        ),
        schedule_path=schedule_path,
        policy_pack_path=policy_path,
        output_directory=tmp_path / "first-party-hf-side",
    )

    receipt = json.loads(
        bundle.directory.joinpath("runtime-provider.receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["plugin"]["name"] == "hf_transformers"
    assert receipt["backend"]["name"] == "transformers+torch"
    assert receipt["execution_settings"]["allow_network"] is False


def test_run_side_rejects_nonportable_cross_runtime_batching(tmp_path: Path) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match="batch_size=1"):
        run_side(
            role="baseline",
            provider=provider,
            spec=ModelRuntimeSpec(
                provider_name=provider.name,
                model_id="portable-model",
                settings={"batch_size": 2},
            ),
            context=_context(provider.identity),
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            output_directory=tmp_path / "nonportable-batch",
        )

    assert provider.session is None


@pytest.mark.parametrize("failure", ["score", "order"])
def test_run_side_closes_and_does_not_publish_failures(
    tmp_path: Path, failure: str
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
        score_error=failure == "score",
        reverse_records=failure == "order",
    )

    with pytest.raises((RuntimeError, ValueError)):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="failed-side",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is not None and provider.session.closed
    assert not (tmp_path / "failed-side").exists()


def test_run_side_rejects_unauthorized_provider_before_open(tmp_path: Path) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = build_behavioral_policy_pack(
        tier="conservative",
        schedule_sha256=policy["behavioral_claim"]["schedule_sha256"],
        baseline=_binding(provider_name="llama_cpp", identity=_subject_identity()),
        subject=policy["behavioral_claim"]["subject"],
        metric_kind="exact_match",
        minimum_subject_score=0.5,
        maximum_regression=0.5,
        dataset_identity=policy["compatibility"]["dataset_identity"],
    )
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match="baseline provider_name"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="unauthorized",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is None
    assert not (tmp_path / "unauthorized").exists()


def test_run_side_bounds_policy_reads_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        "invarlock.runtime_behavior.io.MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES",
        32,
    )
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match="size limit"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="oversized-policy",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is None


def test_run_side_rejects_host_execution_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION")
    monkeypatch.setattr(
        runtime_behavior_side,
        "strict_container_boundary_present",
        lambda: False,
    )
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match="actual container execution"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="host-side",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is None


def test_run_side_rejects_unpinned_image_reference_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "registry.example/runtime:mutable")
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match="must embed the exact"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="mutable-image",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is None


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schedule", "f" * 64, "schedule does not match"),
        ("outer_image_digest", "sha256:" + "f" * 64, "outer_image_digest"),
        ("execution_settings_sha256", "f" * 64, "execution_settings_sha256"),
    ],
)
def test_run_side_enforces_exact_directed_policy_binding(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    original = json.loads(policy_path.read_text(encoding="utf-8"))
    baseline = dict(original["behavioral_claim"]["baseline"])
    if field == "schedule":
        _rewrite_policy(policy_path, schedule_sha256=replacement)
    else:
        baseline[field] = replacement
        _rewrite_policy(policy_path, baseline=baseline)
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match=message):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name=f"wrong-{field}",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    if field == "execution_settings_sha256":
        assert provider.session is not None and provider.session.closed
    else:
        assert provider.session is None


def test_run_side_rejects_online_provider_receipt(tmp_path: Path) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
        receipt_allow_network=True,
    )

    with pytest.raises(RuntimeBehaviorError, match="receipt must bind offline"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="online-receipt",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is not None and provider.session.closed


def test_run_side_rejects_receipt_image_mismatch(tmp_path: Path) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
        receipt_image_digest="sha256:" + "f" * 64,
    )

    with pytest.raises(RuntimeBehaviorError, match="receipt image"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="wrong-receipt-image",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert provider.session is not None and provider.session.closed


def test_run_side_destination_race_cannot_clobber_existing_directory(
    tmp_path: Path,
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    destination = tmp_path / "occupied"
    destination.mkdir()
    marker = destination / "owner.txt"
    marker.write_text("existing", encoding="utf-8")
    provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )

    with pytest.raises(RuntimeBehaviorError, match="without clobber"):
        _run(
            tmp_path,
            provider=provider,
            role="baseline",
            directory_name="occupied",
            schedule_path=schedule_path,
            policy_path=policy_path,
        )

    assert marker.read_text(encoding="utf-8") == "existing"
    assert provider.session is not None and provider.session.closed
