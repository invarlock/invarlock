from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import invarlock.evaluation_oci as evaluation_oci
import invarlock.evaluation_transaction as evaluation_transaction
from invarlock.cli.app import app
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.scorer_extension import ScorerExtensionRegistry
from invarlock.evaluation_runtime import CallerRuntimeResources
from invarlock.evaluation_transaction import (
    EvaluationPreflightError,
    EvaluationTransactionError,
    preflight_evaluation_request,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from tests.cli.test_import_journey import (
    _CasefoldTextScorer,
    _key,
    _materialize_request,
    _text_scorer_registry_and_binding,
)


def _materialize_run_request(
    tmp_path: Path, *, baseline_digest: str | None = None
) -> tuple[Path, Path]:
    checkpoint_digests: dict[str, str] = {}
    for side_name in ("baseline", "subject"):
        checkpoint = tmp_path / "models" / side_name
        checkpoint.mkdir(parents=True)
        checkpoint.joinpath("config.json").write_text(
            json.dumps({"model_type": "test", "side": side_name}),
            encoding="utf-8",
        )
        checkpoint_digests[side_name] = checkpoint_tree_sha256(checkpoint).removeprefix(
            "sha256:"
        )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset = b'{"id":"one","prompt":"Return A","expected":"A"}\n'
    (inputs / "records.jsonl").write_bytes(dataset)
    (inputs / "policy.json").write_bytes(
        canonical_json_bytes(
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0.0}}}}
        )
    )

    def side(name: str) -> dict[str, object]:
        digest = (
            baseline_digest
            if name == "baseline" and baseline_digest is not None
            else checkpoint_digests[name]
        )
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
                    "checkpoint_tree_sha256": digest,
                    "context_length": 8,
                    "max_output_tokens": 1,
                    "offline": True,
                    "seed": 0,
                    "timeout_seconds": 30,
                    "tokenizer_metadata_sha256": "d" * 64,
                },
            },
        }

    request = tmp_path / "request.yaml"
    request.write_text(
        yaml.safe_dump(
            {
                "format_version": "invarlock/evaluation-request-v1",
                "comparison": {
                    "baseline": side("baseline"),
                    "subject": side("subject"),
                    "dataset": {
                        "path": "inputs/records.jsonl",
                        "sha256": hashlib.sha256(dataset).hexdigest(),
                        "format": "jsonl",
                        "name": "run-preflight",
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
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")
    return request, signing_key


def test_import_preflight_validates_without_creating_output(tmp_path: Path) -> None:
    material = _materialize_request(tmp_path)
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")
    output = tmp_path / "artifacts/evidence"
    assert not output.exists()

    started = time.perf_counter()
    result = preflight_evaluation_request(
        Path(material["request"]),
        signing_key_path=signing_key,
    )

    assert time.perf_counter() - started < 2.0
    assert result.format_version == "invarlock/evaluation-preflight-v2"
    assert result.execution_mode == "import"
    assert result.record_count == 2
    assert result.providers == {
        "baseline": "hf_transformers",
        "subject": "hf_transformers",
    }
    assert result.checks == (
        "request",
        "artifacts",
        "dataset_schedule",
        "policy",
        "provider_capabilities",
        "signing_key",
        "output_destination",
    )
    assert not output.exists()


def test_preflight_rejects_schedule_below_policy_minimum_record_count(
    tmp_path: Path,
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    (tmp_path / "inputs/policy.json").write_bytes(
        canonical_json_bytes(
            {
                "resolved_policy": {
                    "metrics": {
                        "exact_match": {
                            "delta_min_pp": -10.0,
                            "minimum_record_count": 400,
                            "maximum_interval_width_pp": 10.0,
                        }
                    }
                }
            }
        )
    )

    with pytest.raises(
        EvaluationPreflightError,
        match="schedule has 1 records but policy requires at least 400",
    ):
        preflight_evaluation_request(request, signing_key_path=signing_key)


def test_import_preflight_qualifies_exact_scorer_binding(tmp_path: Path) -> None:
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path,
        scorer_binding=binding,
        scorer_registry=registry,
    )
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")

    result = preflight_evaluation_request(
        Path(material["request"]),
        signing_key_path=signing_key,
        scorer_registry=registry,
    )

    assert "scorer_binding" in result.checks
    assert not (tmp_path / "artifacts/evidence").exists()


def test_import_preflight_rejects_installed_scorer_descriptor_drift(
    tmp_path: Path,
) -> None:
    fixture_registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path,
        scorer_binding=binding,
        scorer_registry=fixture_registry,
    )
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")

    class VersionDriftScorer(_CasefoldTextScorer):
        def descriptor(self):  # noqa: ANN201
            return dataclasses.replace(
                super().descriptor(),
                scorer_version="2.0.0",
            )

    drift_registry = ScorerExtensionRegistry(
        allow_installed=False,
        authorized=(VersionDriftScorer(),),
    )

    with pytest.raises(EvaluationPreflightError, match="installed descriptor"):
        preflight_evaluation_request(
            Path(material["request"]),
            signing_key_path=signing_key,
            scorer_registry=drift_registry,
        )

    assert not (tmp_path / "artifacts/evidence").exists()


def test_cli_preflight_is_machine_readable_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _materialize_request(tmp_path)
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")
    output = tmp_path / "artifacts/evidence"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("evaluation execution must not run during preflight")

    monkeypatch.setattr(evaluation_transaction, "evaluate_request_file", forbidden)
    invoked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(material["request"]),
            "--signing-key",
            str(signing_key),
            "--preflight",
            "--json",
        ],
    )

    assert invoked.exit_code == 0, invoked.stdout
    payload = json.loads(invoked.stdout)
    assert payload["format_version"] == "invarlock/evaluation-preflight-v2"
    assert payload["ok"] is True
    assert payload["execution_mode"] == "import"
    assert payload["output"] == "artifacts/evidence"
    assert not output.exists()


def test_cli_preflight_default_output_states_that_no_execution_occurred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _materialize_request(tmp_path)
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")
    output = tmp_path / "artifacts/evidence"

    monkeypatch.setattr(
        evaluation_transaction,
        "evaluate_request_file",
        lambda *_args, **_kwargs: pytest.fail(
            "evaluation execution must not run during preflight"
        ),
    )
    invoked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(material["request"]),
            "--signing-key",
            str(signing_key),
            "--preflight",
        ],
    )

    assert invoked.exit_code == 0, invoked.stdout
    assert "PASS Preflight complete" in invoked.stdout
    assert "No execution or publication was performed" in invoked.stdout
    assert not output.exists()


def test_cli_evaluate_runs_full_preflight_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _materialize_request(tmp_path)
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")
    events: list[str] = []
    original_preflight = evaluation_transaction.preflight_evaluation_request
    original_evaluate = evaluation_transaction.evaluate_request_file

    def record_preflight(*args: object, **kwargs: object):  # noqa: ANN202
        events.append("preflight")
        return original_preflight(*args, **kwargs)  # type: ignore[arg-type]

    def record_evaluate(*args: object, **kwargs: object):  # noqa: ANN202
        result = original_evaluate(*args, **kwargs)  # type: ignore[arg-type]
        assert events == ["preflight"]
        events.append("evaluate")
        return result

    monkeypatch.setattr(
        evaluation_transaction,
        "preflight_evaluation_request",
        record_preflight,
    )
    monkeypatch.setattr(
        evaluation_transaction,
        "evaluate_request_file",
        record_evaluate,
    )

    invoked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(material["request"]),
            "--signing-key",
            str(signing_key),
        ],
    )

    assert invoked.exit_code == 0, invoked.stdout
    assert events == ["preflight", "evaluate"]
    assert (tmp_path / "artifacts/evidence").is_dir()


def test_cli_evaluate_rejects_bad_run_artifact_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(
        tmp_path,
        baseline_digest="0" * 64,
    )
    runtime = "sha256:" + "a" * 64
    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda _name: "/bin/docker")
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, _image: evaluation_oci._LocalImageInspection(
            config_id=runtime,
            repo_digests=(),
        ),
    )
    monkeypatch.setattr(
        evaluation_oci.OciRuntimeExecutor,
        "execute",
        lambda *_args, **_kwargs: pytest.fail("worker launch must remain blocked"),
    )

    invoked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image",
            runtime,
            "--runtime-image-digest",
            runtime,
            "--json",
        ],
    )

    assert invoked.exit_code == 2
    payload = json.loads(invoked.stdout)
    assert payload["ok"] is False
    assert "tree digest does not match" in payload["errors"][0]
    assert not (tmp_path / "artifacts").exists()


def test_public_evaluate_api_rejects_bad_run_artifact_before_executor(
    tmp_path: Path,
) -> None:
    request, signing_key = _materialize_run_request(
        tmp_path,
        baseline_digest="0" * 64,
    )
    runtime = "sha256:" + "a" * 64

    class ForbiddenExecutor:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("public API executor must remain blocked")

    with pytest.raises(EvaluationTransactionError, match="tree digest does not match"):
        evaluation_transaction.evaluate_request_file(
            request,
            signing_key_path=signing_key,
            runtime_executor=ForbiddenExecutor(),  # type: ignore[arg-type]
            runtime_image_digests={"baseline": runtime, "subject": runtime},
        )

    assert not (tmp_path / "artifacts").exists()


def test_public_evaluate_api_requires_independent_runtime_digests(
    tmp_path: Path,
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)

    class ForbiddenExecutor:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("public API executor must remain blocked")

    with pytest.raises(
        EvaluationTransactionError,
        match="requires both preflight runtime image digests",
    ):
        evaluation_transaction.evaluate_request_file(
            request,
            signing_key_path=signing_key,
            runtime_executor=ForbiddenExecutor(),  # type: ignore[arg-type]
        )

    assert not (tmp_path / "artifacts").exists()


def test_public_evaluate_api_rejects_split_executor_and_resource_authority(
    tmp_path: Path,
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    runtime = "sha256:" + "a" * 64

    class ForbiddenExecutor:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("public API executor must remain blocked")

    with pytest.raises(EvaluationTransactionError, match="mutually exclusive"):
        evaluation_transaction.evaluate_request_file(
            request,
            signing_key_path=signing_key,
            runtime_executor=ForbiddenExecutor(),  # type: ignore[arg-type]
            resource_resolver=CallerRuntimeResources(container_image_digest=runtime),
            runtime_image_digests={"baseline": runtime, "subject": runtime},
        )

    assert not (tmp_path / "artifacts").exists()


def test_preflight_failure_is_closed_json_and_does_not_create_parents(
    tmp_path: Path,
) -> None:
    material = _materialize_request(tmp_path)
    output_parent = tmp_path / "artifacts"
    assert not output_parent.exists()

    invoked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(material["request"]),
            "--signing-key",
            str(tmp_path / "missing.pem"),
            "--preflight",
            "--json",
        ],
    )

    assert invoked.exit_code == 2
    payload = json.loads(invoked.stdout)
    assert payload["ok"] is False
    assert payload["format_version"] == "invarlock/evaluation-preflight-v2"
    assert "signing key" in payload["errors"][0]
    assert not output_parent.exists()


def test_run_preflight_validates_local_inputs_and_runtime_bindings(
    tmp_path: Path,
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    runtime = "sha256:" + "a" * 64

    result = preflight_evaluation_request(
        request,
        signing_key_path=signing_key,
        runtime_image_digests={"baseline": runtime, "subject": runtime},
    )

    assert result.execution_mode == "run"
    assert result.runtime_image_digests == {
        "baseline": runtime,
        "subject": runtime,
    }
    assert set(result.artifact_digests) == {"baseline", "subject"}
    assert all(
        digest.startswith("sha256:") for digest in result.artifact_digests.values()
    )
    assert result.evidence_signer_fingerprint.startswith("sha256:")
    assert result.request_digest.startswith("sha256:")
    assert result.checks[-1] == "runtime_images"
    assert not (tmp_path / "artifacts").exists()


def test_run_preflight_invokes_optional_provider_input_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    runtime = "sha256:" + "a" * 64
    calls: list[str] = []

    def reject_missing_content(
        _provider: object, _spec: object, _resources: object, _batch: object
    ) -> None:
        calls.append("called")
        raise ValueError("schedule-bound content object is unavailable")

    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.validate_evaluation_inputs",
        reject_missing_content,
        raising=False,
    )

    with pytest.raises(EvaluationPreflightError, match="content object is unavailable"):
        preflight_evaluation_request(
            request,
            signing_key_path=signing_key,
            runtime_image_digests={"baseline": runtime, "subject": runtime},
            resource_resolver=CallerRuntimeResources(
                container_image_digest=runtime,
            ),
        )

    assert calls == ["called"]
    assert not (tmp_path / "artifacts").exists()


def test_run_preflight_requires_resources_for_optional_provider_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    runtime = "sha256:" + "a" * 64
    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.validate_evaluation_inputs",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    with pytest.raises(
        EvaluationPreflightError, match="caller-owned runtime resources"
    ):
        preflight_evaluation_request(
            request,
            signing_key_path=signing_key,
            runtime_image_digests={"baseline": runtime, "subject": runtime},
        )


def test_run_preflight_validates_both_sides_with_their_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    runtime = "sha256:" + "a" * 64
    observed: list[str] = []

    def validate(
        _provider: object, spec: object, _resources: object, _schedule: object
    ) -> None:
        observed.append(spec.model_id)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.validate_evaluation_inputs",
        validate,
        raising=False,
    )

    result = preflight_evaluation_request(
        request,
        signing_key_path=signing_key,
        runtime_image_digests={"baseline": runtime, "subject": runtime},
        resource_resolver=CallerRuntimeResources(container_image_digest=runtime),
    )

    assert observed == ["local/baseline", "local/subject"]
    assert result.checks[-1] == "runtime_resources"


def test_run_preflight_rejects_resource_image_digest_drift(tmp_path: Path) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    inspected = "sha256:" + "a" * 64
    drifted = "sha256:" + "b" * 64

    with pytest.raises(
        EvaluationPreflightError,
        match="baseline caller-owned runtime resources do not match",
    ):
        preflight_evaluation_request(
            request,
            signing_key_path=signing_key,
            runtime_image_digests={"baseline": inspected, "subject": inspected},
            resource_resolver=CallerRuntimeResources(container_image_digest=drifted),
        )


def test_import_preflight_does_not_invoke_runtime_resource_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _materialize_request(tmp_path)
    signing_key, _fingerprint = _key(tmp_path / "evidence.pem")
    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.validate_evaluation_inputs",
        lambda *_args, **_kwargs: pytest.fail(
            "import preflight must not resolve runtime resources"
        ),
        raising=False,
    )

    result = preflight_evaluation_request(
        Path(material["request"]),
        signing_key_path=signing_key,
    )

    assert "runtime_resources" not in result.checks


def test_cli_run_preflight_passes_the_oci_resource_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    runtime = "sha256:" + "a" * 64
    original = evaluation_transaction.preflight_evaluation_request
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.validate_evaluation_inputs",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    monkeypatch.setattr(evaluation_oci.shutil, "which", lambda _name: "/bin/docker")
    monkeypatch.setattr(
        evaluation_oci,
        "_inspect_local_image",
        lambda _engine, _image: evaluation_oci._LocalImageInspection(
            config_id=runtime,
            repo_digests=(),
        ),
    )

    def record_resolver(*args: object, **kwargs: object):  # noqa: ANN202
        observed["resource_resolver"] = kwargs.get("resource_resolver")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        evaluation_transaction,
        "preflight_evaluation_request",
        record_resolver,
    )

    invoked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image",
            runtime,
            "--runtime-image-digest",
            runtime,
            "--preflight",
            "--json",
        ],
    )

    assert invoked.exit_code == 0, invoked.stdout
    assert isinstance(observed["resource_resolver"], evaluation_oci.OciRuntimeExecutor)
    assert "runtime_resources" in json.loads(invoked.stdout)["checks"]


def test_run_preflight_rejects_artifact_digest_mismatch_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path, baseline_digest="0" * 64)

    monkeypatch.setattr(
        evaluation_transaction,
        "execute_runtime_comparison",
        lambda *_args, **_kwargs: pytest.fail("runtime execution must not start"),
    )

    with pytest.raises(EvaluationPreflightError, match="tree digest does not match"):
        preflight_evaluation_request(
            request,
            signing_key_path=signing_key,
            runtime_image_digests={
                "baseline": "sha256:" + "a" * 64,
                "subject": "sha256:" + "a" * 64,
            },
        )

    assert not (tmp_path / "artifacts").exists()


def test_run_preflight_rejects_post_load_artifact_parent_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root = tmp_path / "request-root"
    request_root.mkdir()
    request, signing_key = _materialize_run_request(request_root)
    outside = tmp_path / "outside-models"
    original_load = evaluation_transaction.load_evaluation_request

    def load_then_swap(*args: object, **kwargs: object) -> object:
        loaded = original_load(*args, **kwargs)  # type: ignore[arg-type]
        request_root.joinpath("models").rename(outside)
        request_root.joinpath("models").symlink_to(
            outside,
            target_is_directory=True,
        )
        return loaded

    monkeypatch.setattr(
        evaluation_transaction,
        "load_evaluation_request",
        load_then_swap,
    )

    with pytest.raises(
        EvaluationPreflightError,
        match="baseline artifact could not be authenticated",
    ):
        preflight_evaluation_request(
            request,
            signing_key_path=signing_key,
            runtime_image_digests={
                "baseline": "sha256:" + "a" * 64,
                "subject": "sha256:" + "a" * 64,
            },
        )

    assert request_root.joinpath("models").is_symlink()
    assert not request_root.joinpath("artifacts").exists()


def test_normal_input_preparation_defers_artifact_hashing_to_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)

    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.authenticate_artifact",
        lambda *_args, **_kwargs: pytest.fail(
            "normal host preparation must not hash the checkpoint twice"
        ),
    )

    prepared = evaluation_transaction._prepare_evaluation_inputs(  # noqa: SLF001
        request,
        signing_key_path=signing_key,
        scorer_registry=None,
    )

    assert prepared.request.execution.mode == "run"


def test_run_preflight_contextualizes_provider_authentication_contract_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, signing_key = _materialize_run_request(tmp_path)
    calls = 0

    def authenticate(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TypeError("provider authentication contract failed")
        provider = _args[0]
        return provider.identify_artifact(_args[1])  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.authenticate_artifact",
        authenticate,
    )

    with pytest.raises(
        EvaluationPreflightError,
        match="subject artifact could not be authenticated: provider authentication",
    ):
        preflight_evaluation_request(
            request,
            signing_key_path=signing_key,
            runtime_image_digests={
                "baseline": "sha256:" + "a" * 64,
                "subject": "sha256:" + "a" * 64,
            },
        )

    assert calls == 2
    assert not (tmp_path / "artifacts").exists()
