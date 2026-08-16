from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

import invarlock.core.evaluation_request as evaluation_request
from invarlock.core.evaluation_request import (
    EVALUATION_REQUEST_FORMAT,
    EvaluationRequestError,
    ProviderResolver,
    load_evaluation_request,
)
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeProviderCapabilities,
)
from invarlock.core.schedule_preparation import LocalDatasetRequest


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("base: &base {value: 1}\nitem: {<<: *base}\n", "merge keys"),
        ("{1: value}\n", "non-empty strings"),
        ("flag: TRUE\n", "lowercase true or false"),
        ("count: 01\n", "canonical JSON syntax"),
        ("ratio: .inf\n", "finite JSON syntax"),
        ("ratio: 1.0e+999\n", "must be finite"),
        ("value: Null\n", "lowercase null"),
    ],
)
def test_strict_request_yaml_rejects_ambiguous_scalar_and_merge_syntax(
    payload: str,
    message: str,
) -> None:
    """YAML conveniences must not create alternate request interpretations."""

    with pytest.raises(EvaluationRequestError, match=message):
        yaml.load(payload, Loader=evaluation_request._StrictRequestYamlLoader)


def test_strict_request_yaml_preserves_canonical_json_scalar_semantics() -> None:
    """Accepted scalars must retain their JSON value and type."""

    loaded = yaml.load(
        "enabled: true\ncount: -2\nratio: 1.25\nvalue: null\n",
        Loader=evaluation_request._StrictRequestYamlLoader,
    )

    assert loaded == {
        "enabled": True,
        "count": -2,
        "ratio": 1.25,
        "value": None,
    }


@pytest.mark.parametrize("reference", ["/absolute", "safe/../escape"])
def test_reference_parser_rejects_unsafe_paths_at_owned_boundary(
    reference: str,
) -> None:
    """Root confinement must hold even when schema validation is bypassed."""

    with pytest.raises(EvaluationRequestError, match="safe relative reference"):
        evaluation_request._reference_parts(reference, label="artifact")


def test_existing_file_reference_rejects_directory_at_owned_boundary(
    tmp_path: Path,
) -> None:
    """The descriptor-based resolver must enforce the requested file type."""

    (tmp_path / "directory").mkdir()

    with pytest.raises(EvaluationRequestError, match="regular file"):
        evaluation_request._resolve_existing_reference(
            tmp_path,
            "directory",
            label="policy",
            expected="file",
        )


def test_output_reference_accepts_missing_leaf_below_existing_ancestors(
    tmp_path: Path,
) -> None:
    """A safe nested destination remains creatable after descriptor traversal."""

    (tmp_path / "artifacts" / "nested").mkdir(parents=True)

    resolved = evaluation_request._resolve_output_reference(
        tmp_path,
        "artifacts/nested/evidence",
        label="output.evidence",
    )

    assert resolved == tmp_path / "artifacts/nested/evidence"


def _sha(character: str) -> str:
    return character * 64


def _hf_settings(*, batch_size: int = 1) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "checkpoint_tree_sha256": _sha("a"),
        "context_length": 128,
        "immutable_revision": "b" * 40,
        "max_output_tokens": 16,
        "offline": True,
        "seed": 0,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": _sha("c"),
    }


def _materialize_run_inputs(root: Path) -> None:
    (root / "models/baseline").mkdir(parents=True)
    (root / "models/subject").mkdir(parents=True)
    (root / "datasets").mkdir()
    (root / "datasets/acceptance.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "policy").mkdir()
    (root / "policy/acceptance.yaml").write_text("tier: release\n", encoding="utf-8")


def _request_payload(*, mode: str = "run") -> dict[str, object]:
    execution: dict[str, object] = {"mode": mode}
    if mode == "import":
        execution.update(
            {
                "records": "imports/paired-records.json",
                "schedule": "imports/schedule.json",
                "baseline": {
                    "identity": "imports/baseline/model-artifact.identity.json",
                    "receipt": "imports/baseline/runtime-provider.receipt.json",
                    "observation": "imports/baseline/runtime-scoring.observation.json",
                    "run_report": "imports/baseline/runtime-side.report.json",
                    "runtime_manifest": "imports/baseline/runtime.manifest.json",
                    "runtime_config": "imports/baseline/run.yaml",
                },
                "subject": {
                    "identity": "imports/subject/model-artifact.identity.json",
                    "receipt": "imports/subject/runtime-provider.receipt.json",
                    "observation": "imports/subject/runtime-scoring.observation.json",
                    "run_report": "imports/subject/runtime-side.report.json",
                    "runtime_manifest": "imports/subject/runtime.manifest.json",
                    "runtime_config": "imports/subject/run.yaml",
                },
            }
        )
    dataset: object = "datasets/acceptance.jsonl"
    if mode == "run":
        dataset = {
            "path": "datasets/acceptance.jsonl",
            "sha256": hashlib.sha256(b"{}\n").hexdigest(),
            "format": "jsonl",
            "name": "acceptance",
            "split": "validation",
            "input_field": "prompt",
            "expected_output_field": "expected",
        }
    return {
        "format_version": EVALUATION_REQUEST_FORMAT,
        "comparison": {
            "baseline": {
                "artifact": {
                    "path": "models/baseline",
                    "model_id": "org/baseline",
                    "locator": "hf://org/baseline@" + "b" * 40,
                },
                "runtime": {
                    "provider": "hf_transformers",
                    "settings": _hf_settings(),
                },
            },
            "subject": {
                "artifact": {
                    "path": "models/subject",
                    "model_id": "org/subject",
                    "locator": "hf://org/subject@" + "b" * 40,
                },
                "runtime": {
                    "provider": "hf_transformers",
                    "settings": _hf_settings(batch_size=2),
                },
            },
            "dataset": dataset,
            "policy": "policy/acceptance.yaml",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": execution,
        "output": {"evidence": "artifacts/evidence"},
    }


def _materialize_import_inputs(root: Path) -> None:
    for relative in (
        "imports/paired-records.json",
        "imports/schedule.json",
        "imports/baseline/model-artifact.identity.json",
        "imports/baseline/runtime-provider.receipt.json",
        "imports/baseline/runtime-scoring.observation.json",
        "imports/baseline/runtime-side.report.json",
        "imports/baseline/runtime.manifest.json",
        "imports/baseline/run.yaml",
        "imports/subject/model-artifact.identity.json",
        "imports/subject/runtime-provider.receipt.json",
        "imports/subject/runtime-scoring.observation.json",
        "imports/subject/runtime-side.report.json",
        "imports/subject/runtime.manifest.json",
        "imports/subject/run.yaml",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def _write_request(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _valid_request(tmp_path: Path, *, mode: str = "run") -> Path:
    _materialize_run_inputs(tmp_path)
    if mode == "import":
        _materialize_import_inputs(tmp_path)
    return _write_request(tmp_path / "request.yaml", _request_payload(mode=mode))


class _TaskContractProvider:
    abi_version = "1"

    def __init__(self, name: str, *, tasks: tuple[str, ...]) -> None:
        self.name = name
        self._capabilities = RuntimeProviderCapabilities(
            provider_name=name,
            artifact_formats=("hf_snapshot",),
            tasks=tasks,
            metrics=("exact_match",),
            execution_modes=("in_process",),
            required_extra=None,
            required_image=None,
        )

    def validate_config(self, _spec: ModelRuntimeSpec) -> None:
        return None

    def capabilities(self) -> RuntimeProviderCapabilities:
        return self._capabilities


def _task_provider_resolver(
    task_support: dict[str, tuple[str, ...]],
) -> ProviderResolver:
    providers = {
        name: _TaskContractProvider(name, tasks=tasks)
        for name, tasks in task_support.items()
    }
    return lambda name: providers[name]  # type: ignore[return-value]


def test_load_run_request_resolves_two_closed_runtime_sides(tmp_path: Path) -> None:
    request = load_evaluation_request(_valid_request(tmp_path))

    assert request.format_version == EVALUATION_REQUEST_FORMAT
    assert request.root == tmp_path.resolve()
    assert request.comparison.baseline.artifact.path == (tmp_path / "models/baseline")
    assert request.comparison.baseline.runtime.provider == "hf_transformers"
    assert request.comparison.subject.runtime.settings["batch_size"] == 2
    assert request.comparison.metric == "exact_match"
    assert isinstance(request.comparison.dataset, LocalDatasetRequest)
    assert request.comparison.dataset.path == tmp_path / "datasets/acceptance.jsonl"
    assert request.comparison.dataset.sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert request.comparison.dataset.input_field == "prompt"
    assert request.execution.mode == "run"
    assert request.execution.records is None
    assert request.output.evidence == tmp_path / "artifacts/evidence"
    with pytest.raises(TypeError):
        request.comparison.subject.runtime.settings["batch_size"] = 8  # type: ignore[index]


@pytest.mark.parametrize(
    "host_control",
    ["runtime_cpus", "runtime_memory_mib", "runtime_user", "container_engine"],
)
def test_run_request_cannot_select_caller_owned_oci_host_controls(
    tmp_path: Path,
    host_control: str,
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution[host_control] = "untrusted-request-value"

    with pytest.raises(EvaluationRequestError, match="does not match"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_request_resolves_optional_observation_payloads_without_embedding_paths(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    observation = tmp_path / "observations/subject-spectral.json"
    observation.parent.mkdir()
    observation.write_text(
        '{"format":"invarlock/diagnostic-observation-v1","kind":"spectral",'
        '"status":"observation"}\n',
        encoding="utf-8",
    )
    payload = _request_payload()
    payload["observations"] = [
        {
            "id": "subject-spectral",
            "kind": "spectral",
            "scope": "subject",
            "path": "observations/subject-spectral.json",
        }
    ]

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", payload)
    )

    assert len(request.observations) == 1
    assert request.observations[0].path == observation
    assert request.observations[0].scope == "subject"


def test_request_rejects_duplicate_or_prepared_observation_entries(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    observation = tmp_path / "observation.json"
    observation.write_text("{}\n", encoding="utf-8")
    entry = {
        "id": "same",
        "kind": "variance",
        "scope": "comparison",
        "path": "observation.json",
    }
    payload = _request_payload()
    payload["observations"] = [entry, dict(entry)]
    with pytest.raises(EvaluationRequestError, match="duplicate observation id"):
        load_evaluation_request(_write_request(tmp_path / "duplicate.yaml", payload))

    payload["observations"] = [
        {
            "id": "same",
            "kind": "variance",
            "scope": "comparison",
            "payload_digest": "sha256:" + "a" * 64,
        }
    ]
    with pytest.raises(EvaluationRequestError, match="authored observation with path"):
        load_evaluation_request(_write_request(tmp_path / "prepared.yaml", payload))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_request_rejects_observation_payload_symlink_escape(tmp_path: Path) -> None:
    request_root = tmp_path / "request"
    request_root.mkdir()
    _materialize_run_inputs(request_root)
    outside = tmp_path / "outside-observation.json"
    outside.write_text('{"status":"observation"}\n', encoding="utf-8")
    observations = request_root / "observations"
    observations.mkdir()
    (observations / "subject-variance.json").symlink_to(outside)
    payload = _request_payload()
    payload["observations"] = [
        {
            "id": "subject-variance",
            "kind": "variance",
            "scope": "subject",
            "path": "observations/subject-variance.json",
        }
    ]

    with pytest.raises(EvaluationRequestError, match="symlink"):
        load_evaluation_request(_write_request(request_root / "request.yaml", payload))


def test_relative_references_are_anchored_to_request_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _valid_request(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    request = load_evaluation_request(request_path)

    assert isinstance(request.comparison.dataset, LocalDatasetRequest)
    assert request.comparison.dataset.path == tmp_path / "datasets/acceptance.jsonl"
    assert request.comparison.policy == tmp_path / "policy/acceptance.yaml"


def test_load_import_request_requires_all_authenticated_replay_inputs(
    tmp_path: Path,
) -> None:
    request = load_evaluation_request(_valid_request(tmp_path, mode="import"))

    assert request.execution.mode == "import"
    assert request.execution.records == tmp_path / "imports/paired-records.json"
    assert request.execution.schedule == tmp_path / "imports/schedule.json"
    assert request.execution.baseline is not None
    assert request.execution.baseline.identity.name == "model-artifact.identity.json"
    assert request.execution.subject is not None
    assert request.execution.subject.receipt.name == "runtime-provider.receipt.json"
    assert request.comparison.baseline.artifact.path is None
    assert request.comparison.baseline.artifact.model_id == "org/baseline"
    assert request.comparison.baseline.artifact.locator is not None
    assert request.comparison.dataset == tmp_path / "datasets/acceptance.jsonl"


def test_run_requires_pinned_local_dataset_object(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["dataset"] = "datasets/acceptance.jsonl"

    with pytest.raises(EvaluationRequestError, match="pinned local dataset object"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_import_requires_canonical_schedule_reference(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    _materialize_import_inputs(tmp_path)
    payload = _request_payload(mode="import")
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["dataset"] = {
        "path": "datasets/acceptance.jsonl",
        "sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "format": "jsonl",
        "name": "acceptance",
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "expected",
    }

    with pytest.raises(EvaluationRequestError, match="canonical schedule"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_import_does_not_require_source_model_bytes(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    _materialize_import_inputs(tmp_path)
    payload = _request_payload(mode="import")
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    for side_name in ("baseline", "subject"):
        side = comparison[side_name]
        assert isinstance(side, dict)
        artifact = side["artifact"]
        assert isinstance(artifact, dict)
        del artifact["path"]
    (tmp_path / "models/baseline").rmdir()
    (tmp_path / "models/subject").rmdir()
    (tmp_path / "models").rmdir()

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", payload)
    )

    assert request.execution.mode == "import"
    assert request.comparison.baseline.artifact.path is None
    assert request.comparison.subject.artifact.path is None


def test_run_requires_real_local_artifact_bytes(tmp_path: Path) -> None:
    request_path = _valid_request(tmp_path)
    (tmp_path / "models/subject").rmdir()

    with pytest.raises(EvaluationRequestError, match="subject.artifact.path"):
        load_evaluation_request(request_path)


def test_run_requires_artifact_path_field(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    baseline = comparison["baseline"]
    assert isinstance(baseline, dict)
    artifact = baseline["artifact"]
    assert isinstance(artifact, dict)
    del artifact["path"]

    with pytest.raises(EvaluationRequestError, match="artifact.path is required"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_artifact_requires_stable_locator_metadata(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    subject = comparison["subject"]
    assert isinstance(subject, dict)
    artifact = subject["artifact"]
    assert isinstance(artifact, dict)
    del artifact["locator"]

    with pytest.raises(EvaluationRequestError, match="locator"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


@pytest.mark.parametrize(
    "missing",
    ["records", "schedule", "baseline", "subject"],
)
def test_import_rejects_missing_replay_anchor(tmp_path: Path, missing: str) -> None:
    _materialize_run_inputs(tmp_path)
    _materialize_import_inputs(tmp_path)
    payload = _request_payload(mode="import")
    execution = payload["execution"]
    assert isinstance(execution, dict)
    del execution[missing]

    with pytest.raises(EvaluationRequestError, match=missing):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize(
    "missing",
    [
        "identity",
        "receipt",
        "observation",
        "run_report",
        "runtime_manifest",
        "runtime_config",
    ],
)
def test_import_side_rejects_missing_provider_evidence_reference(
    tmp_path: Path, side: str, missing: str
) -> None:
    _materialize_run_inputs(tmp_path)
    _materialize_import_inputs(tmp_path)
    payload = _request_payload(mode="import")
    execution = payload["execution"]
    assert isinstance(execution, dict)
    side_payload = execution[side]
    assert isinstance(side_payload, dict)
    del side_payload[missing]

    with pytest.raises(EvaluationRequestError, match=missing):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_run_rejects_import_only_fields(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["records"] = "imports/paired-records.json"

    with pytest.raises(EvaluationRequestError, match="records"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


@pytest.mark.parametrize(
    "setting",
    ["allow_network", "trust_remote_code", "plugin_install", "host_path"],
)
def test_provider_owned_settings_reject_capability_grants(
    tmp_path: Path, setting: str
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    baseline = comparison["baseline"]
    assert isinstance(baseline, dict)
    runtime = baseline["runtime"]
    assert isinstance(runtime, dict)
    settings = runtime["settings"]
    assert isinstance(settings, dict)
    settings[setting] = True

    with pytest.raises(EvaluationRequestError, match="unsupported hf_transformers"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_provider_owned_settings_reject_unknown_setting(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    subject = comparison["subject"]
    assert isinstance(subject, dict)
    runtime = subject["runtime"]
    assert isinstance(runtime, dict)
    settings = runtime["settings"]
    assert isinstance(settings, dict)
    settings["socket_access"] = True

    with pytest.raises(EvaluationRequestError, match="unsupported hf_transformers"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_runtime_settings_are_model_runtime_spec_scalars(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    baseline = comparison["baseline"]
    assert isinstance(baseline, dict)
    runtime = baseline["runtime"]
    assert isinstance(runtime, dict)
    settings = runtime["settings"]
    assert isinstance(settings, dict)
    settings["batch_size"] = [1]

    with pytest.raises(EvaluationRequestError, match="batch_size"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_unknown_runtime_provider_fails_closed(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    subject = comparison["subject"]
    assert isinstance(subject, dict)
    runtime = subject["runtime"]
    assert isinstance(runtime, dict)
    runtime["provider"] = "missing_provider"

    with pytest.raises(EvaluationRequestError, match="not installed or authorized"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_provider_resolution_rejects_identity_abi_and_unexpected_loader_failures() -> (
    None
):
    wrong_identity = _TaskContractProvider("other", tasks=("text_causal",))
    with pytest.raises(EvaluationRequestError, match="identity mismatch"):
        evaluation_request._resolve_provider(
            "requested",
            resolver=lambda _name: wrong_identity,  # type: ignore[arg-type]
        )

    wrong_abi = _TaskContractProvider("requested", tasks=("text_causal",))
    wrong_abi.abi_version = "unsupported"
    with pytest.raises(EvaluationRequestError, match="unsupported ABI"):
        evaluation_request._resolve_provider(
            "requested",
            resolver=lambda _name: wrong_abi,  # type: ignore[arg-type]
        )

    def broken_resolver(_name: str) -> object:
        raise RuntimeError("entry-point loader failed")

    with pytest.raises(EvaluationRequestError, match="could not be resolved"):
        evaluation_request._resolve_provider(
            "requested",
            resolver=broken_resolver,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "reference",
    ["/tmp/model", "../outside", "file://outside", "C:\\outside", "./models/subject"],
)
def test_all_input_references_must_be_request_relative(
    tmp_path: Path, reference: str
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    subject = comparison["subject"]
    assert isinstance(subject, dict)
    artifact = subject["artifact"]
    assert isinstance(artifact, dict)
    artifact["path"] = reference

    with pytest.raises(EvaluationRequestError, match="artifact.path"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


@pytest.mark.parametrize(
    "reference", ["/tmp/evidence", "../evidence", "https://example.invalid/evidence"]
)
def test_output_reference_must_be_request_relative(
    tmp_path: Path, reference: str
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    output = payload["output"]
    assert isinstance(output, dict)
    output["evidence"] = reference

    with pytest.raises(EvaluationRequestError, match="output.evidence"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


@pytest.mark.parametrize(
    "metric", ["loss", "accuracy", "python:custom_metric", "exact_match_v2"]
)
def test_metric_is_closed(tmp_path: Path, metric: str) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["metric"] = metric

    with pytest.raises(EvaluationRequestError, match="metric"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_normalized_nll_is_supported_by_both_hf_runtime_sides(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["metric"] = "normalized_nll_per_utf8_byte"

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", payload)
    )

    assert request.comparison.metric == "normalized_nll_per_utf8_byte"


def test_request_selects_exactly_one_text_scorer_extension(tmp_path: Path) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison.pop("metric")
    comparison["scorer_extension"] = {
        "format_version": "invarlock/scorer-extension-binding-v1",
        "scorer_abi": "1",
        "scorer_id": "example.casefold_exact",
        "scorer_version": "1.0.0",
        "descriptor_sha256": "d" * 64,
        "configuration": {},
        "configuration_sha256": hashlib.sha256(b"{}").hexdigest(),
    }

    request = load_evaluation_request(
        _write_request(tmp_path / "extension.yaml", payload)
    )
    assert request.comparison.metric is None
    assert request.comparison.collection_metric == "exact_match"
    assert request.comparison.scorer_extension is not None
    assert request.comparison.scorer_extension.scorer_id == "example.casefold_exact"

    comparison["metric"] = "exact_match"
    with pytest.raises(EvaluationRequestError, match="not valid under any"):
        load_evaluation_request(_write_request(tmp_path / "both.yaml", payload))

    comparison.pop("metric")
    comparison.pop("scorer_extension")
    with pytest.raises(EvaluationRequestError, match="not valid under any"):
        load_evaluation_request(_write_request(tmp_path / "neither.yaml", payload))


def test_request_task_must_be_canonical_and_declared_by_both_providers(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["task"] = "vision_text_generation"
    with pytest.raises(EvaluationRequestError, match="does not support task"):
        load_evaluation_request(_write_request(tmp_path / "vision.yaml", payload))

    comparison["task"] = "vision/text"
    with pytest.raises(EvaluationRequestError, match="task"):
        load_evaluation_request(_write_request(tmp_path / "invalid.yaml", payload))


def test_future_task_contract_accepts_matching_providers_and_names_mismatched_side(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["task"] = "audio_transcription_review"
    baseline = comparison["baseline"]
    subject = comparison["subject"]
    assert isinstance(baseline, dict)
    assert isinstance(subject, dict)
    baseline_runtime = baseline["runtime"]
    subject_runtime = subject["runtime"]
    assert isinstance(baseline_runtime, dict)
    assert isinstance(subject_runtime, dict)
    baseline_runtime["provider"] = "baseline_future"
    subject_runtime["provider"] = "subject_future"
    request_path = _write_request(tmp_path / "future.yaml", payload)

    matching = _task_provider_resolver(
        {
            "baseline_future": ("audio_transcription_review",),
            "subject_future": ("audio_transcription_review",),
        }
    )
    request = load_evaluation_request(request_path, provider_resolver=matching)
    assert request.comparison.task == "audio_transcription_review"

    mismatched = _task_provider_resolver(
        {
            "baseline_future": ("audio_transcription_review",),
            "subject_future": ("text_causal",),
        }
    )
    with pytest.raises(
        EvaluationRequestError,
        match=(
            r"comparison\.subject\.runtime provider 'subject_future' "
            r"does not support task 'audio_transcription_review'"
        ),
    ):
        load_evaluation_request(request_path, provider_resolver=mismatched)


def test_request_rejects_metric_not_declared_by_resolved_provider(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["metric"] = "normalized_nll_per_utf8_byte"
    for side_name in ("baseline", "subject"):
        side = comparison[side_name]
        assert isinstance(side, dict)
        runtime = side["runtime"]
        assert isinstance(runtime, dict)
        runtime["provider"] = f"{side_name}_exact_only"
    resolver = _task_provider_resolver(
        {
            "baseline_exact_only": ("text_causal",),
            "subject_exact_only": ("text_causal",),
        }
    )

    with pytest.raises(EvaluationRequestError, match="does not support metric"):
        load_evaluation_request(
            _write_request(tmp_path / "unsupported-metric.yaml", payload),
            provider_resolver=resolver,
        )


def test_run_request_authenticates_content_role_with_all_content_mappings(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["task"] = "vision_text_generation"
    dataset = comparison["dataset"]
    baseline = comparison["baseline"]
    subject = comparison["subject"]
    assert isinstance(dataset, dict)
    assert isinstance(baseline, dict)
    assert isinstance(subject, dict)
    dataset.update(
        {
            "content_role": "image",
            "content_id_field": "content_id",
            "content_sha256_field": "content_sha256",
            "content_byte_length_field": "content_bytes",
            "content_media_type_field": "content_media_type",
        }
    )
    baseline_runtime = baseline["runtime"]
    subject_runtime = subject["runtime"]
    assert isinstance(baseline_runtime, dict)
    assert isinstance(subject_runtime, dict)
    baseline_runtime["provider"] = "baseline_vision"
    subject_runtime["provider"] = "subject_vision"
    resolver = _task_provider_resolver(
        {
            "baseline_vision": ("vision_text_generation",),
            "subject_vision": ("vision_text_generation",),
        }
    )

    request = load_evaluation_request(
        _write_request(tmp_path / "vision.yaml", payload),
        provider_resolver=resolver,
    )

    assert isinstance(request.comparison.dataset, LocalDatasetRequest)
    assert request.comparison.dataset.content_role == "image"

    del dataset["content_role"]
    with pytest.raises(EvaluationRequestError, match="content_role"):
        load_evaluation_request(
            _write_request(tmp_path / "missing-role.yaml", payload),
            provider_resolver=resolver,
        )


def test_run_request_accepts_future_content_role_and_rejects_mapping_collision(
    tmp_path: Path,
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["task"] = "audio_text_generation"
    dataset = comparison["dataset"]
    baseline = comparison["baseline"]
    subject = comparison["subject"]
    assert isinstance(dataset, dict)
    assert isinstance(baseline, dict)
    assert isinstance(subject, dict)
    dataset.update(
        {
            "content_role": "audio",
            "content_id_field": "content_id",
            "content_sha256_field": "content_sha256",
            "content_byte_length_field": "content_bytes",
            "content_media_type_field": "content_media_type",
        }
    )
    baseline_runtime = baseline["runtime"]
    subject_runtime = subject["runtime"]
    assert isinstance(baseline_runtime, dict)
    assert isinstance(subject_runtime, dict)
    baseline_runtime["provider"] = "baseline_audio"
    subject_runtime["provider"] = "subject_audio"
    resolver = _task_provider_resolver(
        {
            "baseline_audio": ("audio_text_generation",),
            "subject_audio": ("audio_text_generation",),
        }
    )

    request = load_evaluation_request(
        _write_request(tmp_path / "audio.yaml", payload),
        provider_resolver=resolver,
    )

    assert isinstance(request.comparison.dataset, LocalDatasetRequest)
    assert request.comparison.dataset.content_role == "audio"
    assert request.comparison.dataset.content_media_type_field == ("content_media_type")

    dataset["content_id_field"] = "prompt"
    with pytest.raises(EvaluationRequestError, match="field mappings.*distinct"):
        load_evaluation_request(
            _write_request(tmp_path / "colliding-fields.yaml", payload),
            provider_resolver=resolver,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_artifact_reference_rejects_symlink_escape(tmp_path: Path) -> None:
    request_root = tmp_path / "request-root"
    request_root.mkdir()
    _materialize_run_inputs(request_root)
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (request_root / "models/subject").rmdir()
    (request_root / "models/subject").symlink_to(outside, target_is_directory=True)

    with pytest.raises(EvaluationRequestError, match="symlink"):
        load_evaluation_request(
            _write_request(request_root / "request.yaml", _request_payload())
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_output_reference_rejects_symlinked_existing_ancestor(tmp_path: Path) -> None:
    request_root = tmp_path / "request-root"
    request_root.mkdir()
    _materialize_run_inputs(request_root)
    outside = tmp_path / "outside-output"
    outside.mkdir()
    (request_root / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(EvaluationRequestError, match="symlink"):
        load_evaluation_request(
            _write_request(request_root / "request.yaml", _request_payload())
        )


def test_output_destination_must_not_exist(tmp_path: Path) -> None:
    request_path = _valid_request(tmp_path)
    (tmp_path / "artifacts/evidence").mkdir(parents=True)

    with pytest.raises(EvaluationRequestError, match="already exists"):
        load_evaluation_request(request_path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unexpected",), True),
        (("comparison", "unexpected"), True),
        (("comparison", "baseline", "unexpected"), True),
        (("comparison", "subject", "runtime", "unexpected"), True),
        (("execution", "unexpected"), True),
        (("output", "unexpected"), True),
    ],
)
def test_request_rejects_unknown_contract_fields(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    _materialize_run_inputs(tmp_path)
    payload = _request_payload()
    cursor: dict[str, object] = payload
    for component in path[:-1]:
        child = cursor[component]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = value

    with pytest.raises(
        EvaluationRequestError, match="unexpected|Additional properties"
    ):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


@pytest.mark.parametrize(
    "document, expected",
    [
        (
            """\
format_version: invarlock/evaluation-request-v1
format_version: invarlock/evaluation-request-v1
""",
            "duplicate key",
        ),
        ("shared: &shared value\ncopy: *shared\n", "aliases and anchors"),
        ("comparison: !include other.yaml\n", "explicit tags"),
        ("include: other.yaml\n", "include directives"),
    ],
)
def test_request_rejects_ambiguous_or_composed_yaml(
    tmp_path: Path, document: str, expected: str
) -> None:
    path = tmp_path / "request.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(EvaluationRequestError, match=expected):
        load_evaluation_request(path)


def test_request_rejects_more_than_ten_thousand_yaml_nodes(tmp_path: Path) -> None:
    payload = _request_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    baseline = comparison["baseline"]
    assert isinstance(baseline, dict)
    runtime = baseline["runtime"]
    assert isinstance(runtime, dict)
    runtime["settings"] = {f"setting_{index}": index for index in range(10_001)}

    with pytest.raises(EvaluationRequestError, match="10,000-node"):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", payload))


def test_request_rejects_more_than_64_levels_of_nesting(tmp_path: Path) -> None:
    path = tmp_path / "request.yaml"
    path.write_text("[" * 65 + "null" + "]" * 65, encoding="utf-8")

    with pytest.raises(EvaluationRequestError, match="64-level"):
        load_evaluation_request(path)


def test_request_rejects_files_over_one_mibibyte(tmp_path: Path) -> None:
    path = tmp_path / "request.yaml"
    path.write_bytes(b"#" + b"x" * (1024 * 1024))

    with pytest.raises(EvaluationRequestError, match="1048576-byte size limit"):
        load_evaluation_request(path)
