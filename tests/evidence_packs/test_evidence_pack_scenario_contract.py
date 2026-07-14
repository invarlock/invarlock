from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock import evidence_pack_scenario_contract as scenario_contract
from invarlock.evidence_pack_scenario_contract import (
    ArtifactClass,
    GenerationKind,
    ProofHandler,
    ScenarioContractError,
    Strictness,
    parse_scenario_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _validation_edit(
    edit_spec: str,
    *,
    version: str = "stress",
    artifact_class: str = "validation_subject_checkpoint",
    runnable: bool | None = None,
    strictness: str = "informational",
) -> dict[str, object]:
    edit_type = edit_spec.split(":", 1)[0]
    if edit_type in {"lora_merge", "fine_tune"} and version == "stress":
        version = "trained"
    record: dict[str, object] = {
        "id": "validation_edit",
        "artifact_class": artifact_class,
        "strictness": strictness,
        "generation": {
            "kind": "edit",
            "edit_spec": edit_spec,
            "version": version,
        },
    }
    if runnable is not None:
        record["runnable"] = runnable
    if edit_type in {"lora_merge", "fine_tune"}:
        profile_id = (
            "tiny_gpt2_lora_v1" if edit_type == "lora_merge" else "tiny_gpt2_full_ft_v1"
        )
        record["training_profile"] = {
            "profile_id": profile_id,
            "profile_sha256": "sha256:" + "a" * 64,
            "snapshot_path": f"metadata/training_profiles/{profile_id}.json",
            "snapshot_sha256": "sha256:" + "b" * 64,
        }
    return record


def _deployable_bnb(
    edit_spec: str = "bnb_8bit:8:all",
    *,
    backend: str = "bitsandbytes",
    version: str = "deployable",
    artifact_class: str = "deployable_optimized_subject",
) -> dict[str, object]:
    return {
        "id": "deployable_bnb",
        "artifact_class": artifact_class,
        "strictness": "informational",
        "runnable": True,
        "optimized_deployment_backend": True,
        "generation": {
            "kind": "deployable_edit",
            "backend": backend,
            "edit_spec": edit_spec,
            "version": version,
        },
    }


def _error(
    error_type: str = "nan_injection",
    *,
    artifact_class: str = "fault_injection_fixture",
) -> dict[str, object]:
    return {
        "id": "error_fixture",
        "artifact_class": artifact_class,
        "strictness": "informational",
        "generation": {"kind": "error", "error_type": error_type},
    }


def _evidence_only() -> dict[str, object]:
    return {
        "id": "report-001",
        "artifact_class": "evidence_only_pack",
        "strictness": "must_pass",
        "generation": {"kind": "evidence_only"},
    }


def test_validation_transform_is_typed_and_canonical_before_dispatch() -> None:
    contract = parse_scenario_contract(
        _validation_edit("quant_rtn:4:32:ffn@layers=2,layer=1")
    )

    assert contract.generation_kind is GenerationKind.EDIT
    assert contract.artifact_class is ArtifactClass.VALIDATION_SUBJECT_CHECKPOINT
    assert contract.strictness is Strictness.INFORMATIONAL
    assert contract.proof_handler is ProofHandler.TRANSFORMATION_REPLAY
    assert contract.edit is not None
    assert contract.edit.edit_type == "quant_rtn"
    assert contract.edit.canonical_spec == "quant_rtn:4:32:ffn@layers=2,layer=1"
    assert dict(contract.edit.parameters) == {"bits": 4, "group_size": 32}
    assert contract.edit.scope == "ffn@layers=2,layer=1"
    assert not contract.edit.is_clean


@pytest.mark.parametrize(
    ("edit_spec", "expected_parameters"),
    [
        ("magnitude_prune:0.5:all", {"target_sparsity": 0.5}),
        ("synthetic_lowrank_delta:8:64:attn", {"rank": 8, "scale": 64.0}),
        (
            "synthetic_dense_update:0.0005:3:ffn",
            {"step_size": 0.0005, "iterations": 3},
        ),
    ],
)
def test_supported_validation_edits_are_closed_and_typed(
    edit_spec: str, expected_parameters: dict[str, int | float]
) -> None:
    contract = parse_scenario_contract(_validation_edit(edit_spec))

    assert contract.edit is not None
    assert dict(contract.edit.parameters) == expected_parameters
    assert contract.edit.canonical_spec == edit_spec
    if edit_spec.startswith("magnitude_prune:"):
        assert contract.proof_handler is ProofHandler.MAGNITUDE_PRUNING_REPLAY
    else:
        assert contract.proof_handler is ProofHandler.TRANSFORMATION_REPLAY


def test_clean_validation_edit_has_no_hidden_static_parameter_or_scope() -> None:
    contract = parse_scenario_contract(
        _validation_edit("magnitude_prune:clean", version="clean")
    )

    assert contract.edit is not None
    assert contract.edit.is_clean
    assert contract.edit.parameters == ()
    assert contract.edit.scope is None
    assert contract.edit.canonical_spec == "magnitude_prune:clean"


def test_deployable_bitsandbytes_is_bound_to_its_generation_kind_and_backend() -> None:
    contract = parse_scenario_contract(_deployable_bnb())

    assert contract.generation_kind is GenerationKind.DEPLOYABLE_EDIT
    assert contract.artifact_class is ArtifactClass.DEPLOYABLE_OPTIMIZED_SUBJECT
    assert contract.proof_handler is ProofHandler.DEPLOYABLE_BITSANDBYTES
    assert contract.edit is not None
    assert contract.edit.edit_type == "bnb_8bit"
    assert contract.edit.backend == "bitsandbytes"
    assert dict(contract.edit.parameters) == {"bits": 8}
    assert contract.edit.scope == "all"


@pytest.mark.parametrize(
    ("edit_spec", "expected_parameters"),
    [
        ("lora_merge:4:8:attn", {"rank": 4, "alpha": 8.0}),
        ("fine_tune:0.0001:2:ffn", {"learning_rate": 0.0001, "steps": 2}),
    ],
)
def test_real_training_edits_are_typed_but_route_to_external_proof(
    edit_spec: str, expected_parameters: dict[str, int | float]
) -> None:
    contract = parse_scenario_contract(_validation_edit(edit_spec))

    assert contract.edit is not None
    assert contract.edit.canonical_spec == edit_spec
    assert dict(contract.edit.parameters) == expected_parameters
    assert contract.proof_handler is ProofHandler.EXTERNAL_TRAINING


def test_error_fixture_is_typed_and_routes_without_an_edit_spec() -> None:
    contract = parse_scenario_contract(_error())

    assert contract.generation_kind is GenerationKind.ERROR
    assert contract.artifact_class is ArtifactClass.FAULT_INJECTION_FIXTURE
    assert contract.proof_handler is ProofHandler.ERROR_INJECTION
    assert contract.edit is None
    assert contract.error_type == "nan_injection"


def test_evidence_only_report_is_typed_without_an_edit_or_proof_route() -> None:
    contract = parse_scenario_contract(_evidence_only())

    assert contract.generation_kind is GenerationKind.EVIDENCE_ONLY
    assert contract.artifact_class is ArtifactClass.EVIDENCE_ONLY_PACK
    assert contract.proof_handler is ProofHandler.EVIDENCE_ONLY
    assert contract.edit is None
    assert contract.error is None


@pytest.mark.parametrize("scenario_id", ["report--001", "report__001", "Report-001"])
def test_noncanonical_evidence_only_identifiers_fail_closed(scenario_id: str) -> None:
    record = _evidence_only()
    record["id"] = scenario_id

    with pytest.raises(ScenarioContractError, match="canonical identifier"):
        parse_scenario_contract(record)


@pytest.mark.parametrize(
    "edit_spec",
    [
        "Quant_Rtn:4:32:all",
        "quant-rtn:4:32:all",
        "bnb8:8:all",
        "fp8_quant:8:all",
        "lowrank_svd:4:all",
        "unknown_edit:1:all",
    ],
)
def test_aliases_and_unsupported_edit_labels_fail_closed(edit_spec: str) -> None:
    with pytest.raises(ScenarioContractError, match="unsupported|canonical"):
        parse_scenario_contract(_validation_edit(edit_spec))


@pytest.mark.parametrize(
    "edit_spec",
    [
        "quant_rtn:clean:ffn",
        "quant_rtn:Clean",
        "quant_rtn:clean ",
        "magnitude_prune:clean:all",
        "synthetic_lowrank_delta:clean:",
        "lora_merge:clean:attn",
        "bnb_8bit:clean",
    ],
)
def test_malformed_or_unsupported_clean_specs_fail_closed(edit_spec: str) -> None:
    kind = "deployable_edit" if edit_spec.startswith("bnb_") else "edit"
    if kind == "deployable_edit":
        record = _deployable_bnb(edit_spec)
    else:
        record = _validation_edit(edit_spec, version="clean")

    with pytest.raises(ScenarioContractError, match="clean|canonical"):
        parse_scenario_contract(record)


@pytest.mark.parametrize(
    "edit_spec",
    [
        "quant_rtn:04:32:all",
        "quant_rtn:4:032:all",
        "quant_rtn:4:32:FFN",
        "synthetic_dense_update:0.00050:3:ffn",
        "magnitude_prune:0.50:all",
        "magnitude_prune:0.5:all@layer=0",
        "synthetic_dense_update:0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001:3:ffn",
    ],
)
def test_noncanonical_parameter_and_scope_spellings_fail_closed(edit_spec: str) -> None:
    with pytest.raises(ScenarioContractError, match="canonical"):
        parse_scenario_contract(_validation_edit(edit_spec))


@pytest.mark.parametrize(
    "record",
    [
        {
            "id": "missing_generation",
            "artifact_class": "fault_injection_fixture",
            "strictness": "informational",
        },
        {
            "id": "missing_kind",
            "artifact_class": "fault_injection_fixture",
            "strictness": "informational",
            "generation": {"error_type": "nan_injection"},
        },
        {
            "id": "unknown_kind",
            "artifact_class": "fault_injection_fixture",
            "strictness": "informational",
            "generation": {"kind": "legacy_edit", "edit_spec": "noop"},
        },
    ],
)
def test_missing_or_unknown_generation_kind_fails_closed(
    record: dict[str, object],
) -> None:
    with pytest.raises(
        ScenarioContractError, match="generation.*kind|generation is required"
    ):
        parse_scenario_contract(record)


@pytest.mark.parametrize(
    "record",
    [
        _validation_edit("quant_rtn:4:32:all", runnable=False),
        _validation_edit(
            "quant_rtn:4:32:all",
            artifact_class="deployable_optimized_subject",
        ),
        _deployable_bnb(artifact_class="validation_subject_checkpoint"),
        _deployable_bnb(backend="gptq"),
        _error(artifact_class="validation_subject_checkpoint"),
    ],
)
def test_nonrunnable_reports_and_kind_artifact_conflicts_fail_closed(
    record: dict[str, object],
) -> None:
    with pytest.raises(ScenarioContractError, match="runnable|artifact_class|backend"):
        parse_scenario_contract(record)


def test_explicit_nonboolean_runnable_value_fails_closed() -> None:
    record = _validation_edit("quant_rtn:4:32:all")
    record["runnable"] = None

    with pytest.raises(ScenarioContractError, match="runnable must be a boolean"):
        parse_scenario_contract(record)


def test_missing_or_alias_strictness_fails_closed() -> None:
    record = _validation_edit("quant_rtn:4:32:all")
    record.pop("strictness")
    with pytest.raises(ScenarioContractError, match="strictness is required"):
        parse_scenario_contract(record)

    record["strictness"] = "pass"
    with pytest.raises(ScenarioContractError, match="strictness.*unsupported"):
        parse_scenario_contract(record)


def test_generation_shape_is_closed_and_error_environment_is_constrained() -> None:
    record = _validation_edit("quant_rtn:4:32:all")
    generation = record["generation"]
    assert isinstance(generation, dict)
    generation["handler"] = "arbitrary.module:handler"
    with pytest.raises(ScenarioContractError, match="unsupported fields"):
        parse_scenario_contract(record)

    error = _error()
    error_generation = error["generation"]
    assert isinstance(error_generation, dict)
    error_generation["env"] = {"UNSCOPED": "1"}
    with pytest.raises(ScenarioContractError, match="environment key"):
        parse_scenario_contract(error)


def test_real_training_scenarios_require_a_closed_profile_snapshot_binding() -> None:
    record = _validation_edit("lora_merge:2:4:attn")
    binding = record.pop("training_profile")
    assert isinstance(binding, dict)
    with pytest.raises(
        ScenarioContractError, match="training_profile must be an object"
    ):
        parse_scenario_contract(record)

    record["training_profile"] = {**binding, "profile_sha256": "not-a-digest"}
    with pytest.raises(ScenarioContractError, match="profile_sha256"):
        parse_scenario_contract(record)

    unrelated = _validation_edit("quant_rtn:4:32:all")
    unrelated["training_profile"] = binding
    with pytest.raises(
        ScenarioContractError, match="only valid for a training-profile"
    ):
        parse_scenario_contract(unrelated)


def _replace_generation(
    record: dict[str, object], **updates: object
) -> dict[str, object]:
    generation = record["generation"]
    assert isinstance(generation, dict)
    generation.update(updates)
    return record


def _replace_training_profile(
    record: dict[str, object], **updates: object
) -> dict[str, object]:
    profile = record["training_profile"]
    assert isinstance(profile, dict)
    profile.update(updates)
    return record


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({**_evidence_only(), "id": "report\n001"}, "canonical string"),
        (
            {**_evidence_only(), "artifact_class": "legacy_report"},
            "artifact_class.*unsupported",
        ),
        (
            _validation_edit("quant_rtn:4:32:all", version="trained"),
            "generation.version",
        ),
        (_validation_edit("magnitude_prune:0:all"), "positive decimal"),
        (_validation_edit("magnitude_prune:1:all"), "sparsity"),
        (_validation_edit("quant_rtn:4:32:"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@layers"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@foo=1"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@layers=01"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@layers=0"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@layers=2,layer=2"), "scope"),
        (_validation_edit("quant_rtn:4:32:all@layer=1,layers=2"), "scope"),
        (_validation_edit("lora_merge:clean", version="clean"), "clean edit_spec"),
        (_validation_edit("quant_rtn:4:32"), "four fields"),
        (_validation_edit("quant_rtn:9:32:all"), "bits must be"),
        (_validation_edit("magnitude_prune:0.5"), "three fields"),
        (_validation_edit("synthetic_lowrank_delta:8:64"), "four fields"),
        (_validation_edit("synthetic_dense_update:0.1:3"), "four fields"),
        (_validation_edit("lora_merge:4:8", version="trained"), "four fields"),
        (_validation_edit("fine_tune:0.1:2", version="trained"), "four fields"),
        (_deployable_bnb("bnb_8bit:8"), "three fields"),
        (_deployable_bnb("bnb_8bit:4:all"), "bits must be"),
        (_deployable_bnb("bnb_8bit:8:ffn"), "scope must be"),
        (_validation_edit(":"), "begin with an edit type"),
        (
            _validation_edit("bnb_8bit:8:all"),
            "requires generation.kind='deployable_edit'",
        ),
        (
            _deployable_bnb("quant_rtn:4:all"),
            "requires generation.kind='edit'",
        ),
        (
            _replace_generation(_error(), env={}),
            "generation.env must not be empty",
        ),
        (
            _replace_generation(_error(), error_type="legacy_fault"),
            "error_type.*unsupported",
        ),
        (
            _replace_generation(_error(), env_by_model={}),
            "env_by_model must not be empty",
        ),
        (
            {
                key: value
                for key, value in _deployable_bnb().items()
                if key != "optimized_deployment_backend"
            },
            "requires optimized_deployment_backend=true",
        ),
        (
            {
                **_validation_edit("quant_rtn:4:32:all"),
                "optimized_deployment_backend": True,
            },
            "non-deployable scenario",
        ),
        (
            _replace_training_profile(
                _validation_edit("lora_merge:4:8:attn"), profile_id="bad/profile"
            ),
            "profile_id is invalid",
        ),
        (
            _replace_training_profile(
                _validation_edit("lora_merge:4:8:attn"),
                snapshot_path="metadata/training_profiles/other.json",
            ),
            "canonical profile snapshot path",
        ),
        (
            _replace_training_profile(
                _validation_edit("lora_merge:4:8:attn"),
                snapshot_sha256="not-a-digest",
            ),
            "snapshot_sha256",
        ),
        (
            {**_error(), "training_profile": {}},
            "only valid for a training-profile edit",
        ),
        (
            {**_evidence_only(), "training_profile": {}},
            "only valid for a training-profile edit",
        ),
    ],
)
def test_dispatch_contract_rejects_ambiguous_boundary_values(
    record: dict[str, object], message: str
) -> None:
    with pytest.raises(ScenarioContractError, match=message):
        parse_scenario_contract(record)


def test_deployable_parser_rejects_internal_noncanonical_combinations() -> None:
    with pytest.raises(ScenarioContractError, match="only supports"):
        scenario_contract._deployable_edit(
            edit_type=scenario_contract.EditType.QUANT_RTN,
            parts=["quant_rtn", "4", "all"],
            version="deployable",
            backend="bitsandbytes",
        )
    with pytest.raises(ScenarioContractError, match="generation.version"):
        scenario_contract._deployable_edit(
            edit_type=scenario_contract.EditType.BNB_8BIT,
            parts=["bnb_8bit", "8", "all"],
            version="legacy",
            backend="bitsandbytes",
        )


def test_active_scenario_manifest_is_within_the_closed_contract() -> None:
    payload = json.loads(
        (REPO_ROOT / "scripts/evidence_packs/scenarios.json").read_text(
            encoding="utf-8"
        )
    )
    scenarios = payload["scenarios"]
    assert isinstance(scenarios, list)

    contracts = [parse_scenario_contract(scenario) for scenario in scenarios]

    assert {contract.generation_kind for contract in contracts} == {
        GenerationKind.EDIT,
        GenerationKind.DEPLOYABLE_EDIT,
        GenerationKind.ERROR,
    }
