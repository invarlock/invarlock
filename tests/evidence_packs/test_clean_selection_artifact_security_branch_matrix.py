from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock import clean_pruning_selection_artifacts as pruning
from invarlock import clean_pruning_selection_contract as pruning_contract
from invarlock import runtime_verify
from invarlock.clean_pruning_selection_common import (
    CleanPruningSelectionEvidenceError,
)
from invarlock.clean_selection import artifacts as selection
from invarlock.clean_selection.common import (
    CleanSelectionEvidenceError,
    canonical_json_sha256,
)
from invarlock.reporting import report_schema
from tests.evidence_packs._support_clean_pruning_selection import (
    _identity as pruning_identity,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _pruning,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _record as pruning_record,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _selection_config as pruning_selection_config,
)
from tests.evidence_packs._support_clean_selection import _record as selection_record


def _assert_unsafe_path_rejected(
    *,
    path_below: Callable[..., Path],
    error: type[Exception],
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-root"
    with pytest.raises(error, match="root"):
        path_below(missing_root, "evidence.json", label="evidence")

    root_file = tmp_path / "root-file"
    root_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(error, match="regular directory"):
        path_below(root_file, "evidence.json", label="evidence")

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(error, match="regular directory"):
        path_below(root_link, "evidence.json", label="evidence")

    with pytest.raises(error, match="missing"):
        path_below(real_root, "missing.json", label="evidence")

    payload = real_root / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    payload_link = real_root / "payload-link.json"
    payload_link.symlink_to(payload)
    with pytest.raises(error, match="symlink"):
        path_below(real_root, "payload-link.json", label="evidence")

    with pytest.raises(error, match="non-directory parent"):
        path_below(real_root, "payload.json/child.json", label="evidence")

    final_directory = real_root / "final-directory"
    final_directory.mkdir()
    with pytest.raises(error, match="regular file"):
        path_below(real_root, "final-directory", label="evidence")


def test_artifact_readers_reject_unsafe_paths_and_digest_substitution(
    tmp_path: Path,
) -> None:
    generic_root = tmp_path / "generic"
    generic_root.mkdir()
    pruning_root = tmp_path / "pruning"
    pruning_root.mkdir()
    _assert_unsafe_path_rejected(
        path_below=selection._path_below,
        error=CleanSelectionEvidenceError,
        tmp_path=generic_root,
    )
    _assert_unsafe_path_rejected(
        path_below=pruning._path_below,
        error=CleanPruningSelectionEvidenceError,
        tmp_path=pruning_root,
    )

    evidence_root = tmp_path / "digest-root"
    evidence_root.mkdir()
    snapshot = evidence_root / "snapshot.json"
    snapshot.write_text('{"authenticated":true}\n', encoding="utf-8")
    with pytest.raises(CleanPruningSelectionEvidenceError, match="digest mismatch"):
        pruning._read_referenced_json_snapshot(
            {"path": snapshot.name, "sha256": "sha256:" + "0" * 64},
            evidence_root=evidence_root,
            label="candidate replay",
        )


def _selection_context(tmp_path: Path) -> dict[str, object]:
    record = selection_record(tmp_path)
    candidate = record["candidates"][0]
    evaluation = candidate["evaluation"]
    report_run = evaluation["reports"][0]
    report_reference = report_run["report"]
    manifest_reference = report_run["runtime_manifest"]
    execution_reference = evaluation["execution"]
    report_path = tmp_path / report_reference["path"]
    manifest_path = tmp_path / manifest_reference["path"]
    return {
        "record": record,
        "candidate": candidate,
        "evaluation": evaluation,
        "report_reference": report_reference,
        "manifest_reference": manifest_reference,
        "report_bytes": report_path.read_bytes(),
        "report": json.loads(report_path.read_text(encoding="utf-8")),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "execution_sha256": execution_reference["sha256"],
    }


def _pruning_context(tmp_path: Path) -> dict[str, object]:
    record = pruning_record(tmp_path)
    candidate = record["candidates"][0]
    evaluation = candidate["evaluation"]
    report_run = evaluation["reports"][0]
    report_reference = report_run["report"]
    manifest_reference = report_run["runtime_manifest"]
    execution_reference = evaluation["execution"]
    report_path = tmp_path / report_reference["path"]
    manifest_path = tmp_path / manifest_reference["path"]
    execution_path = tmp_path / execution_reference["path"]
    return {
        "record": record,
        "candidate": candidate,
        "evaluation": evaluation,
        "report_reference": report_reference,
        "manifest_reference": manifest_reference,
        "report_bytes": report_path.read_bytes(),
        "report": json.loads(report_path.read_text(encoding="utf-8")),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "execution_receipt": json.loads(execution_path.read_text(encoding="utf-8")),
        "execution_sha256": execution_reference["sha256"],
    }


def test_generic_provenance_rejects_every_late_binding_substitution(
    tmp_path: Path,
) -> None:
    context = _selection_context(tmp_path)
    report = context["report"]
    record = context["record"]
    candidate = context["candidate"]
    assert isinstance(report, dict)
    assert isinstance(record, dict)
    assert isinstance(candidate, dict)
    config = record["selection_config"]
    baseline = record["baseline_identity"]
    transformation = candidate["transformation"]
    assert isinstance(config, dict)
    assert isinstance(baseline, dict)
    assert isinstance(transformation, dict)

    def native(value: dict[str, object]) -> dict[str, object]:
        provenance = value["provenance"]
        assert isinstance(provenance, dict)
        payload = provenance["clean_selection_execution"]
        assert isinstance(payload, dict)
        return payload

    def set_nested(
        value: dict[str, object], section: str, field: str, replacement: object
    ) -> None:
        payload = value[section]
        assert isinstance(payload, dict)
        payload[field] = replacement

    mutations = (
        lambda value: native(value).__setitem__(
            "execution_receipt_sha256", "sha256:" + "0" * 64
        ),
        lambda value: native(value).__setitem__(
            "selection_config_sha256", "sha256:" + "0" * 64
        ),
        lambda value: native(value).__setitem__("dataset", {}),
        lambda value: native(value).__setitem__("seed", 999),
        lambda value: set_nested(value, "meta", "seed", 999),
        lambda value: set_nested(value, "dataset", "dataset_name", "swapped"),
        lambda value: set_nested(value["dataset"], "hash", "source", "config_fallback"),  # type: ignore[arg-type]
        lambda value: set_nested(value["dataset"], "windows", "seed", 999),  # type: ignore[arg-type]
        lambda value: value["evaluation_windows"]["preview"].__setitem__(
            "window_ids", [1]
        ),  # type: ignore[index,union-attr]
        lambda value: native(value).__setitem__(
            "ordered_two_arm_schedule_sha256", "sha256:" + "0" * 64
        ),
    )
    for mutate in mutations:
        forged = copy.deepcopy(report)
        mutate(forged)
        with pytest.raises(CleanSelectionEvidenceError):
            selection._assert_report_native_execution_provenance(
                forged,
                execution_receipt_sha256=str(context["execution_sha256"]),
                selection_config=config,
                original_model_key="org/model",
                candidate_id=str(candidate["candidate_id"]),
                transformation=transformation,
                baseline_identity=baseline,
                repeat_index=0,
            )


def test_generic_report_binding_rejects_each_independent_substitution(
    tmp_path: Path,
) -> None:
    context = _selection_context(tmp_path)
    report = context["report"]
    record = context["record"]
    candidate = context["candidate"]
    report_reference = context["report_reference"]
    assert isinstance(report, dict)
    assert isinstance(record, dict)
    assert isinstance(candidate, dict)
    assert isinstance(report_reference, dict)
    config = record["selection_config"]
    baseline = record["baseline_identity"]
    transformation = candidate["transformation"]
    artifact = report_reference["artifact_identity"]
    assert isinstance(config, dict)
    assert isinstance(baseline, dict)
    assert isinstance(transformation, dict)
    assert isinstance(artifact, dict)
    config_sha256 = canonical_json_sha256(config)

    def assert_binding(value: dict[str, object]) -> float:
        return selection._assert_eligible_report(
            value,
            model_key="org/model",
            candidate_id=str(candidate["candidate_id"]),
            transformation=transformation,
            baseline_identity=baseline,
            artifact_identity=artifact,
            selection_config_sha256=config_sha256,
            execution_receipt_sha256=str(context["execution_sha256"]),
            selection_config=config,
            repeat_index=0,
        )

    assert assert_binding(report) == pytest.approx(0.01)
    mutations = (
        ("schema", "retired"),
        ("selection_config_sha256", "sha256:" + "0" * 64),
        ("execution_receipt_sha256", "sha256:" + "0" * 64),
        ("candidate_id", "other"),
        ("transformation", {**transformation, "scope": "all"}),
        ("artifact_identity", {**artifact, "sha256": "sha256:" + "0" * 64}),
        ("quality_loss", 0.5),
    )
    for field, replacement in mutations:
        forged = copy.deepcopy(report)
        binding = forged["clean_selection"]
        assert isinstance(binding, dict)
        binding[field] = replacement
        with pytest.raises(CleanSelectionEvidenceError):
            assert_binding(forged)


def test_artifact_quality_ratio_remains_fail_closed_without_schema_shortcut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic = _selection_context(tmp_path / "generic")
    pruning_context = _pruning_context(tmp_path / "pruning")
    monkeypatch.setattr(report_schema, "validate_report", lambda _value: True)

    for context, verifier, error, key in (
        (
            generic,
            selection._eligible_report_quality_loss,
            CleanSelectionEvidenceError,
            "model_key",
        ),
        (
            pruning_context,
            pruning._eligible_report_quality_loss,
            CleanPruningSelectionEvidenceError,
            "original_model_key",
        ),
    ):
        report = copy.deepcopy(context["report"])
        record = context["record"]
        report_reference = context["report_reference"]
        assert isinstance(report, dict)
        assert isinstance(record, dict)
        assert isinstance(report_reference, dict)
        primary_metric = report["primary_metric"]
        assert isinstance(primary_metric, dict)
        primary_metric["ratio_vs_baseline"] = 0
        kwargs = {
            key: "org/model",
            "baseline_identity": record["baseline_identity"],
            "artifact_identity": report_reference["artifact_identity"],
        }
        with pytest.raises(error, match="positive"):
            verifier(report, **kwargs)


def test_runtime_manifest_links_reject_network_and_subject_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_verify,
        "verify_runtime_manifest_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, errors=[]),
    )
    for context, verifier, error, binding_key, subject_key in (
        (
            _selection_context(tmp_path / "generic"),
            selection._assert_report_runtime_manifest,
            CleanSelectionEvidenceError,
            "clean_selection_execution",
            "transformation",
        ),
        (
            _pruning_context(tmp_path / "pruning"),
            pruning._assert_report_runtime_manifest,
            CleanPruningSelectionEvidenceError,
            "clean_pruning_selection_execution",
            "pruning",
        ),
    ):
        report = context["report"]
        record = context["record"]
        candidate = context["candidate"]
        report_reference = context["report_reference"]
        assert isinstance(report, dict)
        assert isinstance(record, dict)
        assert isinstance(candidate, dict)
        assert isinstance(report_reference, dict)
        config = record["selection_config"]
        baseline = record["baseline_identity"]
        assert isinstance(config, dict)
        assert isinstance(baseline, dict)
        kwargs = {
            "report_bytes": context["report_bytes"],
            "report": report,
            "report_reference": report_reference,
            "manifest_reference": context["manifest_reference"],
            "execution_receipt_sha256": context["execution_sha256"],
            "selection_config_sha256": canonical_json_sha256(config),
            "candidate_id": candidate["candidate_id"],
            subject_key: candidate[subject_key],
            "baseline_identity": baseline,
            "repeat_index": 0,
        }
        kwargs[
            "model_key" if subject_key == "transformation" else "original_model_key"
        ] = "org/model"

        forged_network = copy.deepcopy(context["manifest"])
        forged_network["runtime"]["allow_network"] = True
        with pytest.raises(error, match="allow_network"):
            verifier(manifest=forged_network, **kwargs)

        forged_link = copy.deepcopy(context["manifest"])
        forged_link["context"][binding_key]["candidate_id"] = "swapped"
        with pytest.raises(error, match="linkage mismatch"):
            verifier(manifest=forged_link, **kwargs)


def test_pruning_report_binding_and_evaluator_builder_reject_late_forgery(
    tmp_path: Path,
) -> None:
    context = _pruning_context(tmp_path)
    report = context["report"]
    record = context["record"]
    candidate = context["candidate"]
    report_reference = context["report_reference"]
    receipt = context["execution_receipt"]
    assert isinstance(report, dict)
    assert isinstance(record, dict)
    assert isinstance(candidate, dict)
    assert isinstance(report_reference, dict)
    assert isinstance(receipt, dict)
    config = record["selection_config"]
    baseline = record["baseline_identity"]
    pruning_spec = candidate["pruning"]
    artifact = report_reference["artifact_identity"]
    assert isinstance(config, dict)
    assert isinstance(baseline, dict)
    assert isinstance(pruning_spec, dict)
    assert isinstance(artifact, dict)

    def assert_binding(value: dict[str, object]) -> float:
        return pruning._assert_report_binding(
            value,
            original_model_key="org/model",
            candidate_id=str(candidate["candidate_id"]),
            pruning=pruning_spec,
            baseline_identity=baseline,
            artifact_identity=artifact,
            selection_config_sha256=canonical_json_sha256(config),
            execution_receipt_sha256=str(context["execution_sha256"]),
            repeat_index=0,
        )

    assert assert_binding(report) == pytest.approx(0.04)
    for field, replacement in (
        ("schema", "retired"),
        ("candidate_id", "swapped"),
        ("quality_loss", 0.5),
    ):
        forged = copy.deepcopy(report)
        binding = forged["clean_pruning_selection"]
        assert isinstance(binding, dict)
        binding[field] = replacement
        with pytest.raises(CleanPruningSelectionEvidenceError):
            assert_binding(forged)

    with pytest.raises(CleanPruningSelectionEvidenceError, match="repeat_index"):
        pruning.build_clean_pruning_evaluator_execution_provenance(
            report=report,
            execution_receipt=receipt,
            execution_receipt_sha256=str(context["execution_sha256"]),
            repeat_index=True,
        )
    short_schedule = copy.deepcopy(report)
    short_schedule["evaluation_windows"]["preview"]["window_ids"] = [1]
    with pytest.raises(CleanPruningSelectionEvidenceError, match="max_examples"):
        pruning.build_clean_pruning_evaluator_execution_provenance(
            report=short_schedule,
            execution_receipt=receipt,
            execution_receipt_sha256=str(context["execution_sha256"]),
            repeat_index=0,
        )
    for section, field, replacement in (
        ("meta", "seed", 999),
        ("dataset", "dataset_name", "swapped"),
    ):
        forged = copy.deepcopy(report)
        payload = forged[section]
        assert isinstance(payload, dict)
        payload[field] = replacement
        with pytest.raises(CleanPruningSelectionEvidenceError, match="immutable"):
            pruning.build_clean_pruning_evaluator_execution_provenance(
                report=forged,
                execution_receipt=receipt,
                execution_receipt_sha256=str(context["execution_sha256"]),
                repeat_index=0,
            )
    for section, field, replacement, message in (
        ("hash", "source", "config_fallback", "config_fallback"),
        ("windows", "seed", 999, "window seed"),
    ):
        forged = copy.deepcopy(report)
        dataset = forged["dataset"]
        assert isinstance(dataset, dict)
        payload = dataset[section]
        assert isinstance(payload, dict)
        payload[field] = replacement
        with pytest.raises(CleanPruningSelectionEvidenceError, match=message):
            pruning.build_clean_pruning_evaluator_execution_provenance(
                report=forged,
                execution_receipt=receipt,
                execution_receipt_sha256=str(context["execution_sha256"]),
                repeat_index=0,
            )


def test_pruning_candidate_binding_rejects_invalid_id_and_baseline_swap(
    tmp_path: Path,
) -> None:
    context = _pruning_context(tmp_path)
    record = context["record"]
    candidate = context["candidate"]
    evaluation = context["evaluation"]
    assert isinstance(record, dict)
    assert isinstance(candidate, dict)
    assert isinstance(evaluation, dict)
    replay_reference = evaluation["replay"]
    runtime_reference = evaluation["runtime"]
    replay = json.loads((tmp_path / replay_reference["path"]).read_text())
    runtime = json.loads((tmp_path / runtime_reference["path"]).read_text())
    kwargs = {
        "report": context["report"],
        "replay": replay,
        "runtime": runtime,
        "original_model_key": "org/model",
        "pruning": candidate["pruning"],
        "selection_config": record["selection_config"],
        "execution_receipt": context["execution_receipt"],
        "execution_receipt_sha256": context["execution_sha256"],
        "repeat_index": 0,
    }
    with pytest.raises(CleanPruningSelectionEvidenceError, match="candidate_id"):
        pruning.build_clean_pruning_candidate_report_binding(
            candidate_id="bad id", **kwargs
        )
    forged_replay = copy.deepcopy(replay)
    forged_replay["baseline_identity"] = {
        **forged_replay["baseline_identity"],
        "sha256": "sha256:" + "0" * 64,
    }
    with pytest.raises(CleanPruningSelectionEvidenceError, match="execution receipt"):
        pruning.build_clean_pruning_candidate_report_binding(
            candidate_id=str(candidate["candidate_id"]),
            **{**kwargs, "replay": forged_replay},
        )


PruningMutation = Callable[[dict[str, object]], None]


def _selected_pruning_entry(entry: dict[str, object]) -> dict[str, object]:
    selected = entry["selected_entry"]
    assert isinstance(selected, dict)
    return selected


def _pruning_receipt(entry: dict[str, object]) -> dict[str, object]:
    receipt = _selected_pruning_entry(entry)["selection_receipt"]
    assert isinstance(receipt, dict)
    return receipt


def test_pruning_selected_entry_rejects_forged_receipt_and_outer_claims(
    tmp_path: Path,
) -> None:
    valid = pruning_contract.select_clean_pruning(pruning_record(tmp_path))
    assert pruning_contract._selected_entry(valid) == valid

    def mutate_receipt(field: str, value: object) -> PruningMutation:
        return lambda entry: _pruning_receipt(entry).__setitem__(field, value)

    cases: list[PruningMutation] = [
        mutate_receipt("schema", "retired"),
        mutate_receipt("contract_version", "retired"),
        mutate_receipt("selection_config_sha256", "sha256:" + "0" * 64),
        mutate_receipt("decision_rule_sha256", "sha256:" + "0" * 64),
        mutate_receipt("selected_candidate_id", "wrong"),
        mutate_receipt("selected_pruning", {}),
        mutate_receipt("selected_evaluation", {}),
        lambda entry: entry.__setitem__("schema", "retired"),
        lambda entry: entry.__setitem__("contract_version", "retired"),
        lambda entry: _selected_pruning_entry(entry).__setitem__("status", "claimed"),
        lambda entry: _selected_pruning_entry(entry).__setitem__(
            "selection_receipt_sha256", "sha256:" + "0" * 64
        ),
        lambda entry: _selected_pruning_entry(entry).__setitem__("scope", "forged"),
        lambda entry: entry.__setitem__("original_model_key", "other/model"),
    ]
    for index, mutate in enumerate(cases):
        forged = copy.deepcopy(valid)
        mutate(forged)
        try:
            pruning_contract._selected_entry(forged)
        except CleanPruningSelectionEvidenceError:
            pass
        else:
            raise AssertionError(f"pruning forgery case {index} was accepted")


def test_pruning_bundle_rejects_schema_empty_duplicate_and_stale_digest(
    tmp_path: Path,
) -> None:
    selected = pruning_contract.select_clean_pruning(pruning_record(tmp_path))
    entries = [selected]
    valid = {
        "schema": pruning_contract.CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA,
        "contract_version": pruning_contract.CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "entries": entries,
        "bundle_sha256": pruning_contract.canonical_clean_pruning_bundle_sha256(
            entries
        ),
    }
    assert pruning_contract.verify_clean_pruning_selection_bundle(valid) == valid
    for mutate in (
        lambda bundle: bundle.__setitem__("schema", "retired"),
        lambda bundle: bundle.__setitem__("contract_version", "retired"),
        lambda bundle: bundle.__setitem__("entries", []),
        lambda bundle: bundle.__setitem__("entries", [selected, selected]),
        lambda bundle: bundle.__setitem__("bundle_sha256", "sha256:" + "0" * 64),
    ):
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning_contract.verify_clean_pruning_selection_bundle(forged)


def test_pruning_execution_receipt_rejects_schema_phase_digest_and_expectations() -> (
    None
):
    config = pruning_selection_config()
    pruning_spec = _pruning("ffn", 0.5)
    baseline = pruning_identity("a")
    receipt = pruning_contract.build_clean_pruning_execution_receipt(
        original_model_key="org/model",
        candidate_id="candidate",
        pruning=pruning_spec,
        baseline_identity=baseline,
        selection_config=config,
    )
    assert pruning_contract.validate_clean_pruning_execution_receipt(receipt) == receipt

    for field, value in (
        ("schema", "retired"),
        ("contract_version", "retired"),
        ("phase", "after_evaluation"),
        ("candidate_id", "bad id"),
        ("selection_config_sha256", "sha256:" + "0" * 64),
    ):
        forged = copy.deepcopy(receipt)
        forged[field] = value
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning_contract.validate_clean_pruning_execution_receipt(forged)

    expectation_cases = (
        {"expected_model_key": "other/model"},
        {"expected_candidate_id": "other"},
        {"expected_pruning": {**pruning_spec, "scope": "attn"}},
        {"expected_baseline_identity": pruning_identity("b")},
        {"expected_selection_config": {**config, "seed": 999}},
    )
    for kwargs in expectation_cases:
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning_contract.validate_clean_pruning_execution_receipt(receipt, **kwargs)


def test_pruning_execution_receipt_builder_rejects_invalid_candidate_id() -> None:
    assert (
        pruning_contract._no_bare_selected_by({"nested": ["selection_operator"]})
        is None
    )

    with pytest.raises(CleanPruningSelectionEvidenceError, match="candidate_id"):
        pruning_contract.build_clean_pruning_execution_receipt(
            original_model_key="org/model",
            candidate_id="bad id",
            pruning=_pruning("ffn", 0.5),
            baseline_identity=pruning_identity("a"),
            selection_config=pruning_selection_config(),
        )

    with pytest.raises(CleanPruningSelectionEvidenceError, match="selected_by"):
        pruning_contract._no_bare_selected_by({"nested": ["selected_by_operator"]})
