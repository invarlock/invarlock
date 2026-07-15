from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from invarlock import clean_pruning_selection_common as pruning
from invarlock.clean_selection import bundle as bundle_contract
from invarlock.clean_selection import common as selection
from invarlock.clean_selection.common import CleanSelectionEvidenceError
from tests.evidence_packs._support_clean_selection import _record


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.parametrize(
    ("module", "error", "invoke"),
    [
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection.canonical_json_sha256(object()),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._mapping([], label="value"),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._exact_mapping(
                {}, label="value", fields=frozenset({"x"})
            ),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._text(" bad ", label="value"),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._sha256("bad", label="value"),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._identity(
                {"kind": "remote_revision", "sha256": _digest("x")}, label="value"
            ),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._finite(True, label="value"),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._finite(float("nan"), label="value"),
        ),
        (
            selection,
            selection.CleanSelectionEvidenceError,
            lambda: selection._positive_int(0, label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning.canonical_json_sha256(object()),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._mapping([], label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._exact_mapping({}, label="value", fields=frozenset({"x"})),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._text(" bad ", label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._sha256("bad", label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._identity(
                {"kind": "remote_revision", "sha256": _digest("x")}, label="value"
            ),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._positive_int(0, label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._nonnegative_int(-1, label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._finite(True, label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._finite(float("inf"), label="value"),
        ),
        (
            pruning,
            pruning.CleanPruningSelectionEvidenceError,
            lambda: pruning._scope("embed", label="value"),
        ),
    ],
)
def test_common_scalar_validators_reject_ambiguous_values(
    module: object,
    error: type[ValueError],
    invoke: Callable[[], object],
) -> None:
    del module
    with pytest.raises(error):
        invoke()


@pytest.mark.parametrize(
    "scope",
    [
        None,
        "",
        "ffn@layers=2@layer=1",
        "embed",
        "ffn@",
        "ffn@layers",
        "ffn@depth=2",
        "ffn@layers=02",
        "ffn@layers=0",
        "ffn@layers=2,layer=2",
        "FFN",
        "ffn@layer=1,layers=2",
    ],
)
def test_clean_selection_scope_rejects_noncanonical_grammar(scope: object) -> None:
    with pytest.raises(selection.CleanSelectionEvidenceError):
        selection._scope(scope, label="transform.scope")


@pytest.mark.parametrize(
    "value",
    [
        {"edit_type": "unknown", "parameters": {}, "scope": "all"},
        {
            "edit_type": "quant_rtn",
            "parameters": {"bits": 1, "group_size": 32},
            "scope": "all",
        },
        {
            "edit_type": "quant_rtn",
            "parameters": {"bits": 4.0, "group_size": 32},
            "scope": "all",
        },
        {
            "edit_type": "synthetic_lowrank_delta",
            "parameters": {"rank": 2, "scale": 0},
            "scope": "all",
        },
        {
            "edit_type": "synthetic_lowrank_delta",
            "parameters": {"rank": 999999, "scale": 1},
            "scope": "all",
        },
        {
            "edit_type": "synthetic_dense_update",
            "parameters": {"step_size": 0, "iterations": 1},
            "scope": "all",
        },
        {
            "edit_type": "synthetic_dense_update",
            "parameters": {"step_size": 0.1, "iterations": 999999},
            "scope": "all",
        },
    ],
)
def test_clean_transform_rejects_unsupported_or_noncanonical_parameters(
    value: dict[str, object],
) -> None:
    with pytest.raises(selection.CleanSelectionEvidenceError):
        selection._transform(value, label="transform")


def _selection_config() -> dict[str, object]:
    return {
        "schema": selection.SELECTION_CONFIG_SCHEMA,
        "dataset": {
            "name": "fixture",
            "revision": "a" * 40,
            "split": "validation",
            "content_sha256": _digest("dataset"),
        },
        "seed": 1,
        "schedule": {
            "schema": selection.EVALUATION_SCHEDULE_SCHEMA,
            "candidate_order": "candidate_id_ascending",
            "evaluation_repeats": 1,
            "max_examples": 2,
            "batch_size": 1,
            "shuffle": False,
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "retired"),
        (("dataset", "revision"), "main"),
        (("seed",), -1),
        (("schedule", "schema"), "retired"),
        (("schedule", "candidate_order"), "random"),
        (("schedule", "evaluation_repeats"), 0),
        (("schedule", "shuffle"), "false"),
    ],
)
def test_clean_selection_config_rejects_unbound_execution_policy(
    path: tuple[str, ...], value: object
) -> None:
    payload = _selection_config()
    target = payload
    for part in path[:-1]:
        child = target[part]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value
    with pytest.raises(selection.CleanSelectionEvidenceError):
        selection._selection_config(payload)


def _pruning_config() -> dict[str, object]:
    return {
        "schema": pruning.CLEAN_PRUNING_SELECTION_CONFIG_SCHEMA,
        "dataset": {
            "name": "fixture",
            "revision": "a" * 40,
            "split": "validation",
            "content_sha256": _digest("dataset"),
        },
        "seed": 1,
        "schedule": {
            "schema": pruning.CLEAN_PRUNING_EVALUATION_SCHEDULE_SCHEMA,
            "candidate_order": "candidate_id_ascending",
            "evaluation_repeats": 1,
            "max_examples": 2,
            "batch_size": 1,
            "shuffle": False,
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "retired"),
        (("dataset", "revision"), "main"),
        (("seed",), -1),
        (("schedule", "schema"), "retired"),
        (("schedule", "candidate_order"), "random"),
        (("schedule", "shuffle"), True),
        (("schedule", "batch_size"), 0),
    ],
)
def test_clean_pruning_config_rejects_unbound_execution_policy(
    path: tuple[str, ...], value: object
) -> None:
    payload = _pruning_config()
    target = payload
    for part in path[:-1]:
        child = target[part]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value
    with pytest.raises(pruning.CleanPruningSelectionEvidenceError):
        pruning._selection_config(payload)


def test_clean_pruning_spec_and_fixed_rules_reject_forged_values() -> None:
    with pytest.raises(pruning.CleanPruningSelectionEvidenceError, match="edit_type"):
        pruning._pruning_spec(
            {"edit_type": "quant_rtn", "scope": "all", "target_sparsity": 0.5},
            label="candidate",
        )
    with pytest.raises(
        pruning.CleanPruningSelectionEvidenceError, match=r"in \(0, 1\)"
    ):
        pruning._pruning_spec(
            {"edit_type": "magnitude_prune", "scope": "all", "target_sparsity": 1},
            label="candidate",
        )
    with pytest.raises(
        pruning.CleanPruningSelectionEvidenceError, match="decision_rule"
    ):
        pruning._decision_rule(
            {
                "schema": pruning.CLEAN_PRUNING_DECISION_RULE_SCHEMA,
                "metric": "claimed_quality",
                "direction": "minimize",
                "tie_breaker": "candidate_id_ascending",
            }
        )
    with pytest.raises(
        pruning.CleanPruningSelectionEvidenceError, match="selection_domain"
    ):
        pruning._selection_domain(
            {
                "edit_type": "magnitude_prune",
                "scope_policy": "forged",
                "pruning_algorithm": pruning.PRUNING_ALGORITHM,
                "storage_policy": pruning.PRUNING_STORAGE_POLICY,
                "target_manifest_schema": pruning.PRUNING_TARGET_MANIFEST_SCHEMA,
            }
        )


@pytest.mark.parametrize(
    "path", ["/absolute.json", "../escape.json", "bad.txt", "a//b.json"]
)
def test_clean_selection_reference_paths_must_remain_inside_evidence_root(
    path: str,
) -> None:
    with pytest.raises(selection.CleanSelectionEvidenceError):
        selection._safe_relative_json_path(path, label="reference")
    with pytest.raises(pruning.CleanPruningSelectionEvidenceError):
        pruning._safe_relative_json_path(path, label="reference")


def test_bare_selected_by_claims_are_rejected_at_any_depth() -> None:
    with pytest.raises(selection.CleanSelectionEvidenceError, match="selected_by"):
        selection._no_bare_selected_by({"nested": ["selected_by_claim"]})


def test_clean_pruning_reference_rejects_identity_substitution() -> None:
    baseline = {"kind": "local_checkpoint_tree", "sha256": _digest("baseline")}
    artifact = {"kind": "local_checkpoint_tree", "sha256": _digest("artifact")}
    replay = {"kind": "local_checkpoint_tree", "sha256": _digest("replay")}
    reference = {
        "path": "reports/replay.json",
        "sha256": _digest("sidecar"),
        "artifact_identity": artifact,
        "baseline_identity": baseline,
        "replay_artifact_identity": replay,
    }
    with pytest.raises(
        pruning.CleanPruningSelectionEvidenceError, match="baseline_identity"
    ):
        pruning._bound_reference(
            reference,
            label="reference",
            baseline_identity={**baseline, "sha256": _digest("wrong baseline")},
            artifact_identity=artifact,
            replay_identity=replay,
        )
    with pytest.raises(
        pruning.CleanPruningSelectionEvidenceError, match="artifact_identity"
    ):
        pruning._bound_reference(
            reference,
            label="reference",
            baseline_identity=baseline,
            artifact_identity={**artifact, "sha256": _digest("wrong artifact")},
            replay_identity=replay,
        )
    with pytest.raises(
        pruning.CleanPruningSelectionEvidenceError,
        match="replay_artifact_identity",
    ):
        pruning._bound_reference(
            reference,
            label="reference",
            baseline_identity=baseline,
            artifact_identity=artifact,
            replay_identity={**replay, "sha256": _digest("wrong replay")},
        )


Mutation = Callable[[dict[str, object]], None]


def _receipt(entry: dict[str, object]) -> dict[str, object]:
    selected = entry["selected_entry"]
    assert isinstance(selected, dict)
    receipt = selected["selection_receipt"]
    assert isinstance(receipt, dict)
    return receipt


def _selected(entry: dict[str, object]) -> dict[str, object]:
    selected = entry["selected_entry"]
    assert isinstance(selected, dict)
    return selected


def test_selected_entry_rejects_each_forged_receipt_and_outer_claim(
    tmp_path: Path,
) -> None:
    valid = bundle_contract.select_clean_transformation(_record(tmp_path))

    def mutate_receipt(field: str, value: object) -> Mutation:
        return lambda entry: _receipt(entry).__setitem__(field, value)

    def reverse_candidates(entry: dict[str, object]) -> None:
        candidates = _receipt(entry)["candidates"]
        assert isinstance(candidates, list)
        candidates.reverse()

    def duplicate_transformation(entry: dict[str, object]) -> None:
        candidates = _receipt(entry)["candidates"]
        assert isinstance(candidates, list)
        first = candidates[0]
        second = candidates[1]
        assert isinstance(first, dict) and isinstance(second, dict)
        second["transformation"] = copy.deepcopy(first["transformation"])

    receipt = _receipt(valid)
    candidates = receipt["candidates"]
    assert isinstance(candidates, list)
    winner = candidates[0]
    assert isinstance(winner, dict)
    cases: list[Mutation] = [
        mutate_receipt("schema", "retired"),
        mutate_receipt("contract_version", "retired"),
        lambda entry: _receipt(entry)["selection_domain"].__setitem__(
            "edit_type", "unknown"
        ),  # type: ignore[union-attr]
        lambda entry: _receipt(entry)["selection_domain"].__setitem__(
            "scope_policy", "forged"
        ),  # type: ignore[union-attr]
        mutate_receipt("selection_config_sha256", "sha256:" + "0" * 64),
        mutate_receipt("decision_rule_sha256", "sha256:" + "0" * 64),
        reverse_candidates,
        duplicate_transformation,
        mutate_receipt("candidate_set_sha256", "sha256:" + "0" * 64),
        mutate_receipt("selected_candidate_id", "wrong"),
        mutate_receipt("selected_transformation", {}),
        mutate_receipt("selected_evaluation", {}),
        lambda entry: entry.__setitem__("contract_version", "retired"),
        lambda entry: _selected(entry).__setitem__("status", "claimed"),
        lambda entry: _selected(entry).__setitem__(
            "selection_receipt_sha256", "sha256:" + "0" * 64
        ),
        lambda entry: _selected(entry).__setitem__("scope", "forged"),
        lambda entry: entry.__setitem__("original_model_key", "other/model"),
    ]
    for index, mutate in enumerate(cases):
        forged = copy.deepcopy(valid)
        mutate(forged)
        try:
            bundle_contract.verify_selected_entry(forged)
        except CleanSelectionEvidenceError:
            pass
        else:
            raise AssertionError(f"forgery case {index} was accepted")


def test_selection_bundle_rejects_schema_empty_unsorted_and_duplicate_entries(
    tmp_path: Path,
) -> None:
    selected = bundle_contract.select_clean_transformation(_record(tmp_path))
    valid = {
        "schema": bundle_contract.CLEAN_SELECTION_BUNDLE_SCHEMA,
        "contract_version": bundle_contract.CLEAN_SELECTION_CONTRACT_VERSION,
        "entries": [selected],
    }
    assert bundle_contract.verify_selection_bundle(valid) == valid
    for mutate in (
        lambda bundle: bundle.__setitem__("schema", "retired"),
        lambda bundle: bundle.__setitem__("contract_version", "retired"),
        lambda bundle: bundle.__setitem__("entries", []),
        lambda bundle: bundle.__setitem__("entries", [selected, selected]),
    ):
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanSelectionEvidenceError):
            bundle_contract.verify_selection_bundle(forged)
