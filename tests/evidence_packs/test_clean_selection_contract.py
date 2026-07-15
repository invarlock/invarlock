from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.evidence_packs.python.editing.clean_selection_contract import (
    CANDIDATE_EVALUATION_SCHEMA,
    CANDIDATE_RECORD_SCHEMA,
    CLEAN_SELECTION_BUNDLE_SCHEMA,
    CLEAN_SELECTION_CONTRACT_VERSION,
    DECISION_RULE_SCHEMA,
    EVALUATION_SCHEDULE_SCHEMA,
    SELECTED_ENTRY_SCHEMA,
    SELECTION_CONFIG_SCHEMA,
    SELECTION_RECEIPT_SCHEMA,
    CleanSelectionContractError,
    canonical_candidate_set_sha256,
    canonical_sha256,
    load_candidate_record,
    select_clean_transformation,
    verify_selected_entry,
)
from scripts.evidence_packs.python.editing.create_clean_selection_receipt import (
    main as create_clean_selection_receipt,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _identity(character: str) -> dict[str, str]:
    return {"kind": "local_checkpoint_tree", "sha256": _digest(character)}


def _evaluation(
    *,
    artifact: dict[str, str],
    baseline: dict[str, str],
    selection_config_sha256: str,
    digest_characters: tuple[str, str, str, str, str, str, str],
    quality_loss: float,
) -> dict[str, object]:
    return {
        "schema": CANDIDATE_EVALUATION_SCHEMA,
        "selection_config_sha256": selection_config_sha256,
        "execution": {
            "path": f"candidates/execution-{digest_characters[0]}.json",
            "sha256": _digest(digest_characters[0]),
        },
        "reports": [
            {
                "report": {
                    "path": f"candidates/report-{digest_characters[1]}.json",
                    "sha256": _digest(digest_characters[1]),
                    "artifact_identity": artifact,
                    "baseline_identity": baseline,
                },
                "runtime_manifest": {
                    "path": f"candidates/manifest-{digest_characters[2]}.json",
                    "sha256": _digest(digest_characters[2]),
                },
            },
            {
                "report": {
                    "path": f"candidates/report-{digest_characters[3]}.json",
                    "sha256": _digest(digest_characters[3]),
                    "artifact_identity": artifact,
                    "baseline_identity": baseline,
                },
                "runtime_manifest": {
                    "path": f"candidates/manifest-{digest_characters[4]}.json",
                    "sha256": _digest(digest_characters[4]),
                },
            },
        ],
        "replay": {
            "path": f"candidates/replay-{digest_characters[5]}.json",
            "sha256": _digest(digest_characters[5]),
            "artifact_identity": artifact,
            "baseline_identity": baseline,
        },
        "runtime": {
            "path": f"candidates/runtime-{digest_characters[6]}.json",
            "sha256": _digest(digest_characters[6]),
            "artifact_identity": artifact,
            "replay_artifact_identity": artifact,
            "baseline_identity": baseline,
        },
        "metrics": {"quality_loss": quality_loss},
    }


def _candidate_record() -> dict[str, object]:
    baseline = _identity("b")
    selection_config = {
        "schema": SELECTION_CONFIG_SCHEMA,
        "dataset": {
            "name": "org/frozen-eval-set",
            "revision": "a" * 40,
            "split": "validation",
            "content_sha256": _digest("c"),
        },
        "seed": 17,
        "schedule": {
            "schema": EVALUATION_SCHEDULE_SCHEMA,
            "candidate_order": "candidate_id_ascending",
            "evaluation_repeats": 2,
            "max_examples": 64,
            "batch_size": 4,
            "shuffle": False,
        },
    }
    selection_config_sha256 = canonical_sha256(selection_config)
    record: dict[str, object] = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": "org/model",
        "baseline_identity": baseline,
        "selection_domain": {
            "edit_type": "quant_rtn",
            "scope_policy": "architecture-aware-transformation-v1",
        },
        "selection_config": selection_config,
        "decision_rule": {
            "schema": DECISION_RULE_SCHEMA,
            "kind": "lexicographic_metrics_v1",
            "metric_order": ["quality_loss"],
            "tie_breaker": "candidate_id_ascending",
        },
        "candidates": [
            {
                "candidate_id": "quant4",
                "transformation": {
                    "edit_type": "quant_rtn",
                    "parameters": {"bits": 4, "group_size": 32},
                    "scope": "ffn",
                },
                "evaluation": _evaluation(
                    artifact=_identity("d"),
                    baseline=baseline,
                    selection_config_sha256=selection_config_sha256,
                    digest_characters=("1", "2", "3", "4", "5", "6", "7"),
                    quality_loss=0.01,
                ),
            },
            {
                "candidate_id": "quant8",
                "transformation": {
                    "edit_type": "quant_rtn",
                    "parameters": {"bits": 8, "group_size": 32},
                    "scope": "ffn",
                },
                "evaluation": _evaluation(
                    artifact=_identity("f"),
                    baseline=baseline,
                    selection_config_sha256=selection_config_sha256,
                    digest_characters=("8", "9", "a", "b", "c", "d", "e"),
                    quality_loss=0.02,
                ),
            },
        ],
    }
    record["candidate_set_sha256"] = canonical_candidate_set_sha256(record)
    return record


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_clean_selection_protocol_has_one_repaired_v1_generation() -> None:
    assert CLEAN_SELECTION_CONTRACT_VERSION == "clean-transformation-selection-v1"
    assert (
        CANDIDATE_RECORD_SCHEMA == "invarlock/clean-transformation-candidate-record-v1"
    )
    assert (
        CANDIDATE_EVALUATION_SCHEMA
        == "invarlock/clean-transformation-candidate-evaluation-v1"
    )
    assert (
        SELECTION_RECEIPT_SCHEMA
        == "invarlock/clean-transformation-selection-receipt-v1"
    )
    assert SELECTED_ENTRY_SCHEMA == "invarlock/clean-transformation-selected-entry-v1"
    assert (
        CLEAN_SELECTION_BUNDLE_SCHEMA
        == "invarlock/clean-transformation-selection-bundle-v1"
    )


def test_selects_real_candidate_from_bound_evaluation_receipts() -> None:
    record = _candidate_record()

    selected = select_clean_transformation(record)

    assert selected["schema"] == SELECTED_ENTRY_SCHEMA
    assert selected["original_model_key"] == "org/model"
    entry = selected["selected_entry"]
    assert isinstance(entry, dict)
    assert entry["status"] == "selected"
    assert entry["parameters"] == {"bits": 4, "group_size": 32}
    receipt = entry["selection_receipt"]
    assert isinstance(receipt, dict)
    assert receipt["selected_candidate_id"] == "quant4"
    assert receipt["baseline_identity"] == _identity("b")
    assert receipt["selection_config"]["dataset"] == {
        "name": "org/frozen-eval-set",
        "revision": "a" * 40,
        "split": "validation",
        "content_sha256": _digest("c"),
    }
    assert receipt["candidate_set_sha256"] == record["candidate_set_sha256"]
    assert entry["selection_receipt_sha256"] == canonical_sha256(receipt)
    assert verify_selected_entry(selected) == selected


def test_candidate_set_digest_binds_every_candidate_and_immutable_inputs() -> None:
    record = _candidate_record()
    record["selection_config"]["seed"] = 18  # type: ignore[index]

    with pytest.raises(CleanSelectionContractError, match="selection_config_sha256"):
        select_clean_transformation(record)

    record = _candidate_record()
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    candidates[1]["evaluation"]["metrics"]["quality_loss"] = 0.005

    with pytest.raises(CleanSelectionContractError, match="candidate_set_sha256"):
        select_clean_transformation(record)


def test_rejects_bare_claims_missing_receipts_and_arbitrary_overrides() -> None:
    record = _candidate_record()
    record["selected_by_operator"] = "anything"
    with pytest.raises(CleanSelectionContractError, match="bare selected_by claim"):
        select_clean_transformation(record)

    record = _candidate_record()
    candidate = record["candidates"][0]  # type: ignore[index]
    candidate["evaluation"].pop("runtime")
    with pytest.raises(CleanSelectionContractError, match="unbound, missing"):
        select_clean_transformation(record)

    record = _candidate_record()
    record["selected_candidate_id"] = "quant8"
    with pytest.raises(CleanSelectionContractError, match="arbitrary fields"):
        select_clean_transformation(record)


def test_rejects_report_replay_runtime_identity_mismatch() -> None:
    record = _candidate_record()
    candidate = record["candidates"][0]  # type: ignore[index]
    candidate["evaluation"]["replay"]["artifact_identity"] = _identity("f")

    with pytest.raises(CleanSelectionContractError, match="artifact_identity mismatch"):
        select_clean_transformation(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["candidates"][0].update(candidate_id="bad id"),
            "candidate_id is invalid",
        ),
        (
            lambda record: record["candidates"][0]["evaluation"].update(reports={}),
            "reports must be a list",
        ),
        (
            lambda record: record["candidates"][0]["evaluation"]["reports"][1][
                "report"
            ].update(
                path=record["candidates"][0]["evaluation"]["reports"][0]["report"][
                    "path"
                ]
            ),
            "distinct report and runtime-manifest paths",
        ),
        (
            lambda record: record["candidates"][0]["evaluation"]["reports"][1][
                "report"
            ].update(artifact_identity=_identity("9")),
            "artifact identities must match",
        ),
        (
            lambda record: record["selection_domain"].update(scope_policy="arbitrary"),
            "scope_policy is unsupported",
        ),
        (
            lambda record: record.update(candidates=record["candidates"][:1]),
            "at least two candidates",
        ),
        (
            lambda record: record["candidates"].reverse(),
            "sorted and unique",
        ),
        (
            lambda record: record["candidates"][1].update(
                transformation=copy.deepcopy(record["candidates"][0]["transformation"])
            ),
            "duplicate canonical transformations",
        ),
        (
            lambda record: record.update(candidate_set_sha256=_digest("0")),
            "candidate_set_sha256",
        ),
    ],
)
def test_candidate_record_rejects_ambiguous_or_rebound_selection_evidence(
    mutate: object, message: str
) -> None:
    record = _candidate_record()
    mutate(record)  # type: ignore[operator]

    with pytest.raises(CleanSelectionContractError, match=message):
        select_clean_transformation(record)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("selection_domain", "edit_type"),
            "noop",
            "unsupported",
        ),
        (
            ("selection_domain", "edit_type"),
            "fp8_quant",
            "unsupported",
        ),
        (
            ("candidates", 0, "transformation", "scope"),
            "FFN",
            "canonical scope",
        ),
        (
            ("candidates", 0, "transformation", "parameters", "bits"),
            4.0,
            "positive integer",
        ),
    ],
)
def test_rejects_unsupported_or_noncanonical_transformations(
    path: tuple[object, ...], value: object, message: str
) -> None:
    record = _candidate_record()
    target: object = record
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(CleanSelectionContractError, match=message):
        select_clean_transformation(record)


def test_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = tmp_path / "candidate-record.json"
    raw.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")

    with pytest.raises(CleanSelectionContractError, match="duplicate key"):
        load_candidate_record(raw)


def test_verify_rejects_forged_or_copied_receipts() -> None:
    first = select_clean_transformation(_candidate_record())
    second_record = _candidate_record()
    second_record["original_model_key"] = "other/model"
    second_record["candidate_set_sha256"] = canonical_candidate_set_sha256(
        {
            key: value
            for key, value in second_record.items()
            if key != "candidate_set_sha256"
        }
    )
    second = select_clean_transformation(second_record)
    forged = copy.deepcopy(first)
    forged_entry = forged["selected_entry"]
    second_entry = second["selected_entry"]
    assert isinstance(forged_entry, dict)
    assert isinstance(second_entry, dict)
    forged_entry["selection_receipt"] = second_entry["selection_receipt"]
    forged_entry["selection_receipt_sha256"] = canonical_sha256(
        forged_entry["selection_receipt"]
    )

    with pytest.raises(
        CleanSelectionContractError, match="original_model_key mismatch"
    ):
        verify_selected_entry(forged)


def test_cli_writes_idempotently_and_refuses_override(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate-record.json"
    output_path = tmp_path / "selected-entry.json"
    record = _candidate_record()
    _write_json(candidate_path, record)

    assert (
        create_clean_selection_receipt(
            ["--candidate-record", str(candidate_path), "--out", str(output_path)]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert verify_selected_entry(payload) == payload
    assert (
        create_clean_selection_receipt(
            ["--candidate-record", str(candidate_path), "--out", str(output_path)]
        )
        == 0
    )

    changed = _candidate_record()
    candidates = changed["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["evaluation"]["metrics"]["quality_loss"] = 0.05
    changed["candidate_set_sha256"] = canonical_candidate_set_sha256(
        {key: value for key, value in changed.items() if key != "candidate_set_sha256"}
    )
    _write_json(candidate_path, changed)
    with pytest.raises(CleanSelectionContractError, match="refusing to overwrite"):
        create_clean_selection_receipt(
            ["--candidate-record", str(candidate_path), "--out", str(output_path)]
        )
