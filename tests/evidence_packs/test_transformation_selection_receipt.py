from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.clean_selection.common import canonical_json_sha256
from scripts.evidence_packs.python.editing.attach_transformation_selection_receipt import (
    TRANSFORMATION_SELECTION_RECEIPT_SCHEMA,
    SelectionReceiptError,
    attach_transformation_selection_receipt,
)
from tests.evidence_packs._support_clean_selection import _bundle, _record


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _selected_bundle(root: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    record = _record(root)
    bundle_path, _ = _bundle(root, record)
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    selected_candidate = candidates[0]
    assert isinstance(selected_candidate, dict)
    evaluation = selected_candidate["evaluation"]
    assert isinstance(evaluation, dict)
    replay = root / str(evaluation["replay"]["path"])
    return bundle_path, json.loads(replay.read_text()), evaluation


def test_attach_receipt_binds_final_replay_to_exact_v1_candidate(
    tmp_path: Path,
) -> None:
    bundle_path, replay, _ = _selected_bundle(tmp_path)
    final_replay = tmp_path / "final_replay.json"
    _write(final_replay, replay)

    receipt = attach_transformation_selection_receipt(
        replay_path=final_replay,
        selection_bundle_path=bundle_path,
        scenario_id="clean_quant_rtn",
        model_key="org/model",
        edit_type="quant_rtn",
        parameters={"bits": 8, "group_size": 32},
        scope="attn",
    )

    payload = json.loads(final_replay.read_text())
    assert receipt == payload["selection_receipt"]
    assert receipt["schema"] == TRANSFORMATION_SELECTION_RECEIPT_SCHEMA
    assert receipt["selected_candidate_id"] == "attn8"
    assert (
        receipt["selection_bundle_path"]
        == "metadata/clean_selection/selection_bundle.json"
    )
    assert payload["selection_receipt_sha256"] == canonical_json_sha256(receipt)
    assert (
        attach_transformation_selection_receipt(
            replay_path=final_replay,
            selection_bundle_path=bundle_path,
            scenario_id="clean_quant_rtn",
            model_key="org/model",
            edit_type="quant_rtn",
            parameters={"bits": 8, "group_size": 32},
            scope="attn",
        )
        == receipt
    )


def test_attach_receipt_rejects_copied_or_wrong_final_artifact(tmp_path: Path) -> None:
    bundle_path, replay, _ = _selected_bundle(tmp_path)
    replay["artifact_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "f" * 64,
    }
    final_replay = tmp_path / "final_replay.json"
    _write(final_replay, replay)

    with pytest.raises(SelectionReceiptError, match="artifact identity"):
        attach_transformation_selection_receipt(
            replay_path=final_replay,
            selection_bundle_path=bundle_path,
            scenario_id="clean_quant_rtn",
            model_key="org/model",
            edit_type="quant_rtn",
            parameters={"bits": 8, "group_size": 32},
            scope="attn",
        )


def test_attach_receipt_rejects_missing_candidate_sidecar(tmp_path: Path) -> None:
    bundle_path, replay, evaluation = _selected_bundle(tmp_path)
    (tmp_path / str(evaluation["runtime"]["path"])).unlink()
    final_replay = tmp_path / "final_replay.json"
    _write(final_replay, replay)

    with pytest.raises(
        SelectionReceiptError, match="runtime reload proof sidecar is missing"
    ):
        attach_transformation_selection_receipt(
            replay_path=final_replay,
            selection_bundle_path=bundle_path,
            scenario_id="clean_quant_rtn",
            model_key="org/model",
            edit_type="quant_rtn",
            parameters={"bits": 8, "group_size": 32},
            scope="attn",
        )
