from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock.clean_selection.common import CleanSelectionEvidenceError
from invarlock.clean_selection.snapshot import (
    _verify_candidate_artifacts,
    snapshot_selection_bundle_file,
)
from invarlock.evidence_pack_json import sha256_prefixed
from scripts.evidence_packs.python.editing import (
    attach_transformation_selection_receipt as attach_module,
)
from scripts.evidence_packs.python.editing import (
    clean_selection_bundle as bundle_module,
)
from scripts.evidence_packs.python.editing.clean_selection_contract import (
    clean_edit_dir_name,
)
from tests.evidence_packs._support_clean_selection import _bundle, _record


def _selected_replay(root: Path, record: dict[str, object]) -> dict[str, object]:
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    replay_path = root / str(evaluation["replay"]["path"])
    return json.loads(replay_path.read_text(encoding="utf-8"))


def test_stage_publishes_verified_snapshot_not_later_source_substitution(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    record = _record(source)
    bundle_path, _ = _bundle(source, record)
    snapshot = snapshot_selection_bundle_file(bundle_path)
    bundle_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        bundle_module,
        "snapshot_selection_bundle_file",
        lambda _path: snapshot,
    )
    staged = bundle_module.stage_selection_bundle(
        bundle_path=bundle_path,
        destination=tmp_path / "staged",
    )

    assert staged.read_bytes() == snapshot.bundle_bytes
    assert (staged.parent / "candidates" / "attn8" / "execution.json").read_bytes() == (
        snapshot.sidecar_bytes["candidates/attn8/execution.json"]
    )


def test_snapshot_accepts_an_explicit_evidence_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    record = _record(source)
    bundle_path, bundle = _bundle(source, record)

    snapshot = snapshot_selection_bundle_file(bundle_path, evidence_root=source)

    assert snapshot.bundle == bundle


def test_candidate_snapshot_rejects_reused_sidecar_path(tmp_path: Path) -> None:
    record = _record(tmp_path)
    _bundle_path, bundle = _bundle(tmp_path, record)
    entry = copy.deepcopy(bundle["entries"][0])
    receipt = entry["selected_entry"]["selection_receipt"]
    candidate = receipt["candidates"][0]
    candidate["evaluation"]["replay"] = copy.deepcopy(
        candidate["evaluation"]["execution"]
    )

    with pytest.raises(CleanSelectionEvidenceError, match="reuse one sidecar path"):
        _verify_candidate_artifacts(entry, tmp_path)


def test_attach_uses_verified_bundle_snapshot_not_later_substitution(
    tmp_path: Path, monkeypatch
) -> None:
    record = _record(tmp_path)
    bundle_path, _ = _bundle(tmp_path, record)
    snapshot = snapshot_selection_bundle_file(bundle_path)
    final_replay = tmp_path / "final_replay.json"
    final_replay.write_text(
        json.dumps(_selected_replay(tmp_path, record), sort_keys=True),
        encoding="utf-8",
    )
    bundle_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        attach_module,
        "snapshot_selection_bundle_file",
        lambda _path: snapshot,
    )
    receipt = attach_module.attach_transformation_selection_receipt(
        replay_path=final_replay,
        selection_bundle_path=bundle_path,
        scenario_id="clean_quant_rtn",
        model_key="org/model",
        edit_type="quant_rtn",
        parameters={"bits": 8, "group_size": 32},
        scope="attn",
    )

    assert receipt["selection_bundle_sha256"] == sha256_prefixed(snapshot.bundle_bytes)


def test_clean_selection_bridge_uses_the_selected_full_transform_identity(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)
    entries = bundle["entries"]
    assert isinstance(entries, list)
    selected = entries[0]
    assert isinstance(selected, dict)

    expected = clean_edit_dir_name(selected)
    resolved = bundle_module.resolve_clean_selection(
        bundle_path=bundle_path,
        model_key="org/model",
        edit_type="quant_rtn",
    )

    assert resolved["edit_dir_name"] == expected
    assert expected.startswith(
        "generated--quant_rtn--bits-8--group_size-32--scope-attn--version-clean"
    )
    assert "--sha256-" in expected


@pytest.mark.parametrize("payload", ['{"ok": false, "ok": true}', '{"score": NaN}'])
def test_clean_selection_cli_json_rejects_ambiguous_values(payload: str) -> None:
    with pytest.raises(CleanSelectionEvidenceError, match="not valid JSON"):
        bundle_module._parse_json_argument(payload, label="--edit-specs-json")
