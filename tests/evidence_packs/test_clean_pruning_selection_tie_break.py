from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from invarlock.clean_pruning_selection_common import (
    CleanPruningSelectionEvidenceError,
)
from invarlock.clean_pruning_selection_contract import (
    canonical_clean_pruning_bundle_sha256,
)
from invarlock.clean_pruning_selection_contracts.snapshot import (
    verify_clean_pruning_selection_bundle_file,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _record,
    _refresh_record_and_bundle,
    _refresh_report_and_manifest,
    _write,
)


def test_winner_tie_breaks_by_candidate_id_and_rejects_changed_receipt(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    candidates = cast(list[dict[str, object]], record["candidates"])
    for candidate in candidates:
        evaluation = cast(dict[str, object], candidate["evaluation"])
        evaluation["metrics"] = {"quality_loss": 0.02}
        for repeat_index, report_run in enumerate(
            cast(list[dict[str, object]], evaluation["reports"])
        ):
            report_ref = cast(dict[str, object], report_run["report"])
            report_path = tmp_path / cast(str, report_ref["path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["primary_metric"]["ratio_vs_baseline"] = 1.02
            report["clean_pruning_selection"]["quality_loss"] = 0.02
            _write(report_path, report)
            _refresh_report_and_manifest(tmp_path, candidate, repeat_index)
    bundle_path, bundle = _refresh_record_and_bundle(tmp_path, record)
    selected = cast(dict[str, object], bundle["entries"][0])["selected_entry"]
    assert cast(dict[str, object], selected)["scope"] == "attn"
    assert verify_clean_pruning_selection_bundle_file(bundle_path) == bundle

    forged = copy.deepcopy(bundle)
    entry = cast(dict[str, object], forged["entries"][0])
    selected_entry = cast(dict[str, object], entry["selected_entry"])
    receipt = cast(dict[str, object], selected_entry["selection_receipt"])
    receipt["selected_candidate_id"] = "ffn-20"
    forged["bundle_sha256"] = canonical_clean_pruning_bundle_sha256(
        cast(list[dict[str, object]], forged["entries"])
    )
    _write(tmp_path / "forged-bundle.json", forged)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="winner"):
        verify_clean_pruning_selection_bundle_file(tmp_path / "forged-bundle.json")
