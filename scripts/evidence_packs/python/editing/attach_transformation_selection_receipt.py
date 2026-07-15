# ruff: noqa: UP045  # Evidence-pack shell hosts still include Python 3.9.
"""Attach a v1 candidate-bound clean-selection receipt to a replay sidecar.

The retired tuned-parameter file only stated a selected value.  This helper
instead consumes a staged v1 bundle whose every candidate has retained,
verified report/replay/runtime JSON evidence.  A final clean replay is accepted
only when it is the deterministic winner's exact artifact and baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from invarlock.clean_selection.common import (
    CLEAN_SELECTION_CONTRACT_VERSION,
    CleanSelectionEvidenceError,
    canonical_json_sha256,
    strict_json_object_snapshot,
)
from invarlock.clean_selection.snapshot import (
    selected_entry_for,
    snapshot_selection_bundle_file,
)
from invarlock.evidence_pack_json import sha256_prefixed

try:
    from .transformation_contract import (
        TRANSFORMATION_CONTRACT_VERSION,
        TransformationContractError,
        canonical_transformation_spec,
        validate_transformation_scope,
    )
except ImportError:  # pragma: no cover - direct script-path loading
    from transformation_contract import (  # type: ignore[no-redef]
        TRANSFORMATION_CONTRACT_VERSION,
        TransformationContractError,
        canonical_transformation_spec,
        validate_transformation_scope,
    )


TRANSFORMATION_SELECTION_RECEIPT_SCHEMA = (
    "invarlock/generated-transformation-selection-v1"
)
TRANSFORMATION_SELECTION_SOURCE_PATH = "metadata/clean_selection/selection_bundle.json"
_SCENARIO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SelectionReceiptError(ValueError):
    """Raised when final clean replay evidence cannot bind a v1 winner."""


def _scenario_id(value: object) -> str:
    if not isinstance(value, str) or _SCENARIO_ID_RE.fullmatch(value) is None:
        raise SelectionReceiptError("scenario_id must be a safe single identifier")
    return value


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise SelectionReceiptError(
            f"could not atomically write replay payload: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_transformation(
    *, edit_type: object, parameters: object, scope: object
) -> tuple[dict[str, object], str]:
    try:
        transformation = canonical_transformation_spec(edit_type, parameters)
        canonical_scope = validate_transformation_scope(scope)
    except TransformationContractError as exc:
        raise SelectionReceiptError(str(exc)) from exc
    if not isinstance(transformation, dict):  # defensive contract boundary
        raise SelectionReceiptError("canonical transformation is invalid")
    return transformation, canonical_scope


def attach_transformation_selection_receipt(
    *,
    replay_path: Path,
    selection_bundle_path: Path,
    scenario_id: str,
    model_key: str,
    edit_type: str,
    parameters: Mapping[str, object],
    scope: str,
) -> dict[str, object]:
    """Attach the sole accepted v1 clean-selection receipt idempotently."""

    try:
        bundle_snapshot = snapshot_selection_bundle_file(selection_bundle_path)
        bundle = bundle_snapshot.bundle
        selected = selected_entry_for(
            bundle,
            model_key=model_key,
            edit_type=edit_type,
            requested_scope=scope,
        )
        _, replay = strict_json_object_snapshot(
            replay_path, label="final transformation replay"
        )
    except CleanSelectionEvidenceError as exc:
        raise SelectionReceiptError(str(exc)) from exc
    transformation, canonical_scope = _canonical_transformation(
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
    )
    entry = selected["selected_entry"]
    assert isinstance(entry, Mapping)
    candidate_receipt = entry["selection_receipt"]
    assert isinstance(candidate_receipt, Mapping)
    selected_transformation = candidate_receipt["selected_transformation"]
    selected_evaluation = candidate_receipt["selected_evaluation"]
    assert isinstance(selected_transformation, Mapping)
    assert isinstance(selected_evaluation, Mapping)
    if selected_transformation != {
        "edit_type": transformation["edit_type"],
        "parameters": transformation["parameters"],
        "scope": canonical_scope,
    }:
        raise SelectionReceiptError(
            "final transformation parameters do not match the v1-selected candidate"
        )
    artifact_identity = replay.get("artifact_identity")
    baseline_identity = replay.get("baseline_identity")
    replay_ref = selected_evaluation.get("replay")
    if not isinstance(replay_ref, Mapping):
        raise SelectionReceiptError("selected candidate has no replay reference")
    if artifact_identity != replay_ref.get("artifact_identity"):
        raise SelectionReceiptError(
            "final replay artifact identity does not match selected candidate evidence"
        )
    if baseline_identity != replay_ref.get("baseline_identity"):
        raise SelectionReceiptError(
            "final replay baseline identity does not match selected candidate evidence"
        )
    if replay.get("ok") is not True or replay.get("issues") != []:
        raise SelectionReceiptError("final transformation replay must be successful")
    if (
        replay.get("edit_type") != selected_transformation["edit_type"]
        or replay.get("parameters") != selected_transformation["parameters"]
        or replay.get("scope") != selected_transformation["scope"]
        or replay.get("algorithm") != transformation["algorithm"]
        or replay.get("transformation") != transformation
    ):
        raise SelectionReceiptError(
            "final replay transformation does not match selected candidate"
        )

    receipt: dict[str, object] = {
        "schema": TRANSFORMATION_SELECTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "transformation_contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scenario_id": _scenario_id(scenario_id),
        "selection_bundle_path": TRANSFORMATION_SELECTION_SOURCE_PATH,
        "selection_bundle_sha256": sha256_prefixed(bundle_snapshot.bundle_bytes),
        "original_model_key": model_key,
        "edit_type": transformation["edit_type"],
        "algorithm": transformation["algorithm"],
        "parameters": transformation["parameters"],
        "scope": canonical_scope,
        "selected_candidate_id": candidate_receipt["selected_candidate_id"],
        "candidate_set_sha256": candidate_receipt["candidate_set_sha256"],
        "selected_entry_sha256": canonical_json_sha256(selected),
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
    }
    has_receipt = "selection_receipt" in replay
    has_digest = "selection_receipt_sha256" in replay
    if has_receipt != has_digest:
        raise SelectionReceiptError("replay has a partial selection receipt")
    receipt_digest = canonical_json_sha256(receipt)
    if has_receipt:
        if replay.get("selection_receipt") != receipt:
            raise SelectionReceiptError(
                "replay already has a different selection receipt"
            )
        if replay.get("selection_receipt_sha256") != receipt_digest:
            raise SelectionReceiptError(
                "replay selection receipt digest does not match"
            )
        return receipt
    replay["selection_receipt"] = receipt
    replay["selection_receipt_sha256"] = receipt_digest
    _write_json_atomically(replay_path, replay)
    return receipt


def _parameters_from_cli(raw_parameters: str) -> dict[str, object]:
    try:
        value = json.loads(raw_parameters)
    except json.JSONDecodeError as exc:
        raise SelectionReceiptError("--parameters-json must be valid JSON") from exc
    if not isinstance(value, dict):
        raise SelectionReceiptError("--parameters-json must be a JSON object")
    return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--selection-bundle", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--edit-type", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args(argv)
    attach_transformation_selection_receipt(
        replay_path=args.replay,
        selection_bundle_path=args.selection_bundle,
        scenario_id=args.scenario_id,
        model_key=args.model_key,
        edit_type=args.edit_type,
        parameters=_parameters_from_cli(args.parameters_json),
        scope=args.scope,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "SelectionReceiptError",
    "TRANSFORMATION_SELECTION_RECEIPT_SCHEMA",
    "TRANSFORMATION_SELECTION_SOURCE_PATH",
    "attach_transformation_selection_receipt",
]
