"""Script-facing clean-selection contract operations.

The verifier must also work from an installed wheel, so the canonical parser,
candidate-set digest, winner recomputation, and sidecar checks come directly
from their package owners instead of a script-local or aggregate schema.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .implementations import generated_transformation_edit_dir_name
except ImportError:  # pragma: no cover - direct script-path loading
    from implementations import generated_transformation_edit_dir_name

from invarlock.clean_selection.bundle import (
    select_clean_transformation,
    verify_selected_entry,
    verify_selection_bundle,
)
from invarlock.clean_selection.candidate import canonical_candidate_set_sha256
from invarlock.clean_selection.common import (
    CANDIDATE_EVALUATION_SCHEMA,
    CANDIDATE_RECORD_SCHEMA,
    CLEAN_SELECTION_BUNDLE_SCHEMA,
    CLEAN_SELECTION_CONTRACT_VERSION,
    DECISION_RULE_SCHEMA,
    EVALUATION_SCHEDULE_SCHEMA,
    REPORT_SELECTION_BINDING_SCHEMA,
    SELECTED_ENTRY_SCHEMA,
    SELECTION_CONFIG_SCHEMA,
    SELECTION_RECEIPT_SCHEMA,
    CleanSelectionEvidenceError,
    canonical_json_sha256,
    strict_json_object,
)
from invarlock.clean_selection.snapshot import selected_entry_for

CleanSelectionContractError = CleanSelectionEvidenceError
canonical_sha256 = canonical_json_sha256


def load_candidate_record(path: Path) -> dict[str, object]:
    """Load a regular duplicate-free candidate record JSON object."""

    return strict_json_object(path, label="candidate record")


def load_selection_bundle(path: Path) -> dict[str, object]:
    """Load and structurally verify a canonical v1 selected-entry bundle."""

    return verify_selection_bundle(strict_json_object(path, label="selection bundle"))


def clean_edit_dir_name(selected_entry: Mapping[str, object]) -> str:
    """Return the deterministic final directory for a verified v1 selection."""

    entry = verify_selected_entry(selected_entry)["selected_entry"]
    assert isinstance(entry, Mapping)
    edit_type = entry["edit_type"]
    parameters = entry["parameters"]
    scope = entry["scope"]
    assert isinstance(edit_type, str)
    assert isinstance(parameters, Mapping)
    assert isinstance(scope, str)
    try:
        return generated_transformation_edit_dir_name(
            edit_type=edit_type,
            parameters=parameters,
            scope=scope,
            version="clean",
        )
    except ValueError as exc:
        raise CleanSelectionContractError(
            "selected entry has no verifier-grade generated-transformation directory"
        ) from exc


__all__ = [
    "CANDIDATE_EVALUATION_SCHEMA",
    "CANDIDATE_RECORD_SCHEMA",
    "CLEAN_SELECTION_BUNDLE_SCHEMA",
    "CLEAN_SELECTION_CONTRACT_VERSION",
    "CleanSelectionContractError",
    "DECISION_RULE_SCHEMA",
    "EVALUATION_SCHEDULE_SCHEMA",
    "REPORT_SELECTION_BINDING_SCHEMA",
    "SELECTION_CONFIG_SCHEMA",
    "SELECTION_RECEIPT_SCHEMA",
    "SELECTED_ENTRY_SCHEMA",
    "canonical_candidate_set_sha256",
    "canonical_sha256",
    "clean_edit_dir_name",
    "load_candidate_record",
    "load_selection_bundle",
    "select_clean_transformation",
    "selected_entry_for",
    "verify_selected_entry",
    "verify_selection_bundle",
]
