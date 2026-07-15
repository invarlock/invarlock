from __future__ import annotations

import pytest

from invarlock.core.checkpoint_identity import LEGACY_MODEL_IDENTITY_FIELDS
from invarlock.reporting import verify_baseline as baseline_mod
from tests.cli.verify._support_runtime_provenance import (
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
)
from tests.reporting.validation._support_strict_verifier_branch_contracts import (
    _baseline_errors,
)


def test_strict_checkpoint_identity_requires_canonical_remote_revisions() -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report["meta"]["model_identity"] = {
        "kind": "remote_revision",
        "revision": "main",
    }
    report["subject_ref"]["model_identity"] = report["meta"]["model_identity"].copy()

    errors = _baseline_errors(report, supplied)

    assert any(
        "subject model identity must declare one canonical" in error for error in errors
    )

    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    supplied["meta"]["model_identity"] = {
        "kind": "remote_revision",
        "revision": "main",
    }
    report["baseline_ref"]["model_identity"] = supplied["meta"]["model_identity"].copy()
    errors = _baseline_errors(report, supplied)
    assert any(
        "baseline model identity must declare one canonical" in error
        for error in errors
    )


def test_strict_checkpoint_identity_rejects_local_tree_substitution() -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report["meta"]["model_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "c" * 64,
    }
    report["subject_ref"] = {
        "model_id": report["meta"]["model_id"],
        "model_identity": {
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "d" * 64,
        },
    }
    supplied["meta"]["model_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "e" * 64,
    }
    report["baseline_ref"]["model_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "f" * 64,
    }

    errors = _baseline_errors(report, supplied)

    assert any("subject_ref model_identity mismatch" in e for e in errors)
    assert any("baseline_ref model_identity mismatch" in e for e in errors)


def test_strict_checkpoint_identity_rejects_legacy_duplicate_fields() -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report["meta"]["model_revision"] = "c" * 40
    report["meta"]["model_checkpoint_tree_sha256"] = "sha256:" + "d" * 64

    errors: list[str] = []
    baseline_mod._append_checkpoint_identity_binding_errors(
        errors,
        subject=report,
        baseline=supplied,
    )

    assert any(
        "subject meta must not declare legacy model_revision" in e for e in errors
    )
    assert any(
        "subject meta must not declare legacy model_checkpoint_tree_sha256" in e
        for e in errors
    )

    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report.pop("subject_ref")
    report.pop("baseline_ref")

    errors = []
    baseline_mod._append_checkpoint_identity_binding_errors(
        errors,
        subject=report,
        baseline=supplied,
    )

    assert "Strict subject model identity requires report.subject_ref." in errors
    assert "Strict baseline model identity requires report.baseline_ref." in errors


@pytest.mark.parametrize("legacy_field", LEGACY_MODEL_IDENTITY_FIELDS)
@pytest.mark.parametrize(
    ("side", "surface"),
    [
        ("subject", "meta"),
        ("subject", "reference"),
        ("baseline", "meta"),
        ("baseline", "reference"),
    ],
)
def test_strict_checkpoint_identity_rejects_every_legacy_field_on_every_surface(
    legacy_field: str,
    side: str,
    surface: str,
) -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    if side == "subject" and surface == "meta":
        target = report["meta"]
    elif side == "subject":
        target = report["subject_ref"]
    elif surface == "meta":
        target = supplied["meta"]
    else:
        target = report["baseline_ref"]
    target[legacy_field] = "conflicting-legacy-identity"

    errors: list[str] = []
    baseline_mod._append_checkpoint_identity_binding_errors(
        errors,
        subject=report,
        baseline=supplied,
    )

    expected_surface = f"{side} meta" if surface == "meta" else f"{side}_ref"
    assert any(
        f"{expected_surface} must not declare legacy {legacy_field}" in error
        for error in errors
    )
