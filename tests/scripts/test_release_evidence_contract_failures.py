from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.smoke.guard_validation_smoke import build_guard_validation_smoke
from tests.scripts._support_evidence_contracts import load_evidence_contracts_module


def test_guard_validation_manifest_reports_malformed_rate_evidence(
    tmp_path: Path,
) -> None:
    module = load_evidence_contracts_module()
    payload = build_guard_validation_smoke(replicates=5, seed=7)
    rows = payload["rate_rows"]
    assert isinstance(rows, list)
    rows[0] = "not-an-object"
    malformed_shape = copy.deepcopy(rows[1])
    malformed_shape["extra"] = True
    rows[1] = malformed_shape
    forged_outcome = copy.deepcopy(rows[2])
    forged_outcome["null_outcomes"][0] = not forged_outcome["null_outcomes"][0]
    rows[2] = forged_outcome
    forged_rate = copy.deepcopy(rows[3])
    forged_rate["shifted_trigger_rate"] = 0.5
    rows[3] = forged_rate
    wrong_binding = copy.deepcopy(rows[4])
    wrong_binding["derived_seed"] += 1
    rows[4] = wrong_binding
    markdown = tmp_path / "guard-validation.md"
    failures: list[str] = []

    module.GuardValidationSmokeManifest(
        json_path=tmp_path / "guard-validation.json",
        markdown_path=markdown,
        payload=payload,
        markdown_bytes=b"not canonical\n",
    ).validate(failures)

    assert "guard-validation rate_rows[0] must be an object." in failures
    assert any("fields must match v1 exactly" in item for item in failures)
    assert any(
        "raw outcomes do not match deterministic replay" in item for item in failures
    )
    assert any(
        "shifted_trigger_rate does not match outcomes" in item for item in failures
    )
    assert any("identity or production binding" in item for item in failures)
    assert any("evidence_sha256 does not match" in item for item in failures)
    assert any("markdown bytes do not match" in item for item in failures)


def test_strict_verify_evidence_ignores_unidentified_results_and_fails_closed(
    tmp_path: Path,
) -> None:
    module = load_evidence_contracts_module()
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}\n", encoding="utf-8")
    failures: list[str] = []
    evidence = module.StrictVerifyEvidence(
        path=tmp_path / "verify.json",
        payload={
            "summary": {"ok": True},
            "results": [None, {"id": ""}, {"id": 7}],
        },
    )

    evidence.validate(report_path=report, failures=failures)

    assert failures == [
        "strict verifier output does not reference the strict report.",
        "strict verifier output must prove report/manifest binding to a "
        "independently supplied runtime image digest pin.",
    ]


@pytest.mark.parametrize(
    ("field", "tampered_value", "expected_failure"),
    (
        ("scope", "empirical model-family proof", "scope is not recognized"),
        ("seed", True, "seed must be a signed 64-bit integer"),
        ("replicates", 0, "replicates must be an integer in [1, 10000]"),
        (
            "production_primitives",
            {},
            "must name the production primitive and role for every required guard",
        ),
    ),
)
def test_guard_validation_manifest_rejects_top_level_contract_tampering(
    tmp_path: Path,
    field: str,
    tampered_value: object,
    expected_failure: str,
) -> None:
    module = load_evidence_contracts_module()
    payload = build_guard_validation_smoke(replicates=5, seed=7)
    payload[field] = tampered_value
    failures: list[str] = []

    module.GuardValidationSmokeManifest(
        json_path=tmp_path / "guard-validation.json",
        markdown_path=tmp_path / "guard-validation.md",
        payload=payload,
        markdown_bytes=None,
    ).validate(failures)

    assert any(expected_failure in failure for failure in failures)


@pytest.mark.parametrize("tampered_threshold", (float("inf"), 10**400))
def test_guard_validation_row_rejects_unrepresentable_numeric_thresholds(
    tampered_threshold: float | int,
) -> None:
    module = load_evidence_contracts_module()
    payload = build_guard_validation_smoke(replicates=5, seed=7)
    row = copy.deepcopy(payload["rate_rows"][0])
    row["threshold"] = tampered_threshold
    failures: list[str] = []

    module.GuardValidationSmokeManifest._validate_row(
        index=0,
        row=row,
        guard="spectral",
        windows=16,
        seed=7,
        replicates=5,
        threshold=module._guard_thresholds()["spectral"],
        failures=failures,
    )

    assert failures == ["guard-validation rate_rows[0] scalar field types are invalid."]
