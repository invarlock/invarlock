from __future__ import annotations

import json
from pathlib import Path

from tests.scripts._support_release_evidence_check import (
    release_checker_module as _release_checker_module,
)
from tests.scripts._support_release_evidence_check import repo_root as _repo_root


def test_release_checklist_is_not_part_of_public_repo_surface() -> None:
    repo_root = _repo_root()
    assert not (repo_root / ".github" / "release-checklist.md").exists()
    assert not (repo_root / "docs" / "release").exists()


def test_release_evidence_check_requires_independent_pin_for_matching_report(
    tmp_path: Path,
) -> None:
    module = _release_checker_module(_repo_root())
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}", encoding="utf-8")
    verify = tmp_path / "verify.json"
    valid_provenance = {
        "status": "expected_image_digest_matched",
        "verified": True,
        "binding_verified": True,
        "expected_digest_matched": True,
    }

    for field, weak_value in (
        ("status", "manifest_bound"),
        ("verified", False),
        ("binding_verified", False),
        ("expected_digest_matched", False),
    ):
        weak_provenance = dict(valid_provenance)
        weak_provenance[field] = weak_value
        verify.write_text(
            json.dumps(
                {
                    "summary": {"ok": True},
                    "results": [
                        {
                            "id": str(report),
                            "verification": {
                                "runtime_provenance": weak_provenance,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        failures: list[str] = []
        module._validate_strict_verify(verify, report, failures)
        assert any(
            "independently supplied runtime image digest pin" in item
            for item in failures
        ), field

    verify.write_text(
        json.dumps(
            {
                "summary": {"ok": True},
                "results": [
                    {
                        "id": str(tmp_path / "decoy" / "evaluation.report.json"),
                        "verification": {
                            "runtime_provenance": valid_provenance,
                        },
                    },
                    {
                        "id": str(report),
                        "verification": {
                            "runtime_provenance": {
                                **valid_provenance,
                                "expected_digest_matched": False,
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    failures = []
    module._validate_strict_verify(verify, report, failures)
    assert any(
        "independently supplied runtime image digest pin" in item for item in failures
    )
