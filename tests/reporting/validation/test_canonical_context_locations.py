from __future__ import annotations

import pytest

from invarlock.reporting.verify_baseline import _append_context_binding_errors


def _canonical_subject() -> dict[str, object]:
    return {"assurance": {"profile": "ci", "tier": "balanced"}}


def _canonical_baseline_context() -> dict[str, object]:
    return {
        "context": {
            "profile": "ci",
            "auto": {"tier": "balanced"},
            "assurance": {"mode": "strict"},
        }
    }


def test_strict_baseline_context_accepts_only_canonical_locations() -> None:
    errors: list[str] = []

    _append_context_binding_errors(
        errors,
        subject=_canonical_subject(),
        baseline=_canonical_baseline_context(),
    )

    assert errors == []


@pytest.mark.parametrize(
    ("subject", "baseline", "expected"),
    [
        (
            {"context": {"profile": "ci"}, "assurance": {"tier": "balanced"}},
            _canonical_baseline_context(),
            "profile mismatch",
        ),
        (
            {"assurance": {"profile": "ci"}, "auto": {"tier": "balanced"}},
            _canonical_baseline_context(),
            "tier mismatch",
        ),
        (
            _canonical_subject(),
            {
                "context": {
                    "auto": {"tier": "balanced"},
                    "assurance": {"mode": "strict"},
                },
                "meta": {"profile": "ci"},
            },
            "context.profile",
        ),
        (
            _canonical_subject(),
            {
                "context": {
                    "profile": "ci",
                    "tier": "balanced",
                    "assurance": {"mode": "strict"},
                }
            },
            "context.auto.tier",
        ),
        (
            _canonical_subject(),
            {
                "context": {"profile": "ci", "auto": {"tier": "balanced"}},
                "assurance": {"mode": "strict"},
            },
            "context.assurance.mode",
        ),
    ],
)
def test_strict_baseline_context_rejects_alternate_locations(
    subject: dict[str, object],
    baseline: dict[str, object],
    expected: str,
) -> None:
    errors: list[str] = []

    _append_context_binding_errors(errors, subject=subject, baseline=baseline)

    assert any(expected in error for error in errors)
