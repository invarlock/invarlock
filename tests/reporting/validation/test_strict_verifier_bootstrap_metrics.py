from __future__ import annotations

import math

import pytest

from invarlock.reporting import verify_bootstrap as bootstrap_mod
from tests.cli.verify._support_runtime_provenance import (
    _strict_provenance_gate_cert,
)
from tests.reporting.validation._support_strict_verifier_branch_contracts import (
    _bootstrap_errors,
)


def test_bootstrap_metric_recompute_rejects_overflow_and_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = bootstrap_mod._RawFinalWindows((1, 2), (1.0, 1.0), (1, 1))
    errors: list[str] = []
    monkeypatch.setattr(
        bootstrap_mod.math, "exp", lambda _value: (_ for _ in ()).throw(OverflowError())
    )
    bootstrap_mod._append_baseline_metric_recompute_errors(
        errors,
        report={},
        baseline_payload={},
        baseline_windows=windows,
        tolerance=1e-9,
    )
    assert errors == [
        "Supplied baseline raw final log-loss overflows finite perplexity."
    ]

    monkeypatch.setattr(bootstrap_mod.math, "exp", lambda _value: math.inf)
    errors = []
    bootstrap_mod._append_baseline_metric_recompute_errors(
        errors,
        report={},
        baseline_payload={},
        baseline_windows=windows,
        tolerance=1e-9,
    )
    assert errors == [
        "Supplied baseline raw final log-loss does not produce finite PPL >= 1."
    ]


def test_bootstrap_reported_ci_and_missing_baseline_fail_closed() -> None:
    for ci, expected in ((None, "two finite"), ([2.0, 1.0], "ordered finite")):
        errors: list[str] = []
        assert (
            bootstrap_mod._reported_ci(errors, {"primary_metric": {"ci": ci}}) is None
        )
        assert any(expected in error for error in errors)

    report = _strict_provenance_gate_cert()
    errors = _bootstrap_errors(report, None)
    assert any(
        "could not load the independently supplied baseline" in error
        for error in errors
    )
