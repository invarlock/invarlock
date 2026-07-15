from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from invarlock.core.assurance_contract import (
    build_assurance_section,
    strict_report_policy_errors,
)
from tests.core._support_assurance_contract import (
    strict_no_adjustment_report as _strict_no_adjustment_report,
)
from tests.core._support_assurance_contract import (
    strict_variance_gain_report as _strict_variance_gain_report,
)

Mutation = Callable[[dict[str, Any], dict[str, Any]], object]


def _variance_guard(report: dict[str, Any]) -> dict[str, Any]:
    return next(entry for entry in report["guards"] if entry["name"] == "variance")


def _strict_errors(report: dict[str, Any]) -> list[str]:
    report["assurance"] = build_assurance_section(report)
    return strict_report_policy_errors(report, require_strict=True)


def _set_noop_windows_used(
    report: dict[str, Any], guard: dict[str, Any], value: object
) -> None:
    report["variance"]["ab_test"]["windows_used"] = value
    guard["metrics"]["ab_windows_used"] = value


def _set_variance_policy_value(
    report: dict[str, Any], guard: dict[str, Any], key: str, value: object
) -> None:
    report["variance"]["policy"][key] = value
    report["resolved_policy"]["variance"][key] = value
    guard["policy"][key] = value
    guard["details"]["policy"][key] = value


def _set_scale_maps(guard: dict[str, Any], *, proposed: float, raw: float) -> None:
    module = "transformer.h.0.mlp.c_proj"
    for key in ("proposed_scales_pre_edit", "proposed_scales_post_edit"):
        guard["metrics"][key] = {module: proposed}
    for key in ("raw_scales_pre_edit", "raw_scales_post_edit"):
        guard["metrics"][key] = {module: raw}
    guard["details"]["proposed_scales"] = {module: proposed}
    stats = guard["details"]["stats"]
    stats["proposed_scales_pre_edit"] = {module: proposed}
    stats["proposed_scales_post_edit"] = {module: proposed}
    stats["raw_scales_pre_edit_normalized"] = {module: raw}
    stats["raw_scales_post_edit_normalized"] = {module: raw}


def _set_coordinated_provenance(
    report: dict[str, Any], guard: dict[str, Any], key: str, value: object
) -> None:
    raw = guard["metrics"]["ab_provenance"]
    top = report["variance"]["ab_test"]["provenance"]
    for condition in ("condition_a", "condition_b"):
        raw[condition][key] = value
        top[condition][key] = value
    guard["details"]["stats"]["ab_provenance"] = {
        condition: dict(raw[condition]) for condition in ("condition_a", "condition_b")
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda _report, guard: guard["metrics"].update(
                ve_enabled_during_validation=True
            ),
            "must both be false for no_adjustment_required",
        ),
        (
            lambda _report, guard: guard["metrics"].update(
                subject_restored_after_ab=False
            ),
            "must both be true for no_adjustment_required",
        ),
        (
            lambda _report, guard: guard["metrics"].update(met_threshold=True),
            "must both be false for no_adjustment_required",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ab_gain=0.1),
            "metrics.ab_gain must be zero for no adjustment",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ppl_with_ve=99.0),
            "no-adjustment PPL arms must be finite, positive, and equal",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ratio_ci=[0.9, 1.0]),
            "metrics.ratio_ci must equal [1.0, 1.0]",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_point_estimates"].update(
                ppl_with_ve=99.0
            ),
            "ab_point_estimates.ppl_with_ve must match",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_a"
            ].update(window_ids=["0", "0"]),
            "condition_a.window_ids must be non-empty and unique",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_b"
            ].update(model_id="other-model"),
            "conditions must share model_id",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_b"
            ].update(tag="pre_edit"),
            "condition_b.tag must be post_edit",
        ),
        (
            lambda _report, guard: guard.update(errors=["calibration failed"]),
            "errors must be empty for strict assurance",
        ),
        (
            lambda _report, guard: guard.update(warnings=["result is uncertain"]),
            "warnings must be empty for strict assurance",
        ),
    ],
)
def test_strict_no_adjustment_rejects_contradictory_raw_evidence(
    mutation: Mutation,
    expected: str,
) -> None:
    report = _strict_no_adjustment_report()
    mutation(report, _variance_guard(report))

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


def test_strict_no_adjustment_binds_policy_calibration_to_raw_counts() -> None:
    report = _strict_no_adjustment_report()
    guard = _variance_guard(report)
    for policy in (
        report["variance"]["policy"],
        guard["policy"],
        guard["details"]["policy"],
    ):
        policy["calibration"]["windows"] = 7

    errors = _strict_errors(report)

    assert any("policy.calibration.windows must match" in error for error in errors), (
        "\n".join(errors)
    )


@pytest.mark.parametrize("value", ["eight", True, -1, 999])
def test_strict_no_adjustment_rejects_coordinated_invalid_window_counts(
    value: object,
) -> None:
    report = _strict_no_adjustment_report()
    guard = _variance_guard(report)
    _set_noop_windows_used(report, guard, value)

    errors = _strict_errors(report)

    assert any(
        "ab_windows_used must be a positive integer equal" in error for error in errors
    ), "\n".join(errors)


def test_strict_no_adjustment_requires_point_estimate_coverage() -> None:
    report = _strict_no_adjustment_report()
    guard = _variance_guard(report)
    report["variance"]["ab_test"]["point_estimates"].pop("coverage")
    guard["metrics"]["ab_point_estimates"].pop("coverage")
    guard["details"]["stats"]["ab_point_estimates"].pop("coverage")

    errors = _strict_errors(report)

    assert any("ab_point_estimates.coverage must match" in error for error in errors), (
        "\n".join(errors)
    )


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("monitor_only", True, "policy.monitor_only must be false"),
        ("predictive_gate", False, "policy.predictive_gate must be true"),
        ("mode", "nonsense", "policy.mode must be ci"),
    ],
)
def test_strict_no_adjustment_rejects_coordinated_policy_attacks(
    key: str,
    value: object,
    expected: str,
) -> None:
    report = _strict_no_adjustment_report()
    guard = _variance_guard(report)
    _set_variance_policy_value(report, guard, key, value)
    if key == "monitor_only":
        guard["metrics"]["monitor_only"] = value

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


def test_strict_no_adjustment_requires_exact_ci_mode() -> None:
    report = _strict_no_adjustment_report()
    guard = _variance_guard(report)
    _set_variance_policy_value(report, guard, "mode", "CI")
    guard["metrics"]["mode"] = "CI"

    errors = _strict_errors(report)

    assert any("policy.mode must be ci" in error for error in errors), "\n".join(errors)


def test_strict_no_adjustment_rejects_stale_aggregate_window_ids() -> None:
    report = _strict_no_adjustment_report()
    guard = _variance_guard(report)
    report["variance"]["ab_test"]["provenance"]["window_ids"] = ["stale"]
    guard["metrics"]["ab_provenance"]["window_ids"] = ["stale"]

    errors = _strict_errors(report)

    assert any(
        "variance.ab_test.provenance.window_ids must match raw condition IDs" in error
        for error in errors
    ), "\n".join(errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report, guard: (
                report["variance"].pop("policy"),
                guard.pop("policy"),
            ),
            "policy and variance.policy are required for no_adjustment_required",
        ),
        (
            lambda _report, guard: guard["policy"].update(seed=999),
            "policy must match variance.policy exactly",
        ),
        (
            lambda report, _guard: report["variance"].pop("ab_test"),
            "variance.ab_test is required for no_adjustment_required assurance",
        ),
        (
            lambda _report, guard: guard.pop("details"),
            "details is required for strict variance assurance",
        ),
        (
            lambda _report, guard: guard["details"].update(stats=None),
            "details.stats is required for strict variance assurance",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"].pop("condition_b"),
            "ab_provenance.condition_b is required",
        ),
        (
            lambda _report, guard: guard["details"]["stats"].pop("calibration"),
            "details.stats.calibration is required",
        ),
        (
            lambda _report, guard: guard["details"]["stats"].pop("pairing_reference"),
            "details.stats.pairing_reference is required",
        ),
        (
            lambda _report, guard: guard["details"]["stats"].pop("dataset_meta"),
            "details.stats.dataset_meta is required",
        ),
        (
            lambda report, _guard: report["variance"]["ab_test"]["provenance"].pop(
                "window_ids"
            ),
            "variance.ab_test.provenance.window_ids is required",
        ),
    ],
)
def test_strict_no_adjustment_rejects_missing_bound_structures(
    mutation: Mutation,
    expected: str,
) -> None:
    report = _strict_no_adjustment_report()
    mutation(report, _variance_guard(report))

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda _report, guard: guard["metrics"].pop("proposed_scales_post_edit"),
            "metrics.proposed_scales_post_edit must be a non-empty object",
        ),
        (
            lambda _report, guard: guard["metrics"].update(target_module_names=[]),
            "target_module_names must enumerate all targets",
        ),
        (
            lambda _report, guard: guard["details"]["stats"]["calibration"].update(
                window_ids=["wrong"]
            ),
            "details.stats.calibration.window_ids must match A/B IDs",
        ),
        (
            lambda _report, guard: guard["details"]["stats"][
                "pairing_reference"
            ].update(digest="wrong"),
            "details.stats.pairing_reference.digest must match A/B provenance",
        ),
        (
            lambda _report, guard: guard["details"]["stats"]["dataset_meta"].update(
                dataset_hash="wrong"
            ),
            "details.stats.dataset_meta.dataset_hash must match A/B provenance",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_b"
            ].update(run_seed=999),
            "conditions must share run_seed",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_a"
            ].update(pairing_digest=""),
            "condition_a.pairing_digest is required",
        ),
    ],
)
def test_strict_gain_rejects_incomplete_or_drifting_evidence(
    mutation: Mutation,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    mutation(report, _variance_guard(report))

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


def test_strict_gain_binds_nested_calibration_seed_not_flat_policy_seed() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    report["variance"]["policy"]["seed"] = 999
    guard["policy"]["seed"] = 999
    guard["details"]["policy"]["seed"] = 999
    report["resolved_policy"]["variance"]["seed"] = 999
    report["assurance"] = build_assurance_section(report)

    assert _strict_errors(report) == []


def test_strict_gain_rejects_calibration_seed_drift_from_ab_seed() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    for policy in (
        report["variance"]["policy"],
        guard["policy"],
        guard["details"]["policy"],
    ):
        policy["calibration"]["seed"] = 999

    errors = _strict_errors(report)

    assert any(
        "metrics.ab_seed_used must match policy.calibration.seed" in error
        for error in errors
    ), "\n".join(errors)


@pytest.mark.parametrize(
    ("proposed", "raw", "expected"),
    [
        (1.0, 1.0, "must be non-identity"),
        (999.0, 999.0, "outside policy.clamp"),
        (0.98, 1.1, "must preserve the raw scale direction"),
        (1.01, 1.1, "must be derived from the raw scale and max_scale_step"),
        (1.02, 1.005, "cannot exceed the raw scale adjustment"),
    ],
)
def test_strict_gain_rejects_coordinated_scale_fabrication(
    proposed: float,
    raw: float,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_scale_maps(guard, proposed=proposed, raw=raw)

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("clamp", [2.0, 3.0], "policy.clamp must be positive, ordered"),
        ("max_scale_step", -1.0, "policy.max_scale_step must be non-negative"),
        ("min_abs_adjust", -1.0, "policy.min_abs_adjust must be non-negative"),
        ("topk_backstop", True, "policy.topk_backstop must be a non-negative integer"),
        ("deadband", -1.0, "policy.deadband must be non-negative"),
        (
            "max_adjusted_modules",
            True,
            "policy.max_adjusted_modules must be a non-negative integer",
        ),
    ],
)
def test_strict_gain_rejects_invalid_scale_policy(
    key: str,
    value: object,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_variance_policy_value(report, guard, key, value)

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


def test_strict_gain_rejects_scale_below_backstop_threshold() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_scale_maps(guard, proposed=1.005, raw=1.005)

    errors = _strict_errors(report)

    assert any(
        "does not meet min_abs_adjust or the configured backstop threshold" in error
        for error in errors
    ), "\n".join(errors)


def test_strict_gain_rejects_more_scales_than_policy_limit() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_variance_policy_value(report, guard, "max_adjusted_modules", 1)
    guard["metrics"]["proposed_scales"] = 2

    errors = _strict_errors(report)

    assert any(
        "proposed_scales exceeds policy.max_adjusted_modules" in error
        for error in errors
    ), "\n".join(errors)


def test_strict_gain_accepts_exact_raw_scale_when_step_limit_is_disabled() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_variance_policy_value(report, guard, "max_scale_step", 0.0)
    _set_scale_maps(guard, proposed=1.1, raw=1.1)

    assert _strict_errors(report) == []


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("model_id", "forged-model", "model_id must match report provenance"),
        ("run_seed", 999, "run_seed must match report provenance"),
        ("dataset_hash", "forged-dataset", "dataset_hash must match report provenance"),
        (
            "tokenizer_hash",
            "forged-tokenizer",
            "tokenizer_hash must match report provenance",
        ),
        (
            "pairing_digest",
            "forged-pairing",
            "pairing_digest must match report provenance",
        ),
    ],
)
def test_strict_gain_rejects_coordinated_report_provenance_fabrication(
    key: str,
    value: object,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_coordinated_provenance(report, guard, key, value)
    if key in {"dataset_hash", "tokenizer_hash"}:
        guard["details"]["stats"]["dataset_meta"][key] = value
    if key == "pairing_digest":
        guard["details"]["stats"]["pairing_reference"]["digest"] = value

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


def test_strict_gain_rejects_policy_not_exactly_resolved() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    for policy in (
        report["variance"]["policy"],
        guard["policy"],
        guard["details"]["policy"],
    ):
        policy["unresolved_extra"] = True

    errors = _strict_errors(report)

    assert any(
        "policy must match report.resolved_policy.variance exactly" in error
        for error in errors
    ), "\n".join(errors)


def test_strict_gain_rejects_metrics_mode_drift() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    guard["metrics"]["mode"] = "threshold-only"

    errors = _strict_errors(report)

    assert any(
        "metrics.mode must match guards[3].policy.mode exactly" in error
        for error in errors
    ), "\n".join(errors)


def test_strict_gain_rejects_window_ids_outside_report_pairing_schedule() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    unrelated = [f"unrelated::{index}" for index in range(8)]
    raw = guard["metrics"]["ab_provenance"]
    top = report["variance"]["ab_test"]["provenance"]
    for condition in ("condition_a", "condition_b"):
        raw[condition]["window_ids"] = list(unrelated)
        top[condition]["window_ids"] = list(unrelated)
    top["window_ids"] = list(unrelated)
    guard["details"]["stats"]["ab_provenance"] = {
        condition: dict(raw[condition]) for condition in ("condition_a", "condition_b")
    }
    guard["details"]["stats"]["calibration"]["window_ids"] = list(unrelated)

    errors = _strict_errors(report)

    assert any(
        "ab_provenance.window_ids must match the consumed prefix" in error
        for error in errors
    ), "\n".join(errors)


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (
            ("data", "dataset_hash"),
            "conflicting-public-dataset",
            "data.dataset_hash must match dataset.hash.dataset",
        ),
        (
            ("dataset", "tokenizer", "hash"),
            "conflicting-public-tokenizer",
            "dataset.tokenizer.hash must match meta.tokenizer_hash",
        ),
    ],
)
def test_strict_gain_rejects_public_dataset_identity_conflicts(
    path: tuple[str, ...],
    value: str,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    target = report
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report: report["dataset"]["hash"].pop("dataset"),
            "report dataset_hash is required for strict variance provenance",
        ),
        (
            lambda report: report["meta"].pop("tokenizer_hash"),
            "report tokenizer_hash is required for strict variance provenance",
        ),
        (
            lambda report: report.pop("evaluation_windows"),
            "report pairing_digest is required for strict variance provenance",
        ),
    ],
)
def test_strict_gain_requires_canonical_public_provenance(
    mutation: Callable[[dict[str, Any]], object], expected: str
) -> None:
    report = _strict_variance_gain_report()
    mutation(report)

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda _report, guard: guard["metrics"].update(proposed_scales=0),
            "metrics.proposed_scales must be positive",
        ),
        (
            lambda _report, guard: guard["metrics"].update(target_modules=0),
            "metrics.target_modules must be positive",
        ),
        (
            lambda _report, guard: guard["details"].update(proposed_scales={}),
            "details.proposed_scales must match metrics.proposed_scales",
        ),
        (
            lambda _report, guard: guard["metrics"]["proposed_scales_post_edit"].update(
                {"transformer.h.0.mlp.c_proj": -1.0}
            ),
            "must map declared targets to positive finite scales",
        ),
        (
            lambda _report, guard: guard["metrics"]["proposed_scales_post_edit"].update(
                {"undeclared.module": 1.01}
            ),
            "must map declared targets to positive finite scales",
        ),
        (
            lambda _report, guard: guard["metrics"].update(proposed_scales=2),
            "proposed_scales must match the post-edit scale map",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ppl_no_ve=0.0),
            "metrics.ppl_no_ve must be finite and positive",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ppl_with_ve=0.0),
            "metrics.ppl_with_ve must be finite and positive",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ab_gain="invalid"),
            "metrics.ab_gain must be finite for ci_gain_met",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ratio_ci=[1.0]),
            "metrics.ratio_ci must be a finite two-value interval",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ratio_ci=[2.0, 1.0]),
            "metrics.ratio_ci must be positive and ordered",
        ),
        (
            lambda _report, guard: guard["metrics"]["calibration"].update(requested=0),
            "metrics.calibration.requested must be positive",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ab_windows_used=0),
            "metrics.ab_windows_used must be positive",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ab_seed_used=True),
            "metrics.ab_seed_used must be an integer",
        ),
        (
            lambda _report, guard: guard["policy"].update(min_gain=-1.0),
            "policy min_gain and tie_breaker_deadband must be finite and non-negative",
        ),
        (
            lambda _report, guard: guard["policy"].update(min_rel_gain=-1.0),
            "policy.min_rel_gain must be non-negative",
        ),
        (
            lambda _report, guard: guard["policy"].update(min_effect_lognll=-1.0),
            "policy.min_effect_lognll must be non-negative",
        ),
        (
            lambda _report, guard: guard["policy"].update(absolute_floor_ppl=-1.0),
            "policy.absolute_floor_ppl must be non-negative",
        ),
        (
            lambda _report, guard: guard["metrics"].pop("ab_point_estimates"),
            "metrics.ab_point_estimates is required",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_point_estimates"].update(
                tag="pre_edit"
            ),
            "ab_point_estimates.tag must be post_edit",
        ),
    ],
)
def test_strict_gain_rejects_malformed_numeric_and_scale_facts(
    mutation: Mutation,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    mutation(report, _variance_guard(report))

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report, _guard: report["variance"].pop("ab_test"),
            "variance.ab_test is required for ci_gain_met assurance",
        ),
        (
            lambda report, _guard: report["variance"]["ab_test"].update(
                point_estimates=None
            ),
            "metrics.ab_point_estimates must match variance.ab_test.point_estimates",
        ),
        (
            lambda _report, guard: guard["metrics"].update(ab_provenance=None),
            "metrics.ab_provenance is required",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_a"
            ].update(mode="wrong"),
            "condition_a.mode is invalid",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_b"
            ].update(status="no_scales"),
            "condition_b.status is invalid",
        ),
        (
            lambda _report, guard: guard["metrics"]["ab_provenance"][
                "condition_a"
            ].update(run_seed=True),
            "condition_a.run_seed must be an integer",
        ),
        (
            lambda _report, guard: guard["details"]["stats"].update(
                target_fingerprint="wrong"
            ),
            "details.stats.target_fingerprint must match A/B provenance",
        ),
        (
            lambda _report, guard: guard["details"]["stats"]["dataset_meta"].update(
                tokenizer_hash="wrong"
            ),
            "details.stats.dataset_meta.tokenizer_hash must match A/B provenance",
        ),
    ],
)
def test_strict_gain_rejects_missing_or_invalid_provenance_structures(
    mutation: Mutation,
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    mutation(report, _variance_guard(report))

    errors = _strict_errors(report)

    assert any(expected in error for error in errors), "\n".join(errors)
