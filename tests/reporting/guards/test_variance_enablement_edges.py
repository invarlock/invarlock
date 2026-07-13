from __future__ import annotations

import invarlock.reporting.guards_common as guards_common_mod
import invarlock.reporting.verify_check_helpers_consistency as verify_helpers_mod


def _report(
    *,
    predictive_gate: object,
    ab_test: object,
    min_effect: object = "bad",
) -> dict[str, object]:
    return {
        "resolved_policy": {"variance": {"min_effect_lognll": min_effect}},
        "variance": {
            "enabled": True,
            "predictive_gate": predictive_gate,
            "ab_test": ab_test,
        },
    }


def test_baseline_guard_payload_edge_branches() -> None:
    assert guards_common_mod._baseline_guard_payload(None, "variance") == {}  # noqa: SLF001
    assert (
        guards_common_mod._baseline_guard_payload(  # noqa: SLF001
            {
                "guards": [
                    "not-a-guard",
                    {"name": "spectral", "metrics": {"sigma": 1.0}},
                    {"name": "variance"},
                ]
            },
            "variance",
        )
        == {}
    )
    assert guards_common_mod._baseline_guard_payload(  # noqa: SLF001
        {"guards": [{"name": "variance", "metrics": {"enabled": True}}]},
        "variance",
    ) == {"enabled": True}

    class BrokenBaseline(dict):
        def get(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("broken")

    assert (
        guards_common_mod._baseline_guard_payload(  # noqa: SLF001
            BrokenBaseline(),
            "variance",
        )
        == {}
    )


def test_validate_variance_enablement_edge_branches() -> None:
    cases = [
        (
            _report(predictive_gate={}, ab_test={}),
            "variance mitigation validation requires predictive_gate evidence.",
        ),
        (
            _report(
                predictive_gate={"passed": True, "mean_delta": None, "delta_ci": []},
                ab_test={
                    "seed": 1,
                    "windows_used": 1,
                    "provenance": {"window_ids": [1]},
                },
            ),
            "variance mitigation validation requires finite predictive_gate.mean_delta.",
        ),
        (
            _report(
                predictive_gate={
                    "passed": True,
                    "mean_delta": 0.0,
                    "delta_ci": [-0.2, -0.1],
                },
                ab_test={
                    "seed": 1,
                    "windows_used": 1,
                    "provenance": {"window_ids": [1]},
                },
            ),
            "variance.predictive_gate.mean_delta must be negative",
        ),
        (
            _report(
                predictive_gate={
                    "passed": True,
                    "mean_delta": -0.001,
                    "delta_ci": [-0.2, -0.1],
                },
                ab_test={
                    "seed": 1,
                    "windows_used": 1,
                    "provenance": {"window_ids": [1]},
                },
                min_effect=0.01,
            ),
            "variance.predictive_gate.mean_delta does not meet",
        ),
        (
            _report(
                predictive_gate={"passed": True, "mean_delta": -0.2, "ci": ["x", -0.1]},
                ab_test={
                    "seed": 1,
                    "windows_used": 1,
                    "provenance": {"window_ids": [1]},
                },
            ),
            "variance mitigation validation requires finite predictive_gate.delta_ci.",
        ),
        (
            _report(
                predictive_gate={
                    "passed": True,
                    "mean_delta": -0.2,
                    "delta_ci": [-0.1, -0.2],
                },
                ab_test={
                    "seed": 1,
                    "windows_used": 1,
                    "provenance": {"window_ids": [1]},
                },
            ),
            "variance.predictive_gate.delta_ci lower bound exceeds upper bound.",
        ),
        (
            _report(
                predictive_gate={
                    "passed": True,
                    "mean_delta": -0.2,
                    "delta_ci": [-0.2, -0.001],
                },
                ab_test={
                    "seed": 1,
                    "windows_used": 1,
                    "provenance": {"window_ids": [1]},
                },
                min_effect=0.01,
            ),
            "variance.predictive_gate.delta_ci upper bound does not meet",
        ),
        (
            _report(
                predictive_gate={
                    "passed": True,
                    "mean_delta": -0.2,
                    "delta_ci": [-0.2, -0.1],
                },
                ab_test={},
            ),
            "variance mitigation validation requires variance.ab_test evidence.",
        ),
        (
            _report(
                predictive_gate={
                    "passed": True,
                    "mean_delta": -0.2,
                    "delta_ci": [-0.2, -0.1],
                },
                ab_test={"seed": "", "windows_used": 0, "provenance": {"nested": []}},
            ),
            "variance mitigation validation requires variance.ab_test.seed.",
        ),
    ]

    for report, expected in cases:
        errors = verify_helpers_mod._validate_variance_enablement(report)  # noqa: SLF001
        assert any(expected in error for error in errors)


def test_collect_provenance_window_ids_recurses_through_lists_and_mappings() -> None:
    assert verify_helpers_mod._collect_provenance_window_ids(  # noqa: SLF001
        {
            "outer": [
                {"window_ids": [1]},
                {"nested": {"window_ids": [2]}},
                {"ignored": object()},
            ]
        }
    ) == [1, 2]
    assert verify_helpers_mod._collect_provenance_window_ids(  # noqa: SLF001
        [{"window_ids": [3]}, "ignored"]
    ) == [3]


def test_validate_variance_enablement_accepts_restored_success() -> None:
    report = _report(
        predictive_gate={
            "passed": True,
            "reason": "ci_gain_met",
            "mean_delta": -0.2,
            "delta_ci": [-0.3, -0.1],
        },
        ab_test={
            "seed": 1,
            "windows_used": 1,
            "provenance": {"window_ids": [1]},
        },
    )
    variance = report["variance"]
    assert isinstance(variance, dict)
    variance.update(
        enabled=False,
        ve_enabled_during_validation=True,
        subject_restored_after_ab=True,
        met_threshold=True,
    )

    assert verify_helpers_mod._validate_variance_enablement(report) == []  # noqa: SLF001


def test_validate_variance_enablement_rejects_unbound_restored_success() -> None:
    report = _report(
        predictive_gate={
            "passed": True,
            "reason": "ci_gain_met",
            "mean_delta": -0.2,
            "delta_ci": [-0.3, -0.1],
        },
        ab_test={
            "seed": 1,
            "windows_used": 1,
            "provenance": {"window_ids": [1]},
        },
    )
    variance = report["variance"]
    assert isinstance(variance, dict)
    variance["enabled"] = False

    errors = verify_helpers_mod._validate_variance_enablement(report)  # noqa: SLF001

    assert any("ve_enabled_during_validation=true" in error for error in errors)
    assert any("subject_restored_after_ab=true" in error for error in errors)
    assert any("met_threshold=true" in error for error in errors)
