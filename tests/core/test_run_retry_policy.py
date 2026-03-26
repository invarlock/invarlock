from __future__ import annotations

from invarlock.core.run_retry_policy import (
    apply_mask_only_head_autotune,
    build_retry_result_summary,
)


def test_build_retry_result_summary_collects_failed_gates() -> None:
    out = build_retry_result_summary(
        {"primary_metric_acceptable": False, "drift_ok": True}
    )

    assert out == {
        "passed": False,
        "failures": ["primary_metric_acceptable"],
        "validation": {
            "primary_metric_acceptable": False,
            "drift_ok": True,
        },
    }


def test_apply_mask_only_head_autotune_updates_search_state() -> None:
    updated, adjustment = apply_mask_only_head_autotune(
        {
            "heads": {
                "mask_only": True,
                "_auto_search": {
                    "keep_low": 0,
                    "keep_high": 8,
                    "keep_current": 4,
                },
            }
        },
        {"primary_metric_acceptable": False, "drift_ok": True},
    )

    assert adjustment == {
        "global_k": 6,
        "keep_low": 4,
        "keep_high": 8,
        "failed_gate_count": 1,
    }
    assert updated["heads"]["global_k"] == 6
    assert updated["heads"]["_auto_search"] == {
        "keep_low": 4,
        "keep_high": 8,
        "keep_current": 6,
    }


def test_apply_mask_only_head_autotune_noops_without_supported_section() -> None:
    original = {"heads": {"mask_only": False}}
    updated, adjustment = apply_mask_only_head_autotune(
        original,
        {"primary_metric_acceptable": False},
    )

    assert updated == original
    assert adjustment is None


def test_apply_mask_only_head_autotune_fails_closed_on_bad_values() -> None:
    updated, adjustment = apply_mask_only_head_autotune(
        {
            "head_budget": {
                "mask_only": True,
                "_auto_search": {"keep_low": "bad", "keep_high": 8},
            }
        },
        {"primary_metric_acceptable": False},
    )

    assert updated == {
        "head_budget": {
            "mask_only": True,
            "_auto_search": {"keep_low": "bad", "keep_high": 8},
        }
    }
    assert adjustment is None
