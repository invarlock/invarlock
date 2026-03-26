from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_retry_result_summary(
    validation: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Build a stable retry summary from evaluation-report validation flags."""
    validation_map = dict(validation or {})
    failures = [str(key) for key, value in validation_map.items() if not value]
    return {
        "passed": not failures,
        "failures": failures,
        "validation": validation_map,
    }


def apply_mask_only_head_autotune(
    edit_config: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Advance mask-only head-search state after a failed validation attempt."""
    updated = dict(edit_config or {})
    validation_map = dict(validation or {})

    for section_key in ("heads", "head_budget", "head_budgets"):
        head_section = updated.get(section_key)
        if not isinstance(head_section, Mapping):
            continue
        search = head_section.get("_auto_search")
        if not (isinstance(search, Mapping) and head_section.get("mask_only")):
            continue
        try:
            keep_low = int(search.get("keep_low", 0))
            keep_high = int(search.get("keep_high", search.get("total_heads", 0)))
            keep_current = int(search.get("keep_current", keep_high))
        except (TypeError, ValueError):
            return updated, None

        # A failed gate always reduces pruning aggressiveness for the next attempt.
        keep_low = max(keep_low, keep_current)
        next_keep = int((keep_low + keep_high + 1) // 2)

        next_search = dict(search)
        next_search.update(
            {
                "keep_low": keep_low,
                "keep_high": keep_high,
                "keep_current": next_keep,
            }
        )
        next_head_section = dict(head_section)
        next_head_section["_auto_search"] = next_search
        next_head_section["global_k"] = next_keep
        updated[section_key] = next_head_section

        return updated, {
            "global_k": next_keep,
            "keep_low": keep_low,
            "keep_high": keep_high,
            "failed_gate_count": len(
                [key for key, value in validation_map.items() if not value]
            ),
        }

    return updated, None
