"""Guard-overhead and release-window planning helpers."""

from __future__ import annotations

import math
from typing import Any

from rich.console import Console

RELEASE_BUFFER_FRACTION = 0.12
RELEASE_MIN_WINDOWS_PER_ARM = 200
RELEASE_CALIBRATION_MIN = 16
RELEASE_CALIBRATION_MAX = 24


def plan_release_windows(
    capacity: dict[str, Any],
    *,
    requested_preview: int,
    requested_final: int,
    max_calibration: int,
    console: Console | None = None,
    event_fn: Any | None = None,
) -> dict[str, Any]:
    """Derive release-tier window plan based on dataset capacity."""
    available_unique = int(capacity.get("available_unique", 0))
    available_nonoverlap = int(capacity.get("available_nonoverlap", 0))
    total_tokens = int(capacity.get("total_tokens", 0))
    dedupe_rate = float(capacity.get("dedupe_rate", 0.0))
    candidate_unique = capacity.get("candidate_unique")
    if candidate_unique is not None and int(candidate_unique) > 0:
        effective_unique = int(candidate_unique)
    else:
        effective_unique = available_unique
    candidate_limit = capacity.get("candidate_limit")

    target_per_arm = int(min(requested_preview, requested_final))
    if target_per_arm <= 0:
        target_per_arm = requested_preview or requested_final or 1

    max_calibration = max(0, int(max_calibration or 0))
    if max_calibration > 0:
        calibration_windows = max(
            RELEASE_CALIBRATION_MIN,
            min(RELEASE_CALIBRATION_MAX, max_calibration // 10),
        )
    else:
        calibration_windows = RELEASE_CALIBRATION_MIN
    calibration_windows = min(calibration_windows, available_unique)

    buffer_windows = math.ceil(RELEASE_BUFFER_FRACTION * effective_unique)
    reserve_windows = min(effective_unique, calibration_windows + buffer_windows)
    available_for_eval = max(0, effective_unique - reserve_windows)
    actual_per_arm_raw = available_for_eval // 2

    coverage_ok = actual_per_arm_raw >= RELEASE_MIN_WINDOWS_PER_ARM
    if not coverage_ok:
        raise RuntimeError(
            "Release profile capacity insufficient: "
            f"available_unique={available_unique}, reserve={reserve_windows} "
            f"(calibration={calibration_windows}, buffer={buffer_windows}), "
            f"usable_per_arm={actual_per_arm_raw}, "
            f"requires ≥{RELEASE_MIN_WINDOWS_PER_ARM} per arm."
        )

    actual_per_arm = min(target_per_arm, actual_per_arm_raw)

    if console and event_fn is not None:
        candidate_msg = ""
        if candidate_unique is not None:
            candidate_msg = f", candidate_unique={int(candidate_unique)}" + (
                f"/{int(candidate_limit)}" if candidate_limit is not None else ""
            )
        event_fn(
            console,
            "METRIC",
            "Release window capacity:"
            f" unique={available_unique}, reserve={reserve_windows} "
            f"(calib {calibration_windows}, buffer {buffer_windows}), "
            f"usable={available_for_eval}, "
            f"per-arm raw={actual_per_arm_raw} → selected {actual_per_arm} "
            f"(target {target_per_arm}{candidate_msg})",
            emoji="📏",
            profile="release",
        )
        if actual_per_arm < target_per_arm:
            event_fn(
                console,
                "WARN",
                f"Adjusted per-arm windows down from {target_per_arm} to {actual_per_arm} based on capacity.",
                emoji="⚠️",
                profile="release",
            )

    plan = {
        "profile": "release",
        "requested_preview": int(requested_preview),
        "requested_final": int(requested_final),
        "target_per_arm": target_per_arm,
        "min_per_arm": RELEASE_MIN_WINDOWS_PER_ARM,
        "actual_preview": int(actual_per_arm),
        "actual_final": int(actual_per_arm),
        "actual_per_arm_raw": int(actual_per_arm_raw),
        "coverage_ok": coverage_ok,
        "capacity": {
            "total_tokens": total_tokens,
            "available_nonoverlap": available_nonoverlap,
            "available_unique": available_unique,
            "effective_unique": effective_unique,
            "dedupe_rate": dedupe_rate,
            "calibration": calibration_windows,
            "buffer_fraction": RELEASE_BUFFER_FRACTION,
            "buffer_windows": buffer_windows,
            "reserve_windows": reserve_windows,
            "usable_after_reserve": available_for_eval,
        },
    }
    if candidate_unique is not None:
        plan["capacity"]["candidate_unique"] = int(candidate_unique)
    if candidate_limit is not None:
        plan["capacity"]["candidate_limit"] = int(candidate_limit)

    return plan
