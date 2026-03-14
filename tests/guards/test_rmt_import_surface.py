from __future__ import annotations

import invarlock.guards.rmt as runtime_rmt
import invarlock.guards.rmt_legacy as legacy_rmt


def test_runtime_module_exposes_guard_surface_only() -> None:
    assert hasattr(runtime_rmt, "RMTGuard")
    assert hasattr(runtime_rmt, "get_rmt_policy")
    assert not hasattr(runtime_rmt, "rmt_detect")
    assert not hasattr(runtime_rmt, "capture_baseline_mp_stats")


def test_legacy_module_exposes_weight_analysis_helpers() -> None:
    assert hasattr(legacy_rmt, "rmt_detect")
    assert hasattr(legacy_rmt, "capture_baseline_mp_stats")
    assert hasattr(legacy_rmt, "clip_full_svd")
