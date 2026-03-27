from __future__ import annotations

import importlib.util

import invarlock.guards.rmt as runtime_rmt
import invarlock.guards.rmt_analysis as rmt_analysis
import invarlock.guards.rmt_detection as rmt_detection
import invarlock.guards.rmt_math as rmt_math


def test_runtime_module_exposes_guard_surface_only() -> None:
    assert hasattr(runtime_rmt, "RMTGuard")
    assert hasattr(runtime_rmt, "get_rmt_policy")
    assert not hasattr(runtime_rmt, "rmt_detect")
    assert not hasattr(runtime_rmt, "capture_baseline_mp_stats")


def test_analysis_module_exposes_weight_analysis_helpers() -> None:
    assert hasattr(rmt_analysis, "capture_baseline_mp_stats")
    assert hasattr(rmt_analysis, "layer_svd_stats")
    assert hasattr(rmt_analysis, "analyze_weight_distribution")


def test_detection_module_exposes_detection_helpers() -> None:
    assert hasattr(rmt_detection, "rmt_detect")
    assert hasattr(rmt_detection, "rmt_detect_report")
    assert hasattr(rmt_detection, "rmt_detect_with_names")


def test_math_module_exposes_pure_rmt_helpers() -> None:
    assert hasattr(rmt_math, "mp_bulk_edge")
    assert hasattr(rmt_math, "mp_bulk_edges")
    assert hasattr(rmt_math, "rmt_growth_ratio")
    assert hasattr(rmt_math, "within_deadband")
    assert hasattr(rmt_math, "clip_full_svd")


def test_legacy_module_is_gone() -> None:
    assert importlib.util.find_spec("invarlock.guards.rmt_legacy") is None
