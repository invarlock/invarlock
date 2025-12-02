import math
from collections.abc import Iterable

import pytest

from invarlock.cli.overhead_utils import _extract_pm_snapshot_for_overhead as extract
from invarlock.reporting.validate import validate_guard_overhead


def _make_core_like_report_with_windows(
    logloss: Iterable[float], token_counts: Iterable[int]
):
    class _CoreLike:
        def __init__(self, evaluation_windows):
            self.metrics = {}
            self.evaluation_windows = evaluation_windows

    evaluation_windows = {
        "preview": {"logloss": list(logloss), "token_counts": list(token_counts)},
        "final": {"logloss": list(logloss), "token_counts": list(token_counts)},
    }
    return _CoreLike(evaluation_windows)


def test_extract_pm_snapshot_prefers_existing_primary_metric() -> None:
    core_like = _make_core_like_report_with_windows([3.2, 3.3], [10, 20])
    core_like.metrics = {
        "primary_metric": {"kind": "ppl_causal", "preview": 25.0, "final": 26.0}
    }

    pm = extract(core_like, kind="ppl_causal")
    assert isinstance(pm, dict)
    assert isinstance(pm.get("final"), int | float)
    assert math.isfinite(float(pm["final"]))


def test_extract_pm_snapshot_skips_invalid_primary_metric_and_uses_windows() -> None:
    core_like = _make_core_like_report_with_windows([3.2, 3.3], [10, 20])
    # Present but non-finite primary metric; helper should fall back to windows
    core_like.metrics = {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 25.0,
            "final": float("nan"),
        }
    }

    pm = extract(core_like, kind="ppl_causal")
    assert isinstance(pm, dict)
    assert isinstance(pm.get("final"), int | float)
    assert math.isfinite(float(pm["final"]))


def test_extract_pm_snapshot_uses_dict_report_when_available() -> None:
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [3.2, 3.3], "token_counts": [10, 20]},
            "final": {"logloss": [3.1, 3.4], "token_counts": [10, 20]},
        }
    }

    pm = extract(report, kind="ppl_causal")
    assert isinstance(pm, dict)
    assert isinstance(pm.get("final"), int | float)
    assert math.isfinite(float(pm["final"]))


def test_extract_pm_snapshot_returns_none_when_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenReport:
        @property
        def metrics(self) -> dict[str, object]:
            raise RuntimeError("broken metrics")

        @property
        def evaluation_windows(self) -> dict[str, object]:
            raise RuntimeError("broken windows")

    # Make the primary metric helper raise when called from dict path
    def _boom(*args, **kwargs):
        raise RuntimeError("cannot compute")

    monkeypatch.setattr(
        "invarlock.eval.primary_metric.compute_primary_metric_from_report",
        _boom,
        raising=False,
    )

    broken = BrokenReport()
    pm = extract(broken, kind="ppl_causal")
    assert pm is None


def test_validate_guard_overhead_with_extracted_pm_evaluates() -> None:
    """Ensure structured validator gets finite ratio from extracted PMs."""
    pm_bare = {"kind": "ppl_causal", "preview": 25.0, "final": 26.0}
    pm_guarded = {"kind": "ppl_causal", "preview": 25.3, "final": 26.1}

    bare_struct = {"metrics": {"primary_metric": pm_bare}}
    guarded_struct = {"metrics": {"primary_metric": pm_guarded}}

    res = validate_guard_overhead(bare_struct, guarded_struct, overhead_threshold=0.5)
    assert hasattr(res, "metrics")
    ratio = res.metrics.get("overhead_ratio")
    assert isinstance(ratio, float)
    assert math.isfinite(ratio)


def test_extract_pm_snapshot_uses_windows_when_metric_missing() -> None:
    core_like = _make_core_like_report_with_windows([3.2, 3.3], [10, 20])

    pm = extract(core_like, kind="ppl_causal")
    assert isinstance(pm, dict)
    assert isinstance(pm.get("final"), int | float)
    assert math.isfinite(float(pm["final"]))
