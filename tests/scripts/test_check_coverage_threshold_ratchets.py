from __future__ import annotations

from scripts.coverage import check_coverage_thresholds as policy


def test_ratchet_can_raise_but_never_lower_behavioral_floor(monkeypatch) -> None:
    path = "scripts/evidence_packs/python/editing/streaming_pruning.py"
    monkeypatch.setitem(
        policy.COVERAGE_RATCHETS,
        path,
        policy.CoverageFloor(line=0.10, branch=0.20),
    )

    floor = policy._effective_floor(path, "behavioral")

    assert floor == policy.CoverageFloor(line=0.95, branch=0.90)

    monkeypatch.setitem(
        policy.COVERAGE_RATCHETS,
        path,
        policy.CoverageFloor(line=0.97, branch=0.94),
    )
    assert policy._effective_floor(path, "behavioral") == policy.CoverageFloor(
        line=0.97,
        branch=0.94,
    )


def test_live_ratchet_can_raise_local_floor_without_erasing_receipt_metadata(
    monkeypatch,
) -> None:
    path = "src/invarlock/guards/exact_svd.py"
    monkeypatch.setitem(
        policy.COVERAGE_RATCHETS,
        path,
        policy.CoverageFloor(line=0.90, branch=0.80),
    )

    assert policy._effective_floor(path, "live_backend") == policy.CoverageFloor(
        line=0.90,
        branch=0.80,
    )
    assert path in policy.LIVE_HARDWARE_CLOSURE_REQUIREMENTS
