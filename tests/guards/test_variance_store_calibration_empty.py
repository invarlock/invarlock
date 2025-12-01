from types import SimpleNamespace

from invarlock.guards.variance import VarianceGuard


def test_store_calibration_batches_empty_observed_ok():
    g = VarianceGuard()
    # Pairing reference present but we pass zero observed; should not raise
    report = SimpleNamespace(
        meta={},
        context={
            "pairing_baseline": {
                "preview": {"window_ids": ["1", "2", "3"]},
                "final": {"window_ids": []},
            }
        },
        edit={"name": "structured", "deltas": {"params_changed": 0}},
    )
    g.set_run_context(report)
    g._store_calibration_batches([])
    ctx = g._stats.get("calibration", {})
    # expected_digest exists, observed_digest is None, counts reflect empty
    assert (
        ctx.get("count") == 0
        and ctx.get("expected_digest")
        and ctx.get("observed_digest") is None
    )
