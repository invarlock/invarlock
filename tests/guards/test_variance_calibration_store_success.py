from types import SimpleNamespace

import torch

from invarlock.guards.variance import VarianceGuard


def test_store_calibration_batches_matches_expected_ids_success():
    g = VarianceGuard()
    # Seed pairing reference via run context
    report = SimpleNamespace(
        meta={},
        context={
            "pairing_baseline": {
                "preview": {"window_ids": [1]},
                "final": {"window_ids": [2]},
            }
        },
        edit={},
    )
    g.set_run_context(report)

    # Build calibration batches whose observed IDs match expected subset
    batches = [
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "metadata": {"window_ids": ["preview::1"]},
        },
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "metadata": {"window_ids": ["final::2"]},
        },
    ]
    # Should not raise and should update calibration stats with expected_digest
    g._store_calibration_batches(batches)
    calib = g._stats.get("calibration", {})
    assert calib.get("observed_digest") and calib.get("expected_digest")
