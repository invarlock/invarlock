from types import SimpleNamespace

import pytest
import torch

from invarlock.guards.variance import VarianceGuard


def test_store_calibration_batches_pairing_mismatch_raises():
    g = VarianceGuard()
    # Configure pairing reference in run context
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

    # Provide observed ids that do NOT match expected subset (force mismatch)
    batches = [
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "metadata": {"window_ids": ["preview::9"]},
        },
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "metadata": {"window_ids": ["preview::10"]},
        },
    ]

    with pytest.raises(RuntimeError):
        g._store_calibration_batches(batches)
