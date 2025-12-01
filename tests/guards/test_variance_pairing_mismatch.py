import pytest
import torch

from invarlock.guards.variance import VarianceGuard


def test_store_calibration_batches_pairing_mismatch_raises():
    g = VarianceGuard()
    # Set a baseline pairing reference that won't match observed ids
    g._pairing_reference = g._normalize_pairing_ids("preview", [0, 1])
    g._pairing_digest = "deadbeef"

    # Build simple batches with no explicit window ids → observed are ["0", "1"]
    batches = [torch.ones(1, 2), torch.ones(1, 2)]

    with pytest.raises(RuntimeError, match="pairing mismatch"):
        g._store_calibration_batches(batches)
