from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_store_calibration_batches_expected_ids_match():
    g = VarianceGuard()
    # Set pairing baseline so expected ids are known
    report = SimpleNamespace(
        meta={},
        context={
            "pairing_baseline": {
                "preview": {"window_ids": ["1", "2"]},
                "final": {"window_ids": []},
            }
        },
        edit={"name": "structured", "deltas": {"params_changed": 0}},
    )
    g.set_run_context(report)
    # Provide matched observed ids with normalized prefixes
    batches = [
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "metadata": {"window_ids": ["preview::1"]},
        },
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "metadata": {"window_ids": ["preview::2"]},
        },
    ]
    # Should not raise and should populate calibration stats
    g._store_calibration_batches(batches)
    ctx = g._stats.get("calibration", {})
    assert ctx.get("count") == 2 and isinstance(ctx.get("observed_digest"), str)


def test_resolve_target_modules_rejections_and_tap_mismatch():
    class BadProj(nn.Module):
        def __init__(self):
            super().__init__()
            # 1D weight simulates unsupported type
            self.weight = nn.Parameter(torch.zeros(3))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(c_proj=BadProj())
            self.mlp = SimpleNamespace(c_proj=BadProj())

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[Block()])

    # Configure a tap pattern that won't match actual normalized names
    g = VarianceGuard(policy={"tap": ["transformer.h.99.mlp.c_proj"], "scope": "both"})
    targets = g._resolve_target_modules(Model(), adapter=None)
    assert isinstance(targets, dict) and len(targets) == 0
    rej = g._stats.get("target_resolution", {}).get("rejected", {})
    # Ensure we recorded some rejections (tap_mismatch or unsupported_type)
    assert any(key in rej for key in ("tap_mismatch", "unsupported_type"))
