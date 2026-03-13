from __future__ import annotations

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_prepare import prepare_guard


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


def test_prepare_guard_policy_update_handles_none_min_effect_absolute_floor_and_empty_focus(
    monkeypatch,
) -> None:
    guard = VarianceGuard(policy={"scope": "both", "max_calib": 20, "min_gain": 0.0})
    model = _TinyModel()

    monkeypatch.setattr(
        guard,
        "_resolve_target_modules",
        lambda _model, _adapter=None: {"transformer.h.0.mlp.c_proj": object()},
    )
    monkeypatch.setattr(guard, "_compute_variance_scales", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(guard, "_store_calibration_batches", lambda _batches: None)

    result = prepare_guard(
        guard,
        model,
        adapter=None,
        calib=None,
        policy={
            "min_effect_lognll": None,
            "absolute_floor_ppl": 0.2,
            "target_modules": ["   ", 123],
        },
    )

    assert result["ready"] is True
    assert guard._focus_modules == set()
    assert guard.ABSOLUTE_FLOOR == 0.2
    assert guard._policy["min_effect_lognll"] is None
