from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 2, bias=False)


def test_set_run_context_pairing_and_monitor_only():
    g = VarianceGuard()
    # Prime some scales to verify monitor_only clears them
    g._scales = {"transformer.h.0.mlp.c_proj": 0.9}
    report = SimpleNamespace(
        meta={"model_id": "m", "seed": 7},
        context={
            "dataset_meta": {"dataset_hash": "h", "tokenizer_hash": "t"},
            "pairing_baseline": {
                "preview": {"window_ids": ["a", "b"]},
                "final": {"window_ids": ["c"]},
            },
        },
        edit={"name": "structured", "deltas": {"params_changed": 0}},
    )
    g.set_run_context(report)
    # Pairing reference digest populated
    pr = g._stats.get("pairing_reference", {})
    assert pr.get("count") == 3 and isinstance(pr.get("digest"), str)
    # Monitor-only enforced and scales cleared
    assert g._monitor_only is True and g._scales == {}

    # Now with no pairing info, pairing_reference should be removed
    report2 = SimpleNamespace(meta={}, context={}, edit={})
    g.set_run_context(report2)
    assert "pairing_reference" not in g._stats


def test_tensorize_and_extract_window_ids():
    g = VarianceGuard()
    # Materialize nested structures
    batch = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": np.ones((1, 3), dtype=np.int64),
        "metadata": {"window_id": "w0"},
        "misc": [torch.tensor([1.0])],
    }
    mat = g._materialize_batch(batch)
    tensored = g._tensorize_calibration_batches([mat])[0]
    assert isinstance(tensored["input_ids"], torch.Tensor)
    assert isinstance(tensored["attention_mask"], torch.Tensor)
    # Extract window ids from metadata
    window_ids = g._extract_window_ids([tensored])
    assert window_ids == ["w0"]


def test_prepare_monitor_mode_insufficient_coverage():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "calibration": {"windows": 3, "min_coverage": 3, "seed": 5},
        }
    )

    # Use a tiny model with transformer.h so targets resolve
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            blk = nn.Module()
            blk.attn = nn.Module()
            blk.attn.c_proj = nn.Linear(2, 2, bias=False)
            blk.mlp = nn.Module()
            blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
            self.transformer.h = nn.ModuleList([blk])

        def forward(self, x):
            return self.transformer.h[0].mlp.c_proj(
                self.transformer.h[0].attn.c_proj(x)
            )

    # Single batch → coverage < min_coverage
    batches = [torch.ones(1, 2)]
    # Drive internal calibration pass directly to avoid target resolution noise
    g._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    g._store_calibration_batches(batches)
    g._calibration_stats = {
        "requested": 3,
        "coverage": 0,
        "min_coverage": 3,
        "seed": 5,
        "status": "pending",
        "tag": "t",
    }
    g._evaluate_calibration_pass(
        M(), g._calibration_batches, min_coverage=3, calib_seed=5, tag="t"
    )
    # Status should be 'insufficient' or remain pending with predictive gate set
    assert g._calibration_stats.get("status") in {"insufficient", "pending"}
    pg = g._stats.get("predictive_gate", {})
    assert pg.get("reason") in {
        "insufficient_coverage",
        "no_scales",
        "ve_enable_failed",
        "no_calibration",
    }
    # condition_b provenance should be present when not evaluated
    ab = g._stats.get("ab_provenance", {})
    assert "condition_a" in ab
    # condition_b may be 'not_evaluated' depending on coverage path
    if "condition_b" in ab:
        assert ab["condition_b"].get("status") in {
            "not_evaluated",
            "evaluated",
            "no_scales",
        }
