import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # minimal module to satisfy finalize prepared checks via manual flags
        self.w = nn.Linear(2, 2, bias=False)


def test_evaluate_ab_gate_min_effect_lognll_branch():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "min_effect_lognll": 0.2,
            "predictive_gate": False,
        }
    )
    # Set A/B results such that log gain is below threshold
    g._ab_gain = 0.1
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 95.0
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and "below_min_effect_lognll" in reason


def test_finalize_secondary_validation_ppl_rise_error():
    g = VarianceGuard(
        policy={
            "scope": "ffn",
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    # Mark prepared and skip post-edit evaluation work
    g._prepared = True
    g._post_edit_evaluated = True
    g._enabled = False
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.8
    # Force final ppl significantly higher to trigger error
    g._final_ppl = 101.0
    # Inflate attempt counts to exercise warnings
    g._enable_attempt_count = 4
    g._disable_attempt_count = 5
    res = g.finalize(TinyModel())
    assert res["passed"] is False
    errs = "\n".join(res.get("errors", []))
    assert "Primary-metric rise" in errs
    warns = "\n".join(res.get("warnings", []))
    assert "Multiple enable attempts" in warns and "Multiple disable attempts" in warns
