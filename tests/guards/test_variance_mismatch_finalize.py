import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 1, bias=False)


def _base(policy):
    g = VarianceGuard(policy=policy)
    g._prepared = True
    g._post_edit_evaluated = True
    return g


def test_finalize_mismatch_warning(monkeypatch):
    # Force should_enable=True but enabled_after_ab remains False due to enable() returning False
    g = _base(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "scope": "ffn",
            "max_calib": 100,
        }
    )
    g._predictive_gate_state.update({"evaluated": True, "passed": True})
    g._ab_gain = 0.2
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 80.0
    g._ratio_ci = (0.7, 0.95)
    g._enabled = False

    # Monkeypatch enable() to fail and not change state
    monkeypatch.setattr(g, "enable", lambda model: False)

    result = g.validate(TinyModel(), adapter=None, context={})
    assert result["passed"] is True
    # Should include a warning about disabled despite approval
    assert any("A/B gate approval" in w for w in result.get("warnings", []))


def test_finalize_mismatch_error(monkeypatch):
    # Force should_enable=False but enabled_after_ab remains True due to disable() not changing _enabled
    g = _base(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.01,
            "scope": "ffn",
            "max_calib": 100,
        }
    )
    g._predictive_gate_state.update({"evaluated": True, "passed": True})
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 100.0
    g._ratio_ci = (0.99, 1.0)  # hi > 1 - min_rel_gain, so should_enable False
    g._enabled = True

    def noop_disable(model):
        # Leave _enabled unchanged
        return None

    monkeypatch.setattr(g, "disable", noop_disable)

    result = g.validate(TinyModel(), adapter=None, context={})
    # Implementation forces alignment by setting enabled_after_ab to False;
    # ensure validation returns a result and does not error
    assert isinstance(result, dict)
