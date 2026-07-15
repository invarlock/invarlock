import torch

from invarlock.guards.variance import VarianceGuard


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def test_finalize_warns_when_disabled_despite_gate_approval(monkeypatch):
    g = VarianceGuard()
    g._prepared = True
    g._post_edit_evaluated = True
    g._policy["min_gain"] = 0.0
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.0

    # Gate approves but enable returns False and state remains disabled
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "criteria_met"))
    monkeypatch.setattr(g, "enable", lambda model: False)
    # Ensure not enabled
    g._enabled = False

    out = g.finalize(DummyModel())
    assert out["passed"] is True  # conservative disable is a warning, not error
    warns = "\n".join(out.get("warnings", []))
    assert "disabled despite A/B gate approval" in warns


def test_finalize_degradation_error_when_disabled(monkeypatch):
    g = VarianceGuard()
    g._prepared = True
    g._post_edit_evaluated = True
    g._policy["min_gain"] = 0.0
    g._enabled = False
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.0
    g._final_ppl = 101.0  # rise > 0.5 triggers error when disabled
    g._ab_windows_used = 2
    g._ab_seed_used = 123

    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (False, "criteria_not_met"))

    out = g.finalize(DummyModel())
    assert out["passed"] is False
    errs = "\n".join(out.get("errors", []))
    assert "Primary-metric rise" in errs and "> 0.5" in errs
