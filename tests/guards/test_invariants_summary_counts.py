import torch
import torch.nn as nn

from invarlock.guards.invariants import InvariantsGuard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)
        self.p = nn.Parameter(torch.ones(2, 2))


def test_invariants_summary_fatal_and_warning_counts():
    # Lenient mode: non_finite is fatal; invariant_violation is warning
    model = Tiny()
    g = InvariantsGuard(strict_mode=False)
    g.prepare(model, adapter=None, calib=None, policy={})
    # Introduce non_finite (fatal)
    with torch.no_grad():
        model.p[0, 0] = float("nan")
    # Also modify structure to trigger invariant_violation by adding a module
    model.extra = nn.Linear(1, 1)
    out = g.finalize(model)
    assert out.metrics["fatal_violations"] >= 1
    assert out.metrics["warning_violations"] >= 1
    assert (
        out.metrics["violations_found"]
        >= (out.metrics["fatal_violations"] + out.metrics["warning_violations"]) - 1
    )
