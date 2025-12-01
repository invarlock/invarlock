import torch

from invarlock.guards import rmt as R


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Name must end with an allowed suffix for analysis
        self.layer = torch.nn.Linear(4, 4)

    def named_modules(self, memo=None, prefix=""):
        # Yield a module whose name ends with the allowed suffix
        yield "block.mlp.c_fc", self.layer


def test_rmt_detect_deadband_partial_baseline_branch():
    model = TinyModel()
    # Baseline sigma equal to current sigma ensures deadband branch executes
    with torch.no_grad():
        s = torch.linalg.svdvals(model.layer.weight.float().cpu())[0].item()
    baseline_sigmas = {"block.mlp.c_fc": s}
    # No baseline_mp_stats to force the 'elif deadband > 0.0 and baseline_sigmas' branch
    out = R.rmt_detect(
        model,
        threshold=1.1,
        detect_only=True,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=None,
        deadband=0.05,
    )
    assert isinstance(out, dict)
    assert "per_layer" in out and isinstance(out["per_layer"], list)
