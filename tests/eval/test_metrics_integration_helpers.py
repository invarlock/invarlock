import torch

from invarlock.eval import metrics as M


def test_analyze_spectral_and_rmt_changes_smoke():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

    before = Tiny()
    after = Tiny()
    with torch.no_grad():
        after.fc.weight.add_(0.01)

    spec = M.analyze_spectral_changes(before, after, scope="ffn")
    rmt = M.analyze_rmt_changes(before, after)
    assert isinstance(spec, dict) and isinstance(rmt, dict)
    # The summaries contain keys like 'layers_analyzed' or 'stability_ratio'
    assert "layers_analyzed" in spec or "error" in spec
    assert "stability_ratio" in rmt or "error" in rmt
