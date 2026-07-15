import torch

from invarlock.eval import metrics as M


def test_analyze_spectral_changes_smoke():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

    before = Tiny()
    after = Tiny()
    with torch.no_grad():
        after.fc.weight.add_(0.01)

    spec = M.analyze_spectral_changes(before, after, scope="ffn")
    assert isinstance(spec, dict)
    # The summary contains keys like 'layers_analyzed'
    assert "layers_analyzed" in spec or "error" in spec
