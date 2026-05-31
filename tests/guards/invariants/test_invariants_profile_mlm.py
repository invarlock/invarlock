from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.invariants import InvariantsGuard


class BertPredModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fake BERT-like structure with cls.predictions.decoder
        decoder = SimpleNamespace(weight=nn.Parameter(nn.Linear(2, 2).weight))
        predictions = SimpleNamespace(decoder=decoder)
        self.cls = SimpleNamespace(predictions=predictions)
        self.config = SimpleNamespace(model_type="bert", is_decoder=False)


def test_profile_mlm_mask_alignment():
    model = BertPredModel()
    g = InvariantsGuard()
    prep = g.prepare(
        model,
        adapter=None,
        calib=None,
        policy={"profile_checks": ["mlm_mask_alignment"]},
    )
    assert prep["ready"] is True
    assert g.baseline_checks.get("profile::mlm_mask_alignment") is True
