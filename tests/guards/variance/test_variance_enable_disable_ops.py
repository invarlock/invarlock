import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, n=1, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d) for _ in range(n)])


def test_variance_enable_disable_applies_and_reverts_scales():
    model = TinyModel()
    g = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0, "max_calib": 0})
    # Resolve target modules normally to get canonical names
    targets = g._resolve_target_modules(model, adapter=None)
    assert targets, "Expected at least one target module"
    # Choose one target (mlp by default tap)
    name, module = next(iter(targets.items()))
    original = module.weight.detach().clone()

    # Prime guard state
    g._prepared = True
    g._target_modules = targets
    # Either exact name or block-form is acceptable; use exact
    g._scales = {name: 0.9}

    assert g.enable(model) is True
    assert g._enabled is True
    # Weight changed by scale
    after_enable = module.weight.detach().clone()
    assert not torch.allclose(original, after_enable)

    # Idempotent enable should not re-apply or error
    assert g.enable(model) is True

    # Disable returns weights to original
    assert g.disable(model) is True
    assert g._enabled is False
    after_disable = module.weight.detach().clone()
    assert torch.allclose(original, after_disable)


def test_variance_validate_payload_fields_present():
    model = TinyModel()
    # Predictive gate disabled to keep flow simple
    g = VarianceGuard(
        policy={
            "scope": "ffn",
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    targets = g._resolve_target_modules(model, adapter=None)
    g._prepared = True
    g._target_modules = targets
    # Provide scales but keep A/B inconclusive; finalize will not enable
    if targets:
        name = next(iter(targets.keys()))
        g._scales = {name: 0.95}
    # Set synthetic A/B results and CI for gate instrumentation
    g.set_ab_results(
        ppl_no_ve=100.0,
        ppl_with_ve=98.0,
        windows_used=3,
        seed_used=123,
        ratio_ci=(0.96, 0.99),
    )

    out = g.validate(model, adapter=None, context={})
    assert isinstance(out, dict)
    # Ensure key payloads are present to raise line coverage
    details = out.get("details", {})
    metrics = out.get("metrics", {})
    assert "policy" in details
    assert "predictive_gate" in metrics
    assert "ab_provenance" in metrics
    assert "proposed_scales" in details
