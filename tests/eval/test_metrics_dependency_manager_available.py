import sys
import types

import torch

from invarlock.eval.metrics import DependencyManager, InputValidator, MetricsConfig


def test_dependency_manager_available_modules(monkeypatch):
    # Create stub modules invarlock.eval.lens2_mi and invarlock.eval.lens3
    lens2 = types.ModuleType("invarlock.eval.lens2_mi")

    def mi_scores(x, y):
        return x[..., 0]

    lens2.mi_scores = mi_scores

    lens3 = types.ModuleType("invarlock.eval.lens3")

    def scan_model_gains(model, first_batch):
        class DF:
            columns = ["name", "gain"]

            def __len__(self):
                return 1

            def __getitem__(self, mask):
                return self

            @property
            def name(self):
                return ["mlp.c_fc"]

            @property
            def gain(self):
                return [0.1]

        return DF()

    lens3.scan_model_gains = scan_model_gains

    # Inject into sys.modules under the package paths used by relative imports
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens3", lens3)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens3", lens3)

    dm = DependencyManager()
    assert dm.is_available("mi_scores")
    assert dm.is_available("scan_model_gains")


def test_validate_tensor_inf_raises_and_replacement():
    t = torch.tensor([float("inf"), -float("inf")])
    cfg_strict = MetricsConfig(strict_validation=True)
    import pytest

    from invarlock.eval.metrics import ValidationError

    with pytest.raises(ValidationError):
        InputValidator.validate_tensor(t, "t", cfg_strict)
    cfg_nonstrict = MetricsConfig(strict_validation=False)
    out = InputValidator.validate_tensor(t, "t", cfg_nonstrict)
    assert torch.isfinite(out).all()
