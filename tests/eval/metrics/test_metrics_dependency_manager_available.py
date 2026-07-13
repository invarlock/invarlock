import torch

from invarlock.eval import metrics_support
from invarlock.eval.metrics import DependencyManager, InputValidator, MetricsConfig
from invarlock.eval.probes.importance import mi_neuron_scores
from invarlock.guards.spectral_measurement import scan_model_gains


def test_dependency_manager_available_modules(monkeypatch):
    class FeatureSelection:
        mutual_info_regression = staticmethod(lambda *_a, **_k: [0.0])

    monkeypatch.setattr(
        metrics_support.importlib, "import_module", lambda _name: FeatureSelection()
    )

    dm = DependencyManager()
    assert dm.get_module("mi_scores") is mi_neuron_scores
    assert dm.get_module("scan_model_gains") is scan_model_gains


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
