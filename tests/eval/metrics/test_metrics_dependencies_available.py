import pytest
import torch

from invarlock.eval import metrics_support
from invarlock.eval.metrics import (
    DependencyManager,
    MetricsConfig,
    ResourceManager,
    get_metrics_info,
    validate_metrics_environment,
)
from invarlock.eval.metrics_activation import _calculate_sigma_max
from invarlock.eval.probes import importance as importance_mod
from invarlock.eval.probes.importance import mi_neuron_scores
from invarlock.guards.spectral_measurement import scan_model_gains


def test_dependency_manager_detects_available_modules(monkeypatch):
    class FeatureSelection:
        mutual_info_regression = staticmethod(lambda *_a, **_k: [0.0])

    monkeypatch.setattr(
        metrics_support.importlib,
        "import_module",
        lambda name: (
            FeatureSelection() if name == "sklearn.feature_selection" else None
        ),
    )

    dm = DependencyManager()
    assert dm.get_module("mi_scores") is mi_neuron_scores
    assert dm.get_module("scan_model_gains") is scan_model_gains


def test_dependency_manager_functions_execute_shipped_metric_contracts(monkeypatch):
    class FeatureSelection:
        mutual_info_regression = staticmethod(lambda *_a, **_k: [0.0])

    monkeypatch.setattr(
        metrics_support.importlib, "import_module", lambda _name: FeatureSelection()
    )
    monkeypatch.setattr(
        importance_mod,
        "mutual_info_regression",
        lambda feature, _target, **_kwargs: [float(feature[:, 0].mean())],
    )
    dm = DependencyManager()

    mi_scores = dm.get_module("mi_scores")(
        torch.tensor([[1.0, 2.0], [3.0, 6.0]]), torch.tensor([0.0, 1.0])
    )
    assert mi_scores.tolist() == pytest.approx([2.0, 4.0])

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    sigma_max = _calculate_sigma_max(model, dm, MetricsConfig(use_cache=False))
    assert sigma_max == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("failure", [ModuleNotFoundError("missing"), None])
def test_metrics_info_reports_mi_availability_truthfully(monkeypatch, failure):
    class FeatureSelection:
        mutual_info_regression = failure

    if isinstance(failure, BaseException):

        def import_feature_selection(_name):
            raise failure

    else:

        def import_feature_selection(_name):
            return FeatureSelection()

    monkeypatch.setattr(
        metrics_support.importlib, "import_module", import_feature_selection
    )

    info = get_metrics_info()
    report = validate_metrics_environment()

    assert "mi_gini" not in info["available_metrics"]
    assert info["unavailable_metrics"] == {"mi_gini": "requires scikit-learn"}
    assert info["available_dependencies"] == []
    assert info["missing_dependencies"][0][0] == "scikit-learn"
    assert report.ok is True
    assert report.available_dependencies == ()


def test_resource_manager_mps_path(monkeypatch):
    # Force MPS to be considered available, CUDA not available
    class FakeMPS:
        def is_available(self):
            return True

    class FakeCUDA:
        def is_available(self):
            return False

    with monkeypatch.context() as m:
        m.setattr(
            "invarlock.eval.metrics_support.torch.backends.mps",
            FakeMPS(),
            raising=False,
        )
        m.setattr(
            "invarlock.eval.metrics_support.torch.cuda", FakeCUDA(), raising=False
        )
        rm = ResourceManager(MetricsConfig())
        assert str(rm.device) == "mps"
