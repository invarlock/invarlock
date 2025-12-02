import sys
import types

from invarlock.eval.metrics import DependencyManager, MetricsConfig, ResourceManager


def test_dependency_manager_detects_available_modules(monkeypatch):
    # Fake lens2_mi and lens3 modules
    lens2_mi = types.ModuleType("invarlock.eval.lens2_mi")

    def mi_scores(*_a, **_k):
        return 0.0

    lens2_mi.mi_scores = mi_scores  # type: ignore[attr-defined]

    lens3 = types.ModuleType("invarlock.eval.lens3")

    def scan_model_gains(*_a, **_k):
        class DF:
            columns = ["name", "gain"]

            def __len__(self):
                return 1

            def __getitem__(self, mask):
                return self

            @property
            def name(self):
                return ["block.attn.c_proj"]

            @property
            def gain(self):
                return [0.1]

        return DF()

    lens3.scan_model_gains = scan_model_gains  # type: ignore[attr-defined]

    with monkeypatch.context() as m:
        # Ensure our fake modules are discovered under the expected names
        m.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2_mi)
        m.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2_mi)
        m.setitem(sys.modules, "invarlock.eval.lens3", lens3)
        m.setitem(sys.modules, "invarlock.eval.lens3", lens3)
        dm = DependencyManager()
        assert dm.is_available("mi_scores") and dm.is_available("scan_model_gains")


def test_resource_manager_mps_branch(monkeypatch):
    # Force MPS to be considered available, CUDA not available
    class FakeMPS:
        def is_available(self):
            return True

    class FakeCUDA:
        def is_available(self):
            return False

    with monkeypatch.context() as m:
        m.setattr("invarlock.eval.metrics.torch.backends.mps", FakeMPS(), raising=False)
        m.setattr("invarlock.eval.metrics.torch.cuda", FakeCUDA(), raising=False)
        rm = ResourceManager(MetricsConfig())
        assert str(rm.device) == "mps"
