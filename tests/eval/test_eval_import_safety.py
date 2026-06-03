from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_mi_probe_import_and_call_are_lazy_for_sklearn(monkeypatch):
    torch = pytest.importorskip("torch")
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            sys.modules.pop(mod, None)
    sys.modules.pop("invarlock.eval.probes.importance", None)

    mod = importlib.import_module("invarlock.eval.probes.importance")

    assert callable(mod.mutual_info_regression)

    with pytest.raises(ModuleNotFoundError, match="scikit-learn"):
        mod.mi_neuron_scores(torch.ones(2, 2), torch.ones(2))


def test_eval_probes_package_root_is_light_and_has_no_heavy_reexports(
    monkeypatch,
):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.eval.probes",
        "invarlock.eval.probes.importance",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.eval.probes")

    assert hasattr(mod, "__all__")
    assert mod.__all__ == []
    assert not hasattr(mod, "compute_head_energy_scores")
    assert not hasattr(mod, "fft_head_energy")
    assert not hasattr(mod, "compute_neuron_mi_scores")
    assert not hasattr(mod, "mi_neuron_scores")
    assert not hasattr(mod, "compute_post_attention_head_scores")
    assert "invarlock.eval.probes.importance" not in sys.modules
