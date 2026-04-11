from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_root_import_does_not_require_torch(monkeypatch):
    # Simulate an environment where torch is not installed.
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    # Ensure a clean import state for the package root.
    for mod in ["invarlock", "invarlock.adapters"]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock")

    # Core public API should be available without importing adapters/torch.
    assert hasattr(mod, "__version__")
    assert hasattr(mod, "CFG")
    assert hasattr(mod, "Defaults")
    assert hasattr(mod, "get_default_config")

    # Top-level package should not auto-expose adapters.
    assert not hasattr(mod, "adapters")


def test_guard_helper_import_does_not_require_torch(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.guards",
        "invarlock.guards.invariants",
        "invarlock.guards.rmt",
        "invarlock.guards.spectral",
        "invarlock.guards.variance",
        "invarlock.guards.spectral_results",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.guards.spectral_results")

    assert hasattr(mod, "build_spectral_finalize_metrics")


def test_utils_import_and_memory_probe_do_not_require_torch(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    sys.modules.pop("invarlock.utils", None)

    mod = importlib.import_module("invarlock.utils")

    assert hasattr(mod, "get_memory_usage")
    memory = mod.get_memory_usage()
    assert "rss_mb" in memory


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
    sys.modules.pop("invarlock.eval.probes.mi", None)

    mod = importlib.import_module("invarlock.eval.probes.mi")

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
        "invarlock.eval.probes.fft",
        "invarlock.eval.probes.mi",
        "invarlock.eval.probes.post_attention",
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
    assert "invarlock.eval.probes.mi" not in sys.modules
    assert "invarlock.eval.probes.fft" not in sys.modules
    assert "invarlock.eval.probes.post_attention" not in sys.modules


def test_reporting_package_root_import_is_light_and_source_compatible(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.reporting",
        "invarlock.reporting.html",
        "invarlock.reporting.render",
        "invarlock.reporting.report_make",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.reporting")

    assert hasattr(mod, "REPORT_SCHEMA_VERSION")
    assert callable(mod.make_report)
    assert callable(mod.render_report_markdown)
    assert callable(mod.render_report_html)
    assert callable(mod.validate_report)

    from invarlock.reporting import (  # noqa: PLC0415
        make_report,
        render_report_html,
        render_report_markdown,
        validate_report,
    )

    assert make_report is mod.make_report
    assert render_report_markdown is mod.render_report_markdown
    assert render_report_html is mod.render_report_html
    assert validate_report is mod.validate_report


def test_report_command_module_import_is_light_without_numpy(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ModuleNotFoundError("numpy not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.cli.commands.report",
        "invarlock.reporting.report_contract",
        "invarlock.reporting.report_make",
        "invarlock.eval.primary_metric",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.cli.commands.report")

    assert hasattr(mod, "report_app")
    assert "invarlock.reporting.report_contract" not in sys.modules
    assert "invarlock.reporting.report_make" not in sys.modules
    assert "invarlock.eval.primary_metric" not in sys.modules
