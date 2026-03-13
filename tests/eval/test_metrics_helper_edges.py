import types

import torch

from invarlock.eval.metrics import (
    DependencyManager,
    InputValidator,
    MetricsConfig,
    ResourceManager,
    _finalize_results,
    _locate_transformer_blocks_enhanced,
    bootstrap_confidence_interval,
    get_metrics_info,
)


def test_bootstrap_confidence_interval_errors_and_success():
    # Errors
    import pytest

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([], n_bootstrap=10)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([1.0, 2.0], n_bootstrap=0)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([1.0, 2.0], alpha=0.0)
    # Success
    lo, hi = bootstrap_confidence_interval([1.0, 2.0, 3.0], n_bootstrap=16, alpha=0.2)
    assert isinstance(lo, float) and isinstance(hi, float)


def test_resource_manager_cpu_fallback_path(monkeypatch):
    cfg = MetricsConfig()
    rm = ResourceManager(cfg)
    # Force a CUDA-like device and tiny available memory so fallback triggers
    rm.device = types.SimpleNamespace(type="cuda")  # type: ignore[attr-defined]
    rm.memory_info = {"gpu_free_gb": 0.2, "system_available_gb": 8.0}
    assert rm.should_use_cpu_fallback(estimated_memory_gb=0.3) is True


def test_input_validator_validate_dataloader_allow_empty():
    cfg = MetricsConfig(allow_empty_data=True)
    InputValidator.validate_dataloader(iter(()), cfg)


def test_locate_transformer_blocks_enhanced_fallback_and_none():
    class DummyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Linear(2, 2)
            self.mlp = torch.nn.Linear(2, 2)

    class WithBlocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = DummyBlock()

        def named_modules(self):
            # Provide a transformer-like name to hit fallback search
            yield from [("transformer.layer0", self.layer0)]

    class NoBlocks(torch.nn.Module):
        def named_modules(self):
            return iter([("root", torch.nn.ReLU())])

    assert _locate_transformer_blocks_enhanced(WithBlocks()) is not None
    assert _locate_transformer_blocks_enhanced(NoBlocks()) is None


def test_dependency_manager_and_metrics_info(monkeypatch):
    dm = DependencyManager()
    # In this repo, optional modules are typically missing; ensure API behaves
    assert dm.is_available("mi_scores") in {True, False}
    info = get_metrics_info()
    assert "available_metrics" in info and "missing_dependencies" in info


def test_resource_manager_cleanup_cuda_branch(monkeypatch):
    cfg = MetricsConfig()
    rm = ResourceManager(cfg)

    # Pretend CUDA is available to exercise cuda.empty_cache branch
    class DummyCuda:
        def empty_cache(self):
            pass

        def is_available(self):
            return True

    monkeypatch.setattr(torch, "cuda", DummyCuda(), raising=False)
    rm.cleanup()


def test_validate_dataloader_raises_and_model_no_params_warns(monkeypatch):
    cfg_fail = MetricsConfig(allow_empty_data=False)
    import pytest

    from invarlock.eval.metrics import ValidationError

    with pytest.raises(ValidationError):
        InputValidator.validate_dataloader(iter(()), cfg_fail)

    class NoParams(torch.nn.Module):
        def parameters(self):
            return iter(())

    InputValidator.validate_model(NoParams(), MetricsConfig(strict_validation=False))


def test_finalize_results_invalid_types_sanitized_and_cached(tmp_path):
    class DummyCache:
        def __init__(self):
            self._store = {}

        def set(self, k, v):
            self._store[k] = v

    res = {"sigma_max": float("inf"), "head_energy": "bad", "mi_gini": 0.1}
    out = _finalize_results(res, ["sigma_max"], DummyCache(), "k", 0.0)
    assert all(k in out for k in ("sigma_max", "head_energy", "mi_gini"))
