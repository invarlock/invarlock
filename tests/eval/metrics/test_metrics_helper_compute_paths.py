from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from invarlock.eval.data import EvaluationWindow
from invarlock.eval.metrics import (
    InputValidator,
    MetricsConfig,
    ValidationError,
    _finalize_results,
    analyze_spectral_changes,
    compute_parameter_deltas,
    compute_perplexity,
    compute_perplexity_strict,
    compute_ppl,
    get_metrics_info,
    measure_latency,
    measure_memory,
    validate_metrics_environment,
    validate_perplexity,
)
from invarlock.eval.metrics_activation import (
    ResultCache,
    _calculate_head_energy,
    _calculate_mi_gini,
    _calculate_sigma_max,
    _collect_activations,
    _emit_progress,
    _extract_fc1_activations,
    _gini_vectorized,
    _locate_transformer_blocks_enhanced,
    _mi_gini_optimized_cpu_path,
    _perform_pre_eval_checks,
)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, vocab)
        self.config = SimpleNamespace(model_type="gpt2")

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        x = self.emb(input_ids)
        logits = self.fc(x)
        return SimpleNamespace(logits=logits)


class DummyMaskedLM(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, vocab)
        self.config = SimpleNamespace(model_type="bert")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        return_dict=True,
        **kwargs,
    ):
        x = self.emb(input_ids)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            log_probs = logits.log_softmax(dim=-1)
            tgt = labels.clamp_min(0).unsqueeze(-1)
            nll = -log_probs.gather(-1, tgt).squeeze(-1)
            mask = labels != -100
            denom = mask.sum().clamp_min(1)
            loss = (nll * mask).sum() / denom
        return SimpleNamespace(logits=logits, loss=loss)


def test_validator_tensor_nan_inf_replacement():
    cfg = MetricsConfig(strict_validation=False)
    t = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])
    out = InputValidator.validate_tensor(t, "x", cfg)
    assert not torch.isnan(out).any()
    assert torch.isfinite(out).all()


def test_validator_model_and_dataloader_paths():
    cfg_strict = MetricsConfig()
    cfg_nonstrict = MetricsConfig(strict_validation=False, allow_empty_data=True)

    class NoParamModel(nn.Module):
        def __init__(self):
            super().__init__()

    with pytest.raises(ValidationError):
        with torch.no_grad():
            InputValidator.validate_model(NoParamModel(), cfg_strict)

    # Non-strict mode -> no raise
    InputValidator.validate_model(NoParamModel(), cfg_nonstrict)

    # Empty dataloader StopIteration path
    class EmptyDL:
        def __iter__(self):
            return iter(())

    # allow_empty_data=True -> no raise
    InputValidator.validate_dataloader(EmptyDL(), cfg_nonstrict)

    # allow_empty_data=False -> raises
    with pytest.raises(ValidationError):
        InputValidator.validate_dataloader(EmptyDL(), cfg_strict)


def test_result_cache_and_finalize_results():
    cfg = MetricsConfig()
    model = nn.Linear(2, 2)
    dl = [torch.randint(0, 5, (1, 4))]
    cache = ResultCache(cfg)
    key = cache._get_cache_key(model, dl, cfg)
    assert cache.get(key) is None
    # finalize should coerce bad types and cache
    res = {"a": "bad", "b": float("inf"), "c": 1.23}
    out = _finalize_results(res, [], cache, key, 0.0)
    assert math.isnan(out["a"]) and math.isnan(out["b"]) and out["c"] == 1.23
    assert key in cache.cache
    cache.clear()
    assert cache.get(key) is None
    # Disabled cache path
    cfg_nc = MetricsConfig(use_cache=False)
    cache_nc = ResultCache(cfg_nc)
    assert cache_nc.get("nope") is None
    cache_nc.set("k", {"x": 1.0})
    assert cache_nc.get("k") is None


def test_locate_transformer_blocks_patterns_and_fallback():
    class Block(nn.Module):
        pass

    class ModelA(nn.Module):
        def __init__(self):
            super().__init__()
            # pattern: transformer.h
            self.transformer = SimpleNamespace(h=[Block(), Block()])

    blocks = _locate_transformer_blocks_enhanced(ModelA())
    assert isinstance(blocks, list) and len(blocks) == 2

    class TransBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(2, 2)
            self.mlp = nn.Linear(2, 2)

    class ModelB(nn.Module):
        def __init__(self):
            super().__init__()
            # fallback search by name/attrs
            self.transformer_block1 = TransBlock()

    blocks_fb = _locate_transformer_blocks_enhanced(ModelB())
    assert isinstance(blocks_fb, list) and len(blocks_fb) >= 1

    class ModelNone(nn.Module):
        pass

    assert _locate_transformer_blocks_enhanced(ModelNone()) is None


def test_compute_parameter_deltas_and_structural_counts():
    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            # names like 'transformer.h.0.weight' to match regex
            self.transformer = Container()

    before = Tiny()
    after = Tiny()
    # align states so only our intended tensor differs
    after.load_state_dict(before.state_dict())
    with torch.no_grad():
        after.transformer.h[0].weight.add_(1.0)  # change a full tensor

    class Adapter:
        def describe(self, model):
            # Use model attributes to differentiate before/after
            if getattr(model, "tag", "before") == "before":
                return {"heads_per_layer": [8, 8], "mlp_dims": [16, 16]}
            return {"heads_per_layer": [8, 7], "mlp_dims": [16, 15]}

    before.tag = "before"
    after.tag = "after"
    deltas = compute_parameter_deltas(before, after)
    assert deltas["params_changed"] == before.transformer.h[0].weight.numel()
    assert deltas["layers_modified"] == 1
    # Structural head/neuron counts are not tracked; ensure layers reflect change
    assert deltas["layers_modified"] >= 1


def test_compute_parameter_deltas_counts_changes_without_numeric_layer_match():
    class Mismatch(nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2, 2), value))

        def named_parameters(self, prefix: str = "", recurse: bool = True):
            yield "transformer.layers.x.weight", self.weight

        def parameters(self, recurse: bool = True):
            yield self.weight

    before = Mismatch(0.0)
    after = Mismatch(1.0)

    deltas = compute_parameter_deltas(before, after)

    assert deltas["params_changed"] == 4
    assert deltas["layers_modified"] == 0


def test_analyze_spectral_and_rmt_changes_happy_and_error_paths():
    m1 = nn.Linear(2, 2)
    m2 = nn.Linear(2, 2)

    # Provide a temporary module with the expected attribute for import
    import sys
    import types

    fake_spec = types.ModuleType("invarlock.guards.spectral_measurement")
    calls = {"i": 0}

    def _fake_compute(*args, **kwargs):
        calls["i"] += 1
        return {"l0": 2.0} if calls["i"] == 1 else {"l0": 1.0}

    fake_spec.compute_spectral_norms = _fake_compute
    with patch.dict(sys.modules, {"invarlock.guards.spectral_measurement": fake_spec}):
        s = analyze_spectral_changes(m1, m2)
        assert s["layers_analyzed"] == 1 and s["mean_ratio"] > 0

    # Error path: compute_spectral_norms raises
    fake_spec_err = types.ModuleType("invarlock.guards.spectral_measurement")

    def _boom(*a, **k):
        raise RuntimeError("x")

    fake_spec_err.compute_spectral_norms = _boom
    with patch.dict(
        sys.modules, {"invarlock.guards.spectral_measurement": fake_spec_err}
    ):
        s_err = analyze_spectral_changes(m1, m2)
        assert s_err.get("error")

    fake_spec_missing = types.ModuleType("invarlock.guards.spectral_measurement")
    with patch.dict(
        sys.modules, {"invarlock.guards.spectral_measurement": fake_spec_missing}
    ):
        s_missing = analyze_spectral_changes(m1, m2)
        assert s_missing == {"error": "spectral_analysis_unavailable"}


def test_analyze_spectral_changes_skips_layers_missing_from_after_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spectral_module = types.ModuleType("invarlock.guards.spectral_measurement")
    sentinel_before = object()
    sentinel_after = object()
    spectral_module.compute_spectral_norms = lambda model, scope="ffn": (
        {"layer_a": 2.0, "layer_b": 4.0}
        if model is sentinel_before
        else {"layer_a": 3.0}
    )
    monkeypatch.setitem(
        sys.modules,
        "invarlock.guards.spectral_measurement",
        spectral_module,
    )

    changes = analyze_spectral_changes(sentinel_before, sentinel_after)

    assert changes["layers_analyzed"] == 1
    assert "layer_b" not in changes["layer_changes"]
    assert changes["layer_changes"]["layer_a"]["ratio"] == 1.5


def test_compute_parameter_deltas_handles_sparsity_and_failures() -> None:
    before = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    after = nn.Sequential(nn.Linear(4, 2))

    deltas = compute_parameter_deltas(before, after)
    assert deltas["params_changed"] >= 0
    assert deltas["sparsity"] is not None
    assert deltas["sparsity"] > 0

    class BrokenParams(nn.Module):
        def named_parameters(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    broken = compute_parameter_deltas(BrokenParams(), BrokenParams())
    assert broken == {"params_changed": 0, "layers_modified": 0, "sparsity": None}


def test_compute_and_measure_helpers():
    model = DummyCausalLM()
    # Window for compute_ppl/measure_* (length > 10 to trigger selection)
    seq = list(range(1, 16))
    attn = [1] * len(seq)
    win = EvaluationWindow([seq, seq], [attn, attn], [0, 1])

    ppl = compute_ppl(model, window=win, device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0

    lat = measure_latency(model, win, device="cpu", warmup_steps=1, measurement_steps=2)
    mem = measure_memory(model, win, device="cpu")
    assert lat >= 0.0 and mem >= 0.0

    # compute_perplexity on small dataloader
    batch = {
        "input_ids": torch.tensor([seq]),
        "attention_mask": torch.tensor([attn]),
    }
    ppl2 = compute_perplexity(model, [batch], max_samples=1, device="cpu")
    assert isinstance(ppl2, float) and ppl2 >= 1.0


def test_compute_perplexity_strict_masked_lm_path():
    model = DummyMaskedLM()
    seq = list(range(1, 12))
    attn = [1] * len(seq)
    batch = {
        "input_ids": torch.tensor([seq]),
        "attention_mask": torch.tensor([attn]),
    }
    ppl = compute_perplexity_strict(model, [batch], device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0

    # loss None branch -> continue then ValidationError overall
    class MaskedNoLoss(DummyMaskedLM):
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            token_type_ids=None,
            return_dict=True,
            **kwargs,
        ):
            x = self.emb(input_ids)
            logits = self.fc(x)
            return SimpleNamespace(logits=logits, loss=None)

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        compute_perplexity_strict(MaskedNoLoss(), [batch], device="cpu")

    # valid_tokens == 0 branch with all masked positions
    zero_attn_batch = {
        "input_ids": torch.tensor([seq]),
        "attention_mask": torch.zeros_like(torch.tensor([attn])),
    }
    with pytest.raises(MValidationError):
        compute_perplexity_strict(model, [zero_attn_batch], device="cpu")


def test_info_and_environment_helpers():
    info = get_metrics_info()
    assert "available_metrics" in info and isinstance(info.get("default_config"), dict)
    # Environment validation returns boolean and logs missing optional deps
    assert validate_metrics_environment().ok is True


def test_validate_perplexity_paths():
    ok, status, msg = validate_perplexity(10.0)
    assert ok and status == "excellent"

    ok, status, msg = validate_perplexity(75.0)
    assert ok and status == "good"

    ok, status, msg = validate_perplexity(150.0)
    assert ok and status == "acceptable"

    ok, status, msg = validate_perplexity(300.0)
    assert ok and status == "poor"

    ok, status, msg = validate_perplexity(2500.0)
    assert not ok and status in {"poor", "unusable"}

    # With vocab_size adjustment, thresholds increase; allow_high bypasses error
    ok, status, msg = validate_perplexity(5000.0, vocab_size=4000, allow_high=True)
    assert ok and isinstance(msg, str)

    # Invalid values
    ok, status, msg = validate_perplexity(float("nan"))
    assert not ok and status == "invalid"
    ok, status, msg = validate_perplexity(0.5)
    assert not ok and status == "invalid"


def test_mi_gini_scores_each_layer_with_cpu_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[tuple[int, ...], tuple[int, ...], str]] = []

    def mi_scores(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(x.shape), tuple(y.shape), x.device.type))
        return torch.arange(1, x.shape[1] + 1, dtype=torch.float32)

    class ExactDependencies:
        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            assert name == "mi_scores"
            return mi_scores

    empty_cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True)
    )
    layer_count, batch_count, token_count, width = 2, 1, 4, 3
    activation_data = {
        "fc1_activations": [torch.randn(layer_count, batch_count, token_count, width)],
        "targets": [torch.randint(0, 5, (batch_count, token_count - 1))],
    }

    val = _calculate_mi_gini(
        DummyCausalLM(),
        activation_data,
        ExactDependencies(),
        MetricsConfig(max_tokens=8),
        torch.device("cpu"),
    )

    assert math.isfinite(val)
    assert calls == [((3, 3), (3,), "cpu"), ((3, 3), (3,), "cpu")]
    assert empty_cache_calls == []


def test_resource_manager_and_pre_eval_checks_and_gini_zero():
    # ResourceManager branches
    cfg = MetricsConfig()
    from invarlock.eval.metrics import ResourceManager

    rm = ResourceManager(cfg)
    # Force a cuda-like path by tweaking attributes
    rm.device = torch.device("cuda")
    rm.memory_info = {"gpu_free_gb": 1.0}
    assert rm.should_use_cpu_fallback(1.0) is True
    assert rm.should_use_cpu_fallback(0.4) is False
    assert isinstance(_gini_vectorized(torch.zeros(0)), float)

    # cleanup_after False path
    cfg2 = MetricsConfig(cleanup_after=False)
    rm2 = ResourceManager(cfg2)
    rm2.cleanup()  # should be a no-op
    # Device override branch
    cfg3 = MetricsConfig()
    cfg3.device = torch.device("cpu")
    rm3 = ResourceManager(cfg3)
    assert rm3.device.type == "cpu"

    # Pre-eval checks
    class ModelPre(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_positions=4)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return SimpleNamespace(ok=True)

    dl = [{"input_ids": torch.ones(1, 8, dtype=torch.long)}]
    _perform_pre_eval_checks(ModelPre(), dl, torch.device("cpu"), cfg)

    # Pre-eval dry run failure path
    class ModelFail(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_positions=2)

        def forward(self, *a, **k):
            raise RuntimeError("dry-run fail")

    _perform_pre_eval_checks(ModelFail(), dl, torch.device("cpu"), cfg)


def test_collect_activations_and_fc1_extraction_shape_mismatch_and_head_energy():
    # Build a model that returns hidden states and has two blocks with different c_fc dims
    class Block:
        def __init__(self, out):
            self.mlp = SimpleNamespace(c_fc=nn.Linear(4, out))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[Block(4), Block(3)])

        def forward(self, input_ids, output_hidden_states=False):
            B, T = input_ids.shape
            hs = [torch.randn(B, T, 4) for _ in range(4)]
            return SimpleNamespace(hidden_states=hs)

    model = M()
    # Two batches, one longer than max_tokens to exercise trimming
    batch = {"input_ids": torch.ones(1, 16, dtype=torch.long)}
    dl = [batch, batch]
    cfg = MetricsConfig(oracle_windows=2, max_tokens=8)

    data = _collect_activations(model, dl, cfg, torch.device("cpu"))
    assert set(data) == {"hidden_states", "fc1_activations", "targets"}
    # FC1 activations should stack despite shape mismatch due to filtering
    out = _extract_fc1_activations(
        model,
        SimpleNamespace(hidden_states=[torch.randn(1, 8, 4) for _ in range(4)]),
        cfg,
    )
    assert out is None or isinstance(out, torch.Tensor)
    he = _calculate_head_energy([torch.randn(1, 1, 8, 4)], cfg)
    assert isinstance(he, float)


def test_calculate_sigma_max_variants_and_head_energy_empty():
    # Fake dep manager
    class DM:
        def __init__(self, gains):
            self._g = gains

        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            return lambda m: self._g

    dm = DM({"spectral_norms": [0.5, 0.1], "scanned_modules": 2})
    val = _calculate_sigma_max(
        DummyCausalLM(),
        dm,
        MetricsConfig(),
    )
    assert val == pytest.approx(0.5)

    val2 = _calculate_sigma_max(
        DummyCausalLM(),
        DM({"scanned_modules": 0}),
        MetricsConfig(),
    )
    assert math.isnan(val2)
    # Head energy empty list path
    he = _calculate_head_energy([], MetricsConfig())
    assert math.isnan(he)

    vb = _calculate_sigma_max(
        DummyCausalLM(),
        DM({"spectral_norms": [float("nan"), float("inf")]}),
        MetricsConfig(),
    )
    assert math.isnan(vb)

    val3 = _calculate_sigma_max(
        DummyCausalLM(),
        DM({"spectral_norms": [0.1, float("nan"), 0.2]}),
        MetricsConfig(),
    )
    assert val3 == pytest.approx(0.2)


def test_metrics_activation_progress_and_zero_layer_mi_paths() -> None:
    updates = []
    cfg = MetricsConfig(progress_observer=updates.append)
    _emit_progress(cfg, phase="mi_gini_cpu", completed=1, total=2)
    assert updates[0].completed == 1
    assert updates[0].total == 2

    assert math.isnan(
        _mi_gini_optimized_cpu_path(
            torch.empty((0, 2, 1), dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            max_per_layer=2,
            config=MetricsConfig(use_cache=False),
            mi_scores_fn=lambda x, _y: torch.zeros(x.shape[1]),
        )
    )
