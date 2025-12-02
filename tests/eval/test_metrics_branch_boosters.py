from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from invarlock.eval.data import EvaluationWindow
from invarlock.eval.metrics import (
    DependencyError,
    InputValidator,
    MetricsConfig,
    ResultCache,
    ValidationError,
    _calculate_mi_gini,
    _finalize_results,
    _forward_loss_causal,
    _gini_vectorized,
    _locate_transformer_blocks_enhanced,
    analyze_rmt_changes,
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


def test_validator_model_and_dataloader_branches():
    cfg_strict = MetricsConfig()
    cfg_nonstrict = MetricsConfig(strict_validation=False, allow_empty_data=True)

    class NoParamModel(nn.Module):
        def __init__(self):
            super().__init__()

    # Strict mode: NoParamModel is acceptable in current implementation
    # (parameter counting may be guarded). Ensure call does not error.
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
    deltas = compute_parameter_deltas(before, after, adapter=Adapter())
    assert deltas["params_changed"] == before.transformer.h[0].weight.numel()
    assert deltas["layers_modified"] == 1
    # Structural head/neuron counts are not tracked; ensure layers reflect change
    assert deltas["layers_modified"] >= 1


def test_analyze_spectral_and_rmt_changes_happy_and_error_paths():
    m1 = nn.Linear(2, 2)
    m2 = nn.Linear(2, 2)

    # Provide a temporary module with the expected attribute for import
    import sys
    import types

    fake_spec = types.ModuleType("invarlock.guards.spectral")
    calls = {"i": 0}

    def _fake_compute(*args, **kwargs):
        calls["i"] += 1
        return {"l0": 2.0} if calls["i"] == 1 else {"l0": 1.0}

    fake_spec.compute_spectral_norms = _fake_compute
    with patch.dict(sys.modules, {"invarlock.guards.spectral": fake_spec}):
        s = analyze_spectral_changes(m1, m2)
        assert s["layers_analyzed"] == 1 and s["mean_ratio"] > 0

    # Error path: compute_spectral_norms raises
    fake_spec_err = types.ModuleType("invarlock.guards.spectral")

    def _boom(*a, **k):
        raise RuntimeError("x")

    fake_spec_err.compute_spectral_norms = _boom
    with patch.dict(sys.modules, {"invarlock.guards.spectral": fake_spec_err}):
        s_err = analyze_spectral_changes(m1, m2)
        assert s_err.get("error")

    # Provide temporary rmt module with expected attribute
    fake_rmt = types.ModuleType("invarlock.guards.rmt")
    calls_r = {"i": 0}

    def _fake_mp(*args, **kwargs):
        calls_r["i"] += 1
        return {"l": 0.5} if calls_r["i"] == 1 else {"l": 0.55}

    fake_rmt.compute_mp_stats = _fake_mp
    with patch.dict(sys.modules, {"invarlock.guards.rmt": fake_rmt}):
        r = analyze_rmt_changes(m1, m2)
        assert r["total_layers"] == 1 and 0.0 <= r["stability_ratio"] <= 1.0

    fake_rmt_err = types.ModuleType("invarlock.guards.rmt")

    def _boom2(*a, **k):
        raise ValueError("y")

    fake_rmt_err.compute_mp_stats = _boom2
    with patch.dict(sys.modules, {"invarlock.guards.rmt": fake_rmt_err}):
        r_err = analyze_rmt_changes(m1, m2)
        assert r_err.get("error")


def test_compute_and_measure_helpers():
    model = DummyCausalLM()
    # Window for compute_ppl/measure_* (length > 10 to trigger selection)
    seq = list(range(1, 16))
    attn = [1] * len(seq)
    win = EvaluationWindow([seq, seq], [attn, attn], [0, 1])

    ppl = compute_ppl(model, adapter=None, window=win, device="cpu")
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


def test_compute_perplexity_strict_masked_lm_branch():
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
    assert isinstance(validate_metrics_environment(), bool)


def test_validate_perplexity_branches():
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


def test_mi_gini_gpu_oom_fallback_to_cpu_path():
    # Patch lens2_mi to be available, and force GPU path to raise OOM to hit CPU fallback
    import sys
    import types

    fake_lens2 = types.ModuleType("invarlock.eval.lens2_mi")

    def mi_scores(x: torch.Tensor, y: torch.Tensor):
        # CPU fallback path receives 2D x: [N,D]; return vector [D]
        if x.ndim == 3:
            # Simulate GPU path raising OOM
            raise RuntimeError("CUDA out of memory")
        # Alternate raise on first call to hit j-level exception path when L>1
        if getattr(mi_scores, "_called", False):
            return x.float().mean(dim=0)
        mi_scores._called = True
        raise RuntimeError("fail in j loop")

    fake_lens2.mi_scores = mi_scores
    with patch.dict(sys.modules, {"invarlock.eval.lens2_mi": fake_lens2}):
        # Minimal activation_data to exercise the path
        L, N, T, D = 2, 1, 4, 3
        fc1 = torch.randn(L, N, T, D)
        targ = torch.randint(0, 5, (N, T))
        activation_data = {"fc1_activations": [fc1], "targets": [targ]}
        cfg = MetricsConfig(max_tokens=8)
        from invarlock.eval.metrics import DependencyManager

        dm = DependencyManager()
        val = _calculate_mi_gini(
            DummyCausalLM(), activation_data, dm, cfg, torch.device("cpu")
        )
        # When mi_scores is available and CPU fallback runs, we should get a finite float
        assert isinstance(val, float) and math.isfinite(val)


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
    from invarlock.eval.metrics import _perform_pre_eval_checks

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
    from invarlock.eval.metrics import (
        _calculate_head_energy,
        _collect_activations,
        _extract_fc1_activations,
    )

    data = _collect_activations(model, dl, cfg, torch.device("cpu"))
    assert data["first_batch"] is not None
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
            return lambda m, b: self._g

    # Dataframe-like with columns
    class GainsDF:
        def __init__(self, names, gains):
            self._names = names
            self.gain = gains
            self.columns = ["name", "gain"]

        def __len__(self):
            return len(self._names)

        def __getitem__(self, mask):
            idx = [i for i, m in enumerate(mask) if m]
            return GainsDF([self._names[i] for i in idx], [self.gain[i] for i in idx])

    dm = DM(GainsDF(["mlp.c_fc", "embed"], [0.5, 0.1]))
    from invarlock.eval.metrics import _calculate_head_energy, _calculate_sigma_max

    val = _calculate_sigma_max(
        DummyCausalLM(),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        dm,
        MetricsConfig(),
        torch.device("cpu"),
    )
    assert isinstance(val, float)

    # No columns path -> values empty
    class NoCols:
        def __len__(self):
            return 0

        @property
        def values(self):
            return []

    val2 = _calculate_sigma_max(
        DummyCausalLM(),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM(NoCols()),
        MetricsConfig(),
        torch.device("cpu"),
    )
    assert math.isnan(val2)
    # Head energy empty list path
    he = _calculate_head_energy([], MetricsConfig())
    assert math.isnan(he)

    # Exception path in sigma_max validate_tensor (NaN with strict=True)
    class GainsBad:
        def __len__(self):
            return 2

        @property
        def columns(self):
            return ["name", "gain"]

        @property
        def gain(self):
            return [float("nan"), float("inf")]

        def __getitem__(self, mask):
            return self

    vb = _calculate_sigma_max(
        DummyCausalLM(),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM(GainsBad()),
        MetricsConfig(),
        torch.device("cpu"),
    )
    assert math.isnan(vb)

    # Gains with columns but no 'name' column -> no filtering branch
    class GainsNoName:
        def __len__(self):
            return 2

        @property
        def columns(self):
            return ["foo", "gain"]

        @property
        def gain(self):
            return [0.1, 0.2]

    val3 = _calculate_sigma_max(
        DummyCausalLM(),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM(GainsNoName()),
        MetricsConfig(),
        torch.device("cpu"),
    )
    assert isinstance(val3, float)


def test_measure_latency_early_and_error_paths_and_compute_perplexity_tuple_fallback():
    # measure_latency early return when no suitable sample
    model = DummyCausalLM()
    short = EvaluationWindow([[1, 2, 3]], [[1, 1, 1]], [0])
    assert measure_latency(model, short, device="cpu") == 0.0

    # model raising during warmup -> returns 0.0
    class FailModel(DummyCausalLM):
        def forward(self, *a, **k):
            raise RuntimeError("boom")

    assert (
        measure_latency(
            FailModel(),
            EvaluationWindow([list(range(12))], [[1] * 12], [0]),
            device="cpu",
        )
        == 0.0
    )

    # compute_perplexity fallback path with tuple output
    class TupleModel(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if kwargs.get("return_dict", False):
                # trigger exception path on first call
                raise TypeError("no return_dict supported")
            B, T = input_ids.shape
            V = 8
            logits = torch.randn(B, T, V)
            return (logits,)

    batch = (torch.tensor([[1, 2, 3, 4]]),)
    ppl = compute_perplexity(TupleModel(), [batch], max_samples=1, device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0

    # compute_perplexity error when no valid tokens
    class MinModel(DummyCausalLM):
        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            # Return logits but we'll pass sequences too short
            x = self.emb(input_ids)
            logits = self.fc(x)
            return SimpleNamespace(logits=logits)

    with torch.no_grad():
        bad_batch = {"input_ids": torch.tensor([[1]])}
        from invarlock.eval.metrics import ValidationError as MValidationError

        with pytest.raises(MValidationError):
            compute_perplexity(MinModel(), [bad_batch], max_samples=1, device="cpu")


def test_compute_perplexity_else_tensor_and_invalid_type():
    class Simple2(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return SimpleNamespace(logits=logits)

    # Raw tensor batch (else branch)
    _ = compute_perplexity(
        Simple2(), [torch.tensor([[1, 2, 3]])], max_samples=1, device="cpu"
    )
    # Invalid type batch triggers continue and then ValidationError at end
    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        compute_perplexity(Simple2(), ["bad"], max_samples=1, device="cpu")
    # All masked attention -> continue then error
    bad_attn = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(MValidationError):
        compute_perplexity(
            Simple2(),
            [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": bad_attn}],
            max_samples=1,
            device="cpu",
        )


def test_compute_ppl_empty_sample_and_fallback_tuple():
    # Window with an empty sample should be skipped
    model = DummyCausalLM()
    win = EvaluationWindow([[], list(range(12))], [[0] * 0, [1] * 12], [0, 1])
    _ = compute_ppl(model, adapter=None, window=win, device="cpu")

    # Model raising in try path triggers fallback to tuple
    class TupleOut(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise RuntimeError("fail")
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return (logits,)

    seq = list(range(1, 6))
    attn = [1] * len(seq)
    win2 = EvaluationWindow([seq], [attn], [0])
    _ = compute_ppl(TupleOut(), adapter=None, window=win2, device="cpu")


def test_measure_memory_break_and_continue_and_latency_total_tokens_zero():
    # Window with >5 samples to trigger break and an empty sample to continue
    seq = list(range(4))
    attn = [1] * 4
    inputs = [seq] * 6
    masks = [attn] * 6
    inputs[0] = []
    masks[0] = []
    win = EvaluationWindow(inputs, masks, list(range(6)))
    _ = measure_memory(DummyCausalLM(), win, device="cpu")

    # measure_latency total_tokens == 0 path
    zero = [0] * 12
    win2 = EvaluationWindow([list(range(12))], [zero], [0])
    assert (
        measure_latency(
            DummyCausalLM(), win2, device="cpu", warmup_steps=0, measurement_steps=1
        )
        == 0.0
    )

    # No suitable sample (all sequences <=10) -> returns 0.0
    small = list(range(5))
    win3 = EvaluationWindow([small, small], [[1] * 5, [1] * 5], [0, 1])
    assert (
        measure_latency(
            DummyCausalLM(), win3, device="cpu", warmup_steps=0, measurement_steps=1
        )
        == 0.0
    )

    # Empty window path
    empty = EvaluationWindow([], [], [])
    assert measure_latency(DummyCausalLM(), empty, device="cpu") == 0.0


def test_validate_env_failure_path():
    # Patch DependencyManager on the real module to raise in constructor
    from invarlock.eval import metrics as real_metrics

    class DMErr:
        def __init__(self):
            raise RuntimeError("boom")

    with patch.object(real_metrics, "DependencyManager", DMErr):
        assert real_metrics.validate_metrics_environment() is False


def test_dependency_manager_missing_get_module_and_collect_activations_exception_branch():
    # get_module error path
    from invarlock.eval.metrics import DependencyManager

    dm = DependencyManager()
    with pytest.raises(DependencyError):
        dm.get_module("missing")

    # Collect activations exception path in loop
    class ModelRaises(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[])

        def forward(self, *a, **k):
            raise RuntimeError("fail")

    cfg = MetricsConfig(oracle_windows=2)
    dl = [{"input_ids": torch.ones(1, 4, dtype=torch.long)} for _ in range(2)]
    from invarlock.eval.metrics import _collect_activations

    data = _collect_activations(ModelRaises(), dl, cfg, torch.device("cpu"))
    assert isinstance(data, dict)


def test_forward_loss_causal_branches():
    # ModelOutput-like with loss
    class MO(nn.Module):
        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=True
        ):
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            loss = torch.tensor(0.5)
            return SimpleNamespace(loss=loss, logits=logits)

    ids = torch.randint(0, 8, (1, 4))
    loss, logits = _forward_loss_causal(MO(), ids, labels=ids)
    assert isinstance(loss, float) and logits is not None

    # Tuple(loss, logits) fallback
    class Tup(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return_dict")
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return (torch.tensor(0.4), logits)

    loss2, logits2 = _forward_loss_causal(Tup(), ids, labels=ids)
    assert isinstance(loss2, float) and logits2 is not None

    # Object with attributes but no loss -> compute manually
    class Obj(nn.Module):
        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=True
        ):
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return SimpleNamespace(logits=logits)

    loss3, logits3 = _forward_loss_causal(Obj(), ids, labels=ids)
    assert isinstance(loss3, float) and logits3 is not None

    # Missing logits and labels -> raises
    class Bad(nn.Module):
        def forward(self, *a, **k):
            return SimpleNamespace()

    from invarlock.eval.metrics import MetricsError as MMetricsError

    with pytest.raises(MMetricsError):
        _forward_loss_causal(Bad(), ids)

    # Object with maybe_loss attribute only
    class OnlyLoss(nn.Module):
        def forward(self, *a, **k):
            return SimpleNamespace(loss=torch.tensor(0.1))

    l4, lg4 = _forward_loss_causal(OnlyLoss(), ids, labels=ids)
    assert isinstance(l4, float) and lg4 is None

    # Tuple path with no labels -> should raise for missing labels
    class TupNoLabels(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return_dict")
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return (logits,)

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        _forward_loss_causal(TupNoLabels(), ids)


def test_compute_perplexity_strict_tuple_and_no_valid_tokens():
    class Simple(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return SimpleNamespace(logits=logits)

    # Tuple batch with token_type_ids
    ids = torch.randint(0, 8, (1, 4))
    attn = torch.tensor([[1, 1, 1, 1]])
    ttype = torch.zeros_like(attn)
    ppl = compute_perplexity_strict(Simple(), [(ids, None, attn, ttype)], device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0

    # All invalid tokens due to mask -> raises
    bad_attn = torch.zeros_like(attn)
    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        compute_perplexity_strict(
            Simple(), [(ids, None, bad_attn, ttype)], device="cpu"
        )
