from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from invarlock.eval.data import EvaluationWindow
from invarlock.eval.metrics import (
    DependencyError,
    MetricsConfig,
    ValidationError,
    compute_perplexity,
    compute_perplexity_strict,
    compute_ppl,
    measure_latency,
    measure_memory,
)
from invarlock.eval.metrics_activation import (
    _collect_activations,
)
from invarlock.eval.metrics_runtime import forward_loss_causal


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


def test_measure_latency_early_and_error_paths_and_compute_perplexity_tuple_fallback():
    # measure_latency now rejects windows without any usable long sample
    model = DummyCausalLM()
    short = EvaluationWindow([[1, 2, 3]], [[1, 1, 1]], [0])
    with pytest.raises(ValidationError) as exc_info:
        measure_latency(model, short, device="cpu")
    assert (
        exc_info.value.details["reason"]
        == "latency measurement requires at least one sequence longer than 10 tokens"
    )

    # model raising during warmup now surfaces the failure
    class FailModel(DummyCausalLM):
        def forward(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="Latency warmup failed"):
        measure_latency(
            FailModel(),
            EvaluationWindow([list(range(12))], [[1] * 12], [0]),
            device="cpu",
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


def test_measure_latency_raises_on_measurement_execution_failure():
    class GoodModel(DummyCausalLM):
        def forward(self, *a, **k):
            return super().forward(*a, **k)

    calls = {"count": 0}

    def _call_model(_model, **_kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("measurement boom")
        return SimpleNamespace(logits=torch.randn(1, 12, 16))

    window = EvaluationWindow([list(range(12))], [[1] * 12], [0])
    from invarlock.eval import metrics_runtime as metrics_runtime_mod

    original = metrics_runtime_mod.call_model
    metrics_runtime_mod.call_model = _call_model
    try:
        with pytest.raises(RuntimeError, match="Latency measurement failed"):
            measure_latency(
                GoodModel(),
                window,
                device="cpu",
                warmup_steps=1,
                measurement_steps=2,
            )
    finally:
        metrics_runtime_mod.call_model = original


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
    ppl = compute_ppl(model, window=win, device="cpu")
    assert ppl >= 1.0

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
    fallback_ppl = compute_ppl(TupleOut(), window=win2, device="cpu")
    assert fallback_ppl >= 1.0


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
    with pytest.raises(ValidationError) as exc_info:
        measure_latency(
            DummyCausalLM(), win2, device="cpu", warmup_steps=0, measurement_steps=1
        )
    assert (
        exc_info.value.details["reason"]
        == "latency measurement requires at least one attended token"
    )

    # No suitable sample (all sequences <=10) -> invalid measurement input
    small = list(range(5))
    win3 = EvaluationWindow([small, small], [[1] * 5, [1] * 5], [0, 1])
    with pytest.raises(ValidationError) as exc_info:
        measure_latency(
            DummyCausalLM(), win3, device="cpu", warmup_steps=0, measurement_steps=1
        )
    assert (
        exc_info.value.details["reason"]
        == "latency measurement requires at least one sequence longer than 10 tokens"
    )

    # Empty window path
    empty = EvaluationWindow([], [], [])
    with pytest.raises(ValidationError) as exc_info:
        measure_latency(DummyCausalLM(), empty, device="cpu")
    assert (
        exc_info.value.details["reason"]
        == "latency measurement requires a non-empty evaluation window"
    )


def test_validate_env_failure_path():
    # Patch DependencyManager on the real module to raise in constructor
    from invarlock.eval import metrics as metrics_environment_mod

    class DMErr:
        def __init__(self):
            raise RuntimeError("boom")

    with patch.object(metrics_environment_mod, "DependencyManager", DMErr):
        assert metrics_environment_mod.validate_metrics_environment().ok is False


def test_dependency_manager_missing_get_module_and_collect_activations_exception_path():
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
    data = _collect_activations(ModelRaises(), dl, cfg, torch.device("cpu"))
    assert isinstance(data, dict)


def test_forward_loss_causal_paths():
    # ModelOutput-like with loss
    class MO(nn.Module):
        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=True
        ):
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            loss = torch.tensor(0.5)
            return SimpleNamespace(loss=loss, logits=logits)

    ids = torch.randint(0, 8, (1, 4))
    loss, logits = forward_loss_causal(MO(), ids, labels=ids)
    assert isinstance(loss, float) and logits is not None

    # Tuple(loss, logits) fallback
    class Tup(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return_dict")
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return (torch.tensor(0.4), logits)

    loss2, logits2 = forward_loss_causal(Tup(), ids, labels=ids)
    assert isinstance(loss2, float) and logits2 is not None

    # Object with attributes but no loss -> compute manually
    class Obj(nn.Module):
        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=True
        ):
            logits = torch.randn(input_ids.size(0), input_ids.size(1), 8)
            return SimpleNamespace(logits=logits)

    loss3, logits3 = forward_loss_causal(Obj(), ids, labels=ids)
    assert isinstance(loss3, float) and logits3 is not None

    # Missing logits and labels -> raises
    class Bad(nn.Module):
        def forward(self, *a, **k):
            return SimpleNamespace()

    from invarlock.eval.metrics import MetricsError as MMetricsError

    with pytest.raises(MMetricsError):
        forward_loss_causal(Bad(), ids)

    # Object with maybe_loss attribute only
    class OnlyLoss(nn.Module):
        def forward(self, *a, **k):
            return SimpleNamespace(loss=torch.tensor(0.1))

    l4, lg4 = forward_loss_causal(OnlyLoss(), ids, labels=ids)
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
        forward_loss_causal(TupNoLabels(), ids)


def test_forward_logits_causal_fallbacks_and_error_paths():
    from invarlock.eval.metrics_runtime import forward_logits_causal

    ids = torch.randint(0, 8, (1, 4))
    logits = torch.randn(1, 4, 8)

    class ObjectLogits(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return dict")
            return SimpleNamespace(logits=logits)

    assert torch.equal(forward_logits_causal(ObjectLogits(), ids), logits)

    class EmptyTuple(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return dict")
            return ()

    from invarlock.eval.metrics import MetricsError as MMetricsError

    with pytest.raises(MMetricsError):
        forward_logits_causal(EmptyTuple(), ids)

    class NonTensorLogits(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return SimpleNamespace(logits=[1, 2, 3])

    with pytest.raises(MMetricsError):
        forward_logits_causal(NonTensorLogits(), ids)

    class RawTensorFallback(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return dict")
            return logits

    assert torch.equal(forward_logits_causal(RawTensorFallback(), ids), logits)


def test_forward_loss_causal_fallback_object_without_loss_computes_from_logits():
    ids = torch.randint(0, 8, (1, 4))
    logits = torch.randn(1, 4, 8)

    class ObjectWithoutLoss(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            if kwargs.get("return_dict", False):
                raise TypeError("no return dict")
            return SimpleNamespace(loss=None, logits=logits)

    loss, returned_logits = forward_loss_causal(ObjectWithoutLoss(), ids, labels=ids)

    assert isinstance(loss, float)
    assert torch.equal(returned_logits, logits)


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
