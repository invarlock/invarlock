from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt_analysis as rmt_analysis
from invarlock.guards import rmt_activation_runtime as runtime
from invarlock.guards.rmt_runtime import build_after_edit_result, build_prepare_result


class IndexedSource:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> int:
        return idx


class IterableOnly:
    def __iter__(self):
        return iter([10, 20, 30])


class BrokenLenSource:
    def __len__(self) -> int:
        raise RuntimeError("bad len")

    def __getitem__(self, idx: int) -> int:
        return idx

    def __iter__(self):
        return iter([7, 8, 9])


class SkippingIndexedSource:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> int:
        if idx == 1:
            raise RuntimeError("skip")
        return idx


def test_collect_calibration_batches_supports_index_and_iterable_sources() -> None:
    assert runtime.collect_calibration_batches(
        IndexedSource(),
        3,
        activation_sampling={"windows": {"indices_policy": "last"}},
    ) == [2, 3, 4]
    assert runtime.collect_calibration_batches(
        IterableOnly(),
        2,
    ) == [10, 20]
    assert runtime.collect_calibration_batches(object(), 2) == []


def test_collect_calibration_batches_skips_invalid_index_records() -> None:
    assert runtime.collect_calibration_batches(
        SkippingIndexedSource(),
        3,
        activation_sampling={"windows": {"indices_policy": "unknown"}},
    ) == [0, 2]


def test_collect_calibration_batches_handles_non_mapping_policy_and_len_failure() -> (
    None
):
    assert runtime.collect_calibration_batches(
        IndexedSource(),
        2,
        activation_sampling=object(),
    ) == [0, 4]
    assert runtime.collect_calibration_batches(BrokenLenSource(), 2) == [7, 8]
    assert runtime.collect_calibration_batches(
        IndexedSource(),
        2,
        activation_sampling={"windows": []},
    ) == [0, 4]


def test_prepare_activation_inputs_normalizes_and_falls_back_to_clone(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is not None
    assert tuple(input_ids.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)


def test_prepare_activation_inputs_drops_mask_when_clone_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_clone = torch.Tensor.clone

    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )

    def _clone(self):
        if int(self.sum().item()) == 2:
            raise RuntimeError("bad clone")
        return original_clone(self)

    monkeypatch.setattr(torch.Tensor, "clone", _clone)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is None


def test_prepare_activation_inputs_returns_none_when_input_tensorization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_as_tensor = torch.as_tensor

    class BadInput:
        pass

    def _as_tensor(value):
        if isinstance(value, BadInput):
            raise RuntimeError("bad tensor")
        return original_as_tensor(value)

    monkeypatch.setattr(torch, "as_tensor", _as_tensor)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        BadInput(),
        torch.device("cpu"),
    )

    assert input_ids is None
    assert attention_mask is None


def test_prepare_activation_inputs_drops_bad_attention_mask_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_as_tensor = torch.as_tensor

    class BadMask:
        pass

    def _as_tensor(value):
        if isinstance(value, BadMask):
            raise RuntimeError("bad mask")
        return original_as_tensor(value)

    monkeypatch.setattr(torch, "as_tensor", _as_tensor)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {"input_ids": [1, 2], "attention_mask": BadMask()},
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is None


def test_prepare_activation_inputs_unsqueezes_tensor_clone_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {
            "input_ids": torch.tensor([5, 6]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )

    assert tuple(input_ids.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)


def test_prepare_activation_inputs_unsqueezes_in_fallback_after_dim_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_as_tensor = torch.as_tensor
    original_dim = torch.Tensor.dim

    input_value = object()
    mask_value = object()
    input_tensor = torch.tensor([7, 8])
    mask_tensor = torch.tensor([1, 1])
    attempts = {"input": 0, "mask": 0}

    def _as_tensor(value):  # noqa: ANN001
        if value is input_value:
            return input_tensor
        if value is mask_value:
            return mask_tensor
        return original_as_tensor(value)

    def _dim(self):  # noqa: ANN001
        if self is input_tensor and attempts["input"] == 0:
            attempts["input"] += 1
            raise RuntimeError("retry input dim")
        if self is mask_tensor and attempts["mask"] == 0:
            attempts["mask"] += 1
            raise RuntimeError("retry mask dim")
        return original_dim(self)

    monkeypatch.setattr(torch, "as_tensor", _as_tensor)
    monkeypatch.setattr(torch.Tensor, "dim", _dim)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {"input_ids": input_value, "attention_mask": mask_value},
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is not None
    assert tuple(input_ids.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)


def test_batch_token_weight_falls_back_to_input_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.Tensor,
        "sum",
        lambda self: (_ for _ in ()).throw(RuntimeError("bad sum")),
    )

    input_ids = torch.ones((1, 3))
    attention_mask = torch.ones((1, 3))

    assert runtime.batch_token_weight(input_ids, attention_mask) == 3


def test_activation_edge_risk_handles_zero_std_mp_edge_failure_and_iters_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sqrt = torch.sqrt

    monkeypatch.setattr(torch, "sqrt", lambda value: torch.tensor(0.0))
    assert runtime.activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(torch, "sqrt", original_sqrt)
    monkeypatch.setattr(
        rmt_analysis,
        "mp_bulk_edge",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert runtime.activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(rmt_analysis, "mp_bulk_edge", lambda *args, **kwargs: 1.0)
    assert (
        runtime.activation_edge_risk(
            torch.randn(3, 2), estimator={"iters": 0, "init": "e0"}
        )
        is not None
    )


def test_activation_edge_risk_negative_iters_and_nan_sigma_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        runtime.activation_edge_risk(
            torch.randn(3, 2), estimator={"iters": -2, "init": "e0"}
        )
        is not None
    )

    original_vector_norm = torch.linalg.vector_norm
    calls = {"count": 0}

    def _vector_norm(*args, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 2:
            return torch.tensor(float("nan"))
        return original_vector_norm(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", _vector_norm)
    assert runtime.activation_edge_risk(torch.randn(3, 2)) is None


def test_activation_svd_outliers_covers_tensor_shape_and_failure_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outliers, max_ratio, sigma_max = runtime.activation_svd_outliers(
        torch.randn(2, 2, 2), margin=1.0, deadband=0.0
    )
    assert isinstance(outliers, int)
    assert max_ratio >= 0.0
    assert sigma_max >= 0.0

    monkeypatch.setattr(
        torch.Tensor,
        "cpu",
        lambda self: (_ for _ in ()).throw(RuntimeError("bad cpu")),
    )
    assert runtime.activation_svd_outliers(
        torch.randn(2, 2), margin=1.0, deadband=0.0
    ) == (
        0,
        0.0,
        0.0,
    )

    monkeypatch.setattr(torch.Tensor, "cpu", lambda self: self)
    monkeypatch.setattr(
        torch.linalg,
        "svdvals",
        lambda *_a, **_k: (_ for _ in ()).throw(torch.linalg.LinAlgError("boom")),
    )
    assert runtime.activation_svd_outliers(
        torch.randn(2, 2), margin=1.0, deadband=0.0
    ) == (
        0,
        0.0,
        0.0,
    )

    monkeypatch.setattr(torch.linalg, "svdvals", lambda *_a, **_k: torch.tensor([]))
    assert runtime.activation_svd_outliers(
        torch.randn(2, 2), margin=1.0, deadband=0.0
    ) == (
        0,
        0.0,
        0.0,
    )


def test_compute_activation_edge_risk_returns_none_when_models_never_produce_scores() -> (
    None
):
    class OneDimModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            return torch.tensor([1.0, 2.0])

    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            raise RuntimeError("boom")

    kwargs = {
        "allowed_suffixes": ("attn",),
        "activation_sampling": None,
        "estimator": None,
        "deadband": 0.0,
        "margin": 0.0,
        "classify_family_fn": lambda name: "attn",
    }
    batch = {"input_ids": torch.ones((1, 2)), "attention_mask": torch.ones((1, 2))}

    assert (
        runtime.compute_activation_edge_risk(OneDimModel(), [batch], **kwargs) is None
    )
    assert (
        runtime.compute_activation_edge_risk(FailingModel(), [batch], **kwargs) is None
    )


def test_compute_activation_edge_risk_handles_attention_mask_typeerror_and_bad_handle_removal() -> (
    None
):
    class BadHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def remove(self) -> None:
            self._handle.remove()
            raise RuntimeError("cannot remove")

    class HookedLinear(nn.Linear):
        def register_forward_hook(self, hook):  # noqa: ANN001
            return BadHandle(super().register_forward_hook(hook))

    class TypeErrorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = HookedLinear(2, 2, bias=False)

        def forward(self, input_ids):  # noqa: ANN001
            return self.attn(input_ids.float())

    result = runtime.compute_activation_edge_risk(
        TypeErrorModel(),
        [{"input_ids": torch.ones((1, 2)), "attention_mask": torch.ones((1, 2))}],
        allowed_suffixes=("attn",),
        activation_sampling=None,
        estimator={"iters": 1, "init": "e0"},
        deadband=0.0,
        margin=0.0,
        classify_family_fn=lambda name: "attn",
    )

    assert result is not None
    assert result["batches_used"] == 1
    assert result["token_weight_total"] == 2
    assert result["edge_risk_by_family"]["attn"] >= 0.0


def test_rmt_runtime_adapter_hook_resolution_and_adapter_paths() -> None:
    class _Adapter:
        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            return {
                "input_ids": torch.ones((1, 2), device=device),
                "attention_mask": torch.ones((1, 2), device=device),
            }

    class _HookModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = attention_mask
            return self.attn(input_ids.float())

    assert runtime._resolve_adapter_hook(None, "prepare_generation_inputs") is None
    assert runtime._resolve_adapter_hook(Mock(), "prepare_generation_inputs") is None
    assert callable(
        runtime._resolve_adapter_hook(_Adapter(), "prepare_generation_inputs")
    )

    kwargs = {
        "allowed_suffixes": ("attn",),
        "activation_sampling": None,
        "estimator": {"iters": 1, "init": "e0"},
        "deadband": 0.0,
        "margin": 0.0,
        "classify_family_fn": lambda name: "attn",
    }
    batch = {"id": "ex-1", "image_path": "/tmp/a.png", "answers": ["cat"]}

    risk = runtime.compute_activation_edge_risk(
        _HookModel(),
        [batch],
        adapter=_Adapter(),
        **kwargs,
    )
    assert risk is not None
    assert risk["batches_used"] == 1


def test_compute_activation_edge_risk_none_weight_fallback_and_restore_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _HookModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = attention_mask
            return self.attn(input_ids.float())

    kwargs = {
        "allowed_suffixes": ("attn",),
        "activation_sampling": None,
        "estimator": {"iters": 1, "init": "e0"},
        "deadband": 0.0,
        "margin": 0.0,
        "classify_family_fn": lambda name: "attn",
    }
    batch = {"input_ids": torch.ones((1, 2)), "attention_mask": torch.ones((1, 2))}

    monkeypatch.setattr(runtime, "activation_edge_risk", lambda *_a, **_k: None)
    assert runtime.compute_activation_edge_risk(_HookModel(), [batch], **kwargs) is None

    class WeirdInt:
        def __bool__(self) -> bool:
            return True

        def __int__(self) -> int:
            raise RuntimeError("bad int")

        def __radd__(self, other: int) -> int:
            return other + 2

    monkeypatch.setattr(
        runtime, "activation_edge_risk", lambda *_a, **_k: (1.5, 1.0, 1.0)
    )
    monkeypatch.setattr(runtime, "batch_token_weight", lambda *_a, **_k: WeirdInt())

    weighted = runtime.compute_activation_edge_risk(_HookModel(), [batch], **kwargs)
    assert weighted is not None
    assert weighted["batches_used"] == 1
    assert weighted["token_weight_total"] == 2

    class _RetryFailModel(_HookModel):
        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            if attention_mask is not None:
                raise TypeError("retry")
            raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "batch_token_weight", runtime.batch_token_weight)
    retry_fail = runtime.compute_activation_edge_risk(
        _RetryFailModel(), [batch], **kwargs
    )
    assert retry_fail is None

    model = _HookModel()
    model.eval()
    restored = runtime.compute_activation_edge_risk(model, [batch], **kwargs)
    assert restored is not None
    assert model.training is False


def test_rmt_compute_activation_outliers_uses_adapter_generation_inputs() -> None:
    class _Adapter:
        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            return {
                "input_ids": torch.ones((1, 2), device=device),
                "attention_mask": torch.ones((1, 2), device=device),
            }

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = attention_mask
            return self.attn(input_ids.float())

    class _Guard:
        adapter = _Adapter()
        margin = 1.0
        deadband = 0.0

        def _get_activation_modules(self, model):  # noqa: ANN001
            return runtime.get_activation_modules(model, allowed_suffixes=("attn",))

        def _activation_svd_outliers(self, output, *, margin, deadband):  # noqa: ANN001
            _ = output, margin, deadband
            return 1, 2.0, 3.0

        def _prepare_activation_inputs(self, batch, device):  # noqa: ANN001
            return runtime.prepare_activation_inputs(batch, device)

        def _batch_token_weight(self, input_ids, attention_mask):  # noqa: ANN001
            return runtime.batch_token_weight(input_ids, attention_mask)

    out = runtime.compute_activation_outliers(
        _Guard(),
        _Model(),
        [{"id": "ex-1", "image_path": "/tmp/a.png", "answers": ["cat"]}],
    )

    assert out is not None
    assert out["outlier_count"] > 0
    assert out["token_weight_total"] == 2


def test_compute_activation_outliers_failure_and_restore_paths() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = attention_mask
            return self.attn(input_ids.float())

    class _RetryFailModel(_Model):
        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            if attention_mask is not None:
                raise TypeError("retry")
            raise RuntimeError("boom")

    class _RuntimeFailModel(_Model):
        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = input_ids, attention_mask
            raise RuntimeError("boom")

    class _BadHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def remove(self) -> None:
            self._handle.remove()
            raise RuntimeError("boom")

    class _BadHandleLinear(nn.Linear):
        def register_forward_hook(self, hook):  # noqa: ANN001
            return _BadHandle(super().register_forward_hook(hook))

    class _BadHandleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _BadHandleLinear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = attention_mask
            return self.attn(input_ids.float())

    class _RaisingHookLinear(nn.Linear):
        def register_forward_hook(self, hook):  # noqa: ANN001
            raise RuntimeError("no hook")

    class _RaisingHookModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _RaisingHookLinear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            _ = attention_mask
            return self.attn(input_ids.float())

    class _Guard:
        adapter = None
        margin = 1.0
        deadband = 0.0

        def _get_activation_modules(self, model):  # noqa: ANN001
            return runtime.get_activation_modules(model, allowed_suffixes=("attn",))

        def _activation_svd_outliers(self, output, *, margin, deadband):  # noqa: ANN001
            _ = output, margin, deadband
            raise RuntimeError("boom")

        def _prepare_activation_inputs(self, batch, device):  # noqa: ANN001
            return runtime.prepare_activation_inputs(batch, device)

        def _batch_token_weight(self, input_ids, attention_mask):  # noqa: ANN001
            return runtime.batch_token_weight(input_ids, attention_mask)

    batch = {"input_ids": torch.ones((1, 2)), "attention_mask": torch.ones((1, 2))}
    assert runtime.compute_activation_outliers(_Guard(), _Model(), [batch]) is not None

    class _GuardNoHook(_Guard):
        def _activation_svd_outliers(self, output, *, margin, deadband):  # noqa: ANN001
            _ = output, margin, deadband
            return 1, 2.0, 3.0

    no_hook = runtime.compute_activation_outliers(
        _GuardNoHook(), _RaisingHookModel(), [batch]
    )
    assert no_hook is not None
    assert no_hook["outlier_count"] == 0

    bad_handle_model = _BadHandleModel()
    bad_handle_model.eval()
    handled = runtime.compute_activation_outliers(
        _GuardNoHook(), bad_handle_model, [batch]
    )
    assert handled is not None
    assert bad_handle_model.training is False

    assert (
        runtime.compute_activation_outliers(_GuardNoHook(), _RetryFailModel(), [batch])
        is None
    )
    assert (
        runtime.compute_activation_outliers(
            _GuardNoHook(), _RuntimeFailModel(), [batch]
        )
        is None
    )


def test_prepare_and_after_edit_result_contract_helpers() -> None:
    prepare = build_prepare_result(
        ready=True,
        baseline_metrics={"edge_risk_by_family": {"attn": 0.2}},
        policy_applied={"activation_required": True},
        preparation_time=1.25,
    )
    assert prepare == {
        "ready": True,
        "baseline_metrics": {"edge_risk_by_family": {"attn": 0.2}},
        "policy_applied": {"activation_required": True},
        "preparation_time": 1.25,
    }

    failed = build_prepare_result(
        ready=False,
        baseline_metrics={},
        policy_applied={},
        preparation_time=0.5,
        error="Activation baseline unavailable",
    )
    assert failed["error"] == "Activation baseline unavailable"

    after = build_after_edit_result(
        edge_risk_by_module={"layer": 0.2},
        edge_risk_by_family={"attn": 0.2},
        token_weight_total=12,
        batches_used=3,
    )
    assert after == {
        "analysis_source": "activations_edge_risk",
        "edge_risk_by_module": {"layer": 0.2},
        "edge_risk_by_family": {"attn": 0.2},
        "token_weight_total": 12,
        "batches_used": 3,
    }
