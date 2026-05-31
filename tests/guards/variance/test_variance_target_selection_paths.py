from __future__ import annotations

import builtins

import torch
import torch.nn as nn

import invarlock.guards.variance_targets as vt


class _Guard:
    def __init__(self) -> None:
        self._tap_patterns = ["plain*", "transformer.h.*.*.c_proj"]
        self._focus_modules: set[str] = set()
        self._policy = {"scope": "both"}
        self._stats: dict[str, object] = {}
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self._pairing_reference = ["preview::0"]
        self._pairing_digest = "digest"
        self._dataset_meta = {}
        self._report_meta = {}

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data
    ):
        self.events.append((operation, level, {"message": message, **data}))


class _BadTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        raise RuntimeError("no numpy")


class _BadStateModule:
    def state_dict(self):
        return {"weight": _BadTensor()}


def test_variance_target_utilities_cover_normalization_and_fingerprint_failure() -> (
    None
):
    guard = _Guard()

    assert vt.normalize_module_name("block") == "block"
    assert vt.matches_tap(guard, "plain_name") is True
    assert (
        vt.scale_matches_target("blockx.attn", "transformer.h.1.attn.c_proj") is False
    )
    assert vt.scale_matches_target("block1.foo", "transformer.h.1.attn.c_proj") is False

    guard._target_modules = {"bad": _BadStateModule()}
    assert vt.fingerprint_targets(guard) is None


def test_scale_matches_target_alias_and_split_error_paths() -> None:
    assert vt.scale_matches_target("block1.attn", "transformer.h.1.attn.c_proj") is True
    assert vt.scale_matches_target("block2.mlp", "transformer.h.2.mlp.c_proj") is True
    assert (
        vt.scale_matches_target("blockx.attn.extra", "transformer.h.1.attn.c_proj")
        is False
    )


def test_resolve_target_modules_logs_adapter_fallback_without_layer_count(
    monkeypatch,
) -> None:
    guard = _Guard()
    calls = {"count": 0}

    def fake_iter_layers(_model):
        calls["count"] += 1
        if calls["count"] == 1:
            return iter(())
        raise RuntimeError("no layers")

    monkeypatch.setattr(vt, "iter_transformer_layers", fake_iter_layers)

    class Adapter:
        def describe(self, _model):
            raise RuntimeError("describe failed")

        def get_layer_modules(self, _model, _index):
            return {}

    targets = vt.resolve_target_modules(guard, object(), adapter=Adapter())
    assert targets == {}
    assert any(op == "adapter_describe_error" for op, _, _ in guard.events)
    assert any(op == "adapter_fallback_no_layers" for op, _, _ in guard.events)


def test_resolve_target_modules_adapter_fallback_skips_non_proj_and_unsupported(
    monkeypatch,
) -> None:
    guard = _Guard()

    monkeypatch.setattr(vt, "iter_transformer_layers", lambda _model: iter(()))

    class Unsupported(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.ones(3)

    class Adapter:
        def describe(self, _model):
            return {"n_layer": 1}

        def get_layer_modules(self, _model, _index):
            return {"skip": nn.Linear(2, 2), "layer.attn.c_proj": Unsupported()}

    targets = vt.resolve_target_modules(guard, object(), adapter=Adapter())
    assert targets == {}
    rejected = guard._stats["target_resolution"]["rejected"]
    assert "unsupported_type" in rejected


def test_resolve_target_modules_importerror_expert_paths_and_outer_fallback_error(
    monkeypatch,
) -> None:
    guard = _Guard()

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.pytorch_utils":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.attn.c_proj = nn.Linear(2, 2, bias=False)
            self.mlp = nn.Module()
            self.mlp.fc2 = nn.Linear(2, 2, bias=False)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": 1})()

    monkeypatch.setattr(vt, "iter_transformer_layers", lambda _model: iter([_Block()]))
    targets = vt.resolve_target_modules(guard, _Model(), adapter=None)
    assert set(targets) == {
        "transformer.h.0.attn.c_proj",
        "transformer.h.0.mlp.c_proj",
    }

    class _BrokenModules:
        def __bool__(self) -> bool:
            return True

        def items(self):
            raise RuntimeError("broken items")

    class _Adapter:
        def get_layer_modules(self, _model, _index):
            return _BrokenModules()

    monkeypatch.setattr(vt, "iter_transformer_layers", lambda _model: iter(()))
    guard2 = _Guard()
    vt.resolve_target_modules(guard2, _Model(), adapter=_Adapter())
    assert any(op == "target_resolution_fallback_error" for op, _, _ in guard2.events)


def test_resolve_target_modules_expert_edge_cases_and_config_describe_fallback(
    monkeypatch,
) -> None:
    class _BadDim:
        ndim = 1

        def dim(self):
            raise RuntimeError("bad dim")

    class _ExpertsNonIterable:
        c_proj = object()
        down_proj = torch.ones(1)

    class _Expert:
        def __init__(self) -> None:
            self.fc2 = type("Proj", (), {"weight": _BadDim()})()

    class _ExpertsIterable:
        def __iter__(self):
            return iter([_Expert()])

    class _Block:
        def __init__(self, mlp) -> None:  # noqa: ANN001
            self.mlp = mlp

    guard = _Guard()
    guard._policy["scope"] = "ffn"
    monkeypatch.setattr(
        vt,
        "iter_transformer_layers",
        lambda _model: iter(
            [
                _Block(type("MlpA", (), {"experts": _ExpertsNonIterable()})()),
                _Block(type("MlpB", (), {"experts": _ExpertsIterable()})()),
            ]
        ),
    )

    targets = vt.resolve_target_modules(guard, object(), adapter=None)
    assert targets == {}

    class _SupportedContainer:
        def __init__(self) -> None:
            self.weight = torch.ones(2, 2)

    guard_tap = _Guard()
    guard_tap._policy["scope"] = "ffn"
    guard_tap._tap_patterns = ["transformer.h.0.attn.c_proj"]
    monkeypatch.setattr(
        vt,
        "iter_transformer_layers",
        lambda _model: iter([_Block(_SupportedContainer())]),
    )
    vt.resolve_target_modules(guard_tap, object(), adapter=None)
    assert "tap_mismatch" in guard_tap._stats["target_resolution"]["rejected"]

    class _Model:
        config = type("Cfg", (), {"num_hidden_layers": 1})()

    class _Adapter:
        def describe(self, _model):
            return "oops"

        def get_layer_modules(self, _model, _index):
            return {"layer.attn.c_proj": nn.Linear(2, 2, bias=False)}

    guard_config = _Guard()
    monkeypatch.setattr(vt, "iter_transformer_layers", lambda _model: iter(()))
    targets = vt.resolve_target_modules(guard_config, _Model(), adapter=_Adapter())
    assert list(targets) == ["transformer.h.0.attn.c_proj"]
    assert not any(op == "adapter_describe_error" for op, _, _ in guard_config.events)


def test_resolve_target_modules_expert_tensor_dim_fallback_uses_ndim(
    monkeypatch,
) -> None:
    original_dim = torch.Tensor.dim
    candidate = torch.ones(2, 2)

    def _dim(self):  # noqa: ANN001
        if self is candidate:
            raise RuntimeError("bad dim")
        return original_dim(self)

    class _Experts:
        down_proj = candidate

    class _Block:
        mlp = type("Mlp", (), {"experts": _Experts()})()

    guard = _Guard()
    guard._policy["scope"] = "ffn"
    monkeypatch.setattr(torch.Tensor, "dim", _dim)
    monkeypatch.setattr(vt, "iter_transformer_layers", lambda _model: iter([_Block()]))

    targets = vt.resolve_target_modules(guard, object(), adapter=None)

    assert list(targets) == ["transformer.h.0.mlp.c_proj"]
