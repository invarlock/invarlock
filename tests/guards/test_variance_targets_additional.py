from __future__ import annotations

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
