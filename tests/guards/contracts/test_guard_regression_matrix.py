from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt as rmt_mod
from invarlock.guards.invariants import (
    InvariantsGuard,
    _check_standard_invariants,
    _detect_adapter_type,
    check_all_invariants,
)


def test_invariants_cover_missing_embedding_shape_inf_and_adapter_type() -> None:
    class _FragileEmbedding(nn.Module):
        @property
        def num_embeddings(self) -> int:
            raise RuntimeError("num_embeddings unavailable")

        @property
        def weight(self) -> object:
            return SimpleNamespace()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = _FragileEmbedding()

    guard = InvariantsGuard()
    checks = guard._capture_invariants(_Model(), adapter=None)
    assert "embedding_vocab_sizes" not in checks

    class _InfModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.tensor([float("inf")]))

    outcome = check_all_invariants(_InfModel())
    assert any(v.get("type") == "inf_violation" for v in outcome.violations)

    std = _check_standard_invariants(_InfModel())
    assert std["parameter_count"]["passed"] is True
    assert _detect_adapter_type(object()) == "none"

    class _ShapeEmbedding(nn.Embedding):
        def __init__(self) -> None:
            super().__init__(9, 4)

        @property
        def num_embeddings(self) -> int:
            raise RuntimeError("num_embeddings unavailable")

        @num_embeddings.setter
        def num_embeddings(self, value: int) -> None:
            self._stored_num_embeddings = value

    class _ShapeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = _ShapeEmbedding()

    shape_checks = guard._capture_invariants(_ShapeModel(), adapter=None)
    assert shape_checks["embedding_vocab_sizes"]["embed"] == 9


def test_invariants_cover_gap_skips_and_scalar_range_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.ln = nn.LayerNorm(4)
            self.transformer = SimpleNamespace(wte=None)
            self.lm_head = nn.Linear(4, 8, bias=False)

    guard = InvariantsGuard()
    model = _TinyModel()
    guard.prepare(model, adapter=None, calib=None, policy={})

    current_checks = dict(guard.baseline_checks)
    current_checks["evidence_gaps"] = [
        "skip-me",
        {"check": "current_gap", "reason": "runtime"},
    ]
    monkeypatch.setattr(
        guard, "_capture_invariants", lambda *_args, **_kwargs: current_checks
    )

    outcome = guard.finalize(model)
    assert outcome.metrics["evidence_gaps"] >= 1

    class _ScalarData:
        def isnan(self):
            return torch.tensor([False])

        def isinf(self):
            return torch.tensor([False])

        def abs(self):
            return self

        def max(self):
            return 1500.0

    class _ScalarParam:
        data = _ScalarData()

        def numel(self) -> int:
            return 1

    class _ScalarModel:
        def parameters(self):
            yield _ScalarParam()

        def named_parameters(self):
            yield "w", _ScalarParam()

    range_outcome = check_all_invariants(_ScalarModel())
    assert any(v.get("type") == "range_violation" for v in range_outcome.violations)
    assert (
        guard._detect_non_finite(
            SimpleNamespace(
                named_parameters=lambda: iter(()),
                named_buffers=lambda: iter((("buf", torch.tensor([1.0])),)),
            )
        )
        == []
    )

    class _BrokenWeight:
        def data_ptr(self) -> int:
            raise RuntimeError("boom")

    tying_checks = InvariantsGuard()._capture_invariants(
        SimpleNamespace(
            transformer=SimpleNamespace(wte=SimpleNamespace(weight=_BrokenWeight())),
            lm_head=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(1))),
            named_modules=lambda: iter(()),
        ),
        adapter=None,
    )
    assert tying_checks["weight_tying"] is False


def test_rmt_guard_covers_default_auto_context_and_none_epsilon() -> None:
    guard = rmt_mod.RMTGuard()
    original = guard.epsilon_default

    guard.set_run_context(SimpleNamespace(context={"profile": "dev", "auto": "bad"}))
    assert guard._run_profile == "dev"
    assert guard._run_tier == "balanced"
    assert guard._require_activation is False

    guard._set_epsilon_default(None)
    assert guard.epsilon_default == original


def test_guard_policies_cover_overlay_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.guards.policies as policies_mod

    baseline = policies_mod.get_spectral_policy("balanced", use_yaml=False)

    def _overlay(_name: str, _guard: str) -> dict[str, object]:
        return {
            "deadband": True,
            "scope": "all",
        }

    monkeypatch.setattr(
        policies_mod,
        "get_tier_guard_config",
        _overlay,
        raising=True,
    )
    overlay = policies_mod.get_spectral_policy("balanced", use_yaml=True)
    assert overlay["deadband"] == baseline["deadband"]
    assert overlay["scope"] == "all"


def test_guard_policies_ignore_non_mapping_rmt_epsilon_family_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.guards.policies as policies_mod

    baseline = policies_mod.get_rmt_policy("balanced", use_yaml=False)

    monkeypatch.setattr(
        policies_mod,
        "get_tier_guard_config",
        lambda _name, _guard: {"epsilon_by_family": "bad"},
        raising=True,
    )

    overlay = policies_mod.get_rmt_policy("balanced", use_yaml=True)

    assert overlay["epsilon_by_family"] == baseline["epsilon_by_family"]
