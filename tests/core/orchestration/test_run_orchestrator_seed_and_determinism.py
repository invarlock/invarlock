from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from invarlock.core import determinism_policy
from invarlock.core.orchestration import environment as environment_mod


def test_set_seed_reseeds_python_and_numpy() -> None:
    determinism_policy.set_seed(123)
    first = (random.random(), float(np.random.rand()))

    determinism_policy.set_seed(123)
    second = (random.random(), float(np.random.rand()))

    assert first == second


def test_set_seed_tolerates_missing_torch(monkeypatch) -> None:
    monkeypatch.setattr(determinism_policy, "torch", None)
    determinism_policy.set_seed(321)
    first = (random.random(), float(np.random.rand()))

    determinism_policy.set_seed(321)
    second = (random.random(), float(np.random.rand()))

    assert first == second


def test_determinism_policy_exports_seed_helper() -> None:
    assert callable(determinism_policy.set_seed)


def test_apply_determinism_preset_handles_partial_or_missing_seed_payloads(
    monkeypatch,
) -> None:
    seed_bundle = {"python": 1, "numpy": 2, "torch": 3}

    monkeypatch.setattr(
        "invarlock.core.determinism_policy.apply_determinism_preset",
        lambda **_kwargs: {"seeds": {"numpy": 17}},
    )

    preset = environment_mod._apply_determinism_preset(
        profile_label="ci",
        resolved_device="cpu",
        seed_bundle=seed_bundle,
        seed_value=1,
    )

    assert preset == {"seeds": {"numpy": 17}}
    assert seed_bundle == {"python": 1, "numpy": 17, "torch": 3}

    monkeypatch.setattr(
        "invarlock.core.determinism_policy.apply_determinism_preset",
        lambda **_kwargs: {"mode": "throughput"},
    )

    preset = environment_mod._apply_determinism_preset(
        profile_label="ci",
        resolved_device="cpu",
        seed_bundle=seed_bundle,
        seed_value=1,
    )

    assert preset == {"mode": "throughput"}
    assert seed_bundle == {"python": 1, "numpy": 17, "torch": 3}


def test_resolve_loss_seed_and_determinism_state_prefers_warn_only_and_seed_fallbacks() -> (
    None
):
    class _Dataset:
        @property
        def seed(self) -> object:
            return "not-an-int"

    class _Cudnn:
        benchmark = True

        @property
        def deterministic(self) -> bool:
            return False

        @deterministic.setter
        def deterministic(self, _value: bool) -> None:
            raise TypeError("deterministic locked")

    class _Torch:
        def __init__(self) -> None:
            self.calls: list[tuple[bool, bool]] = []
            self.backends = SimpleNamespace(cudnn=_Cudnn())

        def use_deterministic_algorithms(
            self, enabled: bool, *, warn_only: bool
        ) -> None:
            self.calls.append((enabled, warn_only))

        def initial_seed(self) -> int:
            return 99

    torch_mod = _Torch()
    emitted: list[object] = []

    state = environment_mod._resolve_loss_seed_and_determinism_state(
        SimpleNamespace(dataset=_Dataset(), eval={"loss": {"type": "auto"}}),
        model_profile=SimpleNamespace(default_loss="mlm"),
        profile_normalized="ci",
        determinism_mode="strict",
        determinism_warn_only=True,
        optional_torch=lambda: torch_mod,
        emit=emitted.append,
        cfg_value=lambda cfg, key: getattr(cfg, key, None),
        config_value_exceptions=(AttributeError, KeyError, TypeError),
        numeric_exceptions=(OverflowError, TypeError, ValueError),
        optional_runtime_exceptions=(
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ),
    )

    assert state.resolved_loss_type == "mlm"
    assert state.seed_value == 42
    assert state.seed_bundle == {"python": 42, "numpy": 42, "torch": 99}
    assert torch_mod.calls == [(True, True)]
    assert emitted and emitted[0].python_seed == 42


def test_resolve_loss_seed_and_determinism_state_without_cudnn_backend() -> None:
    class _TorchNoCudnn:
        def __init__(self) -> None:
            self.calls: list[tuple[bool, bool]] = []
            self.backends = SimpleNamespace()

        def use_deterministic_algorithms(
            self, enabled: bool, *, warn_only: bool
        ) -> None:
            self.calls.append((enabled, warn_only))

        def initial_seed(self) -> int:
            return 7

    torch_mod = _TorchNoCudnn()
    state = environment_mod._resolve_loss_seed_and_determinism_state(
        SimpleNamespace(dataset=SimpleNamespace(seed=5), eval={"loss": {"type": "ce"}}),
        model_profile=SimpleNamespace(default_loss="mlm"),
        profile_normalized="release",
        determinism_mode="strict",
        determinism_warn_only=False,
        optional_torch=lambda: torch_mod,
        emit=lambda _event: None,
        cfg_value=lambda cfg, key: getattr(cfg, key, None),
        config_value_exceptions=(AttributeError, KeyError, TypeError),
        numeric_exceptions=(OverflowError, TypeError, ValueError),
        optional_runtime_exceptions=(
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ),
    )

    assert state.resolved_loss_type == "ce"
    assert state.seed_bundle == {"python": 5, "numpy": 5, "torch": 7}
    assert torch_mod.calls == [(True, False)]
