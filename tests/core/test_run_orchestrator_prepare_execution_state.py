from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core import run_orchestrator_execute_prepare as prepare_mod


def test_prepare_execution_state_emits_snapshot_retry_and_reuses_loaded_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        prepare_mod._execute_helpers_module,
        "_build_run_execution_config_payloads_impl",
        lambda **_kwargs: SimpleNamespace(auto_config={}, edit_config={}),
        raising=False,
    )

    loaded_model = object()
    transitions: list[tuple[str, object]] = []

    class _Adapter:
        def snapshot(self, *_args, **_kwargs):  # pragma: no cover - capability marker
            return None

        def restore(self, *_args, **_kwargs):  # pragma: no cover - capability marker
            return None

    state = prepare_mod._prepare_execution_state(
        cfg=SimpleNamespace(model=SimpleNamespace(id="demo-model")),
        model_profile=SimpleNamespace(),
        profile_normalized="dev",
        resolved_device="cpu",
        run_dir=tmp_path,
        run_id="run-1",
        adapter=_Adapter(),
        edit_op=SimpleNamespace(name="noop"),
        guards=[],
        prefer_local_files_only=False,
        skip_overhead=False,
        skip_overhead_source=None,
        direct_reuse_loaded_model=False,
        emitted_skip_overhead_warning=False,
        retry_controller=None,
        cfg_value=lambda cfg, key: getattr(cfg, key, None),
        emit=lambda _event: None,
        emit_transition=lambda phase, diagnostic: transitions.append(
            (phase, diagnostic)
        ),
        record_timed_step=lambda _name: nullcontext(),
        load_model_with_cfg_fn=lambda *_args, **_kwargs: loaded_model,
        build_snapshot_execution_plan_fn=lambda **_kwargs: SimpleNamespace(
            model=None,
            restore_fn=None,
            skip_model_load=False,
            snapshot_tmpdir=None,
            snapshot_provenance={"restore_failed": False, "reload_path_used": False},
            emitted_skip_overhead_warning=False,
            snapshot_enabled=True,
            diagnostics=(),
        ),
        resolve_snapshot_config_fn=lambda _context: {},
        resolve_snapshot_retry_transition_fn=lambda **_kwargs: SimpleNamespace(
            skip_model_load=False,
            emitted_skip_overhead_warning=False,
            diagnostics=(SimpleNamespace(code="snapshot-retry"),),
        ),
        free_model_memory_fn=lambda _model: None,
        core_runner_type=lambda: object(),
        optional_runtime_exceptions=(
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ),
    )

    assert transitions == [("snapshot_retry", SimpleNamespace(code="snapshot-retry"))]
    assert state.model is loaded_model
    assert state.skip_model_load is True


def test_resolve_loss_seed_defaults_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare_mod, "set_seed", lambda _value: None)
    monkeypatch.setattr(
        prepare_mod.np.random,
        "get_state",
        lambda: ("MT19937", [42], 0, 0, 0),
    )

    state = prepare_mod._resolve_loss_seed_and_determinism_state(
        SimpleNamespace(eval={"loss": {"type": "auto"}}),
        model_profile=SimpleNamespace(default_loss="ce"),
        profile_normalized="dev",
        determinism_mode=None,
        determinism_warn_only=False,
        optional_torch=lambda: None,
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

    assert state.seed_bundle == {"python": 42, "numpy": 42, "torch": None}


def test_prepare_execution_state_uses_snapshot_model_when_not_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        prepare_mod._execute_helpers_module,
        "_build_run_execution_config_payloads_impl",
        lambda **_kwargs: SimpleNamespace(auto_config={}, edit_config={}),
        raising=False,
    )
    loaded_model = object()
    alternate_model = object()

    class _Adapter:
        def snapshot(self, *_args, **_kwargs):
            return None

        def restore(self, *_args, **_kwargs):
            return None

    execution_state = prepare_mod._prepare_execution_state(
        cfg=SimpleNamespace(model=SimpleNamespace(id="demo-model")),
        model_profile=SimpleNamespace(),
        profile_normalized="dev",
        resolved_device="cpu",
        run_dir=tmp_path,
        run_id="run-2",
        adapter=_Adapter(),
        edit_op=SimpleNamespace(name="noop"),
        guards=[],
        prefer_local_files_only=False,
        skip_overhead=False,
        skip_overhead_source=None,
        direct_reuse_loaded_model=False,
        emitted_skip_overhead_warning=False,
        retry_controller=None,
        cfg_value=lambda cfg, key: getattr(cfg, key, None),
        emit=lambda _event: None,
        emit_transition=lambda *_args, **_kwargs: None,
        record_timed_step=lambda _name: nullcontext(),
        load_model_with_cfg_fn=lambda *_args, **_kwargs: loaded_model,
        build_snapshot_execution_plan_fn=lambda **_kwargs: SimpleNamespace(
            model=alternate_model,
            restore_fn=None,
            skip_model_load=False,
            snapshot_tmpdir=None,
            snapshot_provenance={"restore_failed": False, "reload_path_used": False},
            emitted_skip_overhead_warning=False,
            snapshot_enabled=False,
            diagnostics=(),
        ),
        resolve_snapshot_config_fn=lambda _context: {},
        resolve_snapshot_retry_transition_fn=lambda **_kwargs: SimpleNamespace(
            skip_model_load=False,
            emitted_skip_overhead_warning=False,
            diagnostics=(),
        ),
        free_model_memory_fn=lambda _model: None,
        core_runner_type=lambda: object(),
        optional_runtime_exceptions=(
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ),
    )

    assert execution_state.model is alternate_model
    assert execution_state.skip_model_load is False
