from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core import run_orchestrator_execute_execution as execution_mod
from invarlock.core import run_orchestrator_execute_seed as seed_mod


def test_prepare_execution_state_emits_snapshot_retry_and_reuses_loaded_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        execution_mod._execute_helpers_module,
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

    state = execution_mod._prepare_execution_state(
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
    numpy_stub = SimpleNamespace(
        random=SimpleNamespace(get_state=lambda: ("MT19937", [42], 0, 0, 0))
    )

    state = seed_mod._resolve_loss_seed_and_determinism_state(
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
        set_seed_fn=lambda _value: None,
        numpy_module=numpy_stub,
    )

    assert state.seed_bundle == {"python": 42, "numpy": 42, "torch": None}


def test_prepare_execution_state_uses_snapshot_model_when_not_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        execution_mod._execute_helpers_module,
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

    execution_state = execution_mod._prepare_execution_state(
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


def test_resolve_run_components_fails_when_helper_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Registry:
        def get_adapter(self, _name):
            return SimpleNamespace(name="adapter")

        def get_edit(self, _name):
            return SimpleNamespace(name="noop")

        def get_plugin_metadata(self, name, kind):
            return {"name": name, "kind": kind}

    monkeypatch.setattr(
        execution_mod._execute_helpers_module,
        "_resolve_pm_acceptance_range_impl",
        None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="helper dependencies"):
        execution_mod._resolve_run_components(
            cfg=SimpleNamespace(
                model=SimpleNamespace(adapter="hf_causal"),
                edit=SimpleNamespace(name="noop"),
                guards={"order": []},
            ),
            profile="dev",
            eval_device_override=None,
            pairing_schedule=None,
            seed_bundle={},
            run_id="run-missing",
            baseline_report_data=None,
            model_profile=SimpleNamespace(),
            resolved_loss_type="ppl_causal",
            tiny_relax_enabled=False,
            resolved_device="cpu",
            eval_section={},
            run_dir=tmp_path,
            get_registry_fn=lambda: _Registry(),
            run_config_type=SimpleNamespace,
            to_serialisable_dict_fn=lambda obj: obj if isinstance(obj, dict) else {},
            cfg_value=lambda cfg, key: getattr(cfg, key, None),
            emit=lambda _event: None,
            emit_diagnostic=lambda **_kwargs: None,
            halt=lambda *_args, **_kwargs: None,
        )


def test_prepare_execution_state_fails_when_config_builder_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        execution_mod._execute_helpers_module,
        "_build_run_execution_config_payloads_impl",
        None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="config payload builder"):
        execution_mod._prepare_execution_state(
            cfg=SimpleNamespace(model=SimpleNamespace(id="demo-model")),
            model_profile=SimpleNamespace(),
            profile_normalized="dev",
            resolved_device="cpu",
            run_dir=tmp_path,
            run_id="run-builder-missing",
            adapter=object(),
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
            load_model_with_cfg_fn=lambda *_args, **_kwargs: object(),
            build_snapshot_execution_plan_fn=lambda **_kwargs: None,
            resolve_snapshot_config_fn=lambda _context: {},
            resolve_snapshot_retry_transition_fn=lambda **_kwargs: None,
            free_model_memory_fn=lambda _model: None,
            core_runner_type=lambda: object(),
            optional_runtime_exceptions=(
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ),
        )
