from __future__ import annotations

import builtins
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from invarlock.cli import run_config as run_config_mod


class _BrokenConfig:
    def model_dump(self) -> dict:
        raise RuntimeError("broken dump")


class _CfgWrap:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


def _quiet_console() -> Console:
    return Console(file=StringIO(), force_terminal=False)


def _default_config() -> _CfgWrap:
    return _CfgWrap({"model": {}, "edit": {}, "auto": {}})


def _events() -> tuple[list[tuple[str, str]], object]:
    events: list[tuple[str, str]] = []

    def _event_fn(console, tag: str, message: str, **kwargs) -> None:  # noqa: ARG001
        events.append((tag, message))

    return events, _event_fn


def _prepare_config(**overrides):
    kwargs = {
        "config_path": "config.yaml",
        "profile": None,
        "edit": None,
        "tier": None,
        "probes": None,
        "console": _quiet_console(),
        "event_fn": lambda *args, **kwargs: None,
        "invarlock_config_cls": _CfgWrap,
        "load_config_fn": lambda path: _default_config(),
        "apply_profile_fn": lambda cfg, profile: cfg,
        "apply_auto_adapter_fn": lambda cfg: cfg,
    }
    kwargs.update(overrides)
    return run_config_mod.prepare_config_for_run(**kwargs)


def test_prepare_config_for_run_propagates_model_dump_failure_without_auto_adapter() -> (
    None
):
    events, event_fn = _events()

    with pytest.raises(RuntimeError, match="broken dump"):
        _prepare_config(
            profile="dev",
            tier="balanced",
            event_fn=event_fn,
            invarlock_config_cls=lambda payload: payload,
            load_config_fn=lambda path: _BrokenConfig(),  # noqa: ARG005
        )

    assert ("INIT", "Loading configuration: config.yaml") in events


def test_resolve_requested_edit_name_import_error_falls_back_to_known_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orig_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "invarlock.edits":
            raise ImportError("boom")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert run_config_mod._resolve_requested_edit_name("noop") == "noop"


def test_apply_requested_edit_override_normalizes_non_mapping_edit_section() -> None:
    cfg = _CfgWrap({"edit": "not-a-mapping"})
    out = run_config_mod._apply_requested_edit_override(
        cfg,
        "quant_rtn",
        config_cls=_CfgWrap,
    )

    assert out.model_dump()["edit"] == {"name": "quant_rtn"}


def test_prepare_config_for_run_applies_profile_edit_and_auto_overrides() -> None:
    events, event_fn = _events()

    class Cfg:
        def model_dump(self) -> dict:
            return {
                "model": {"id": "gpt2", "adapter": "hf_causal", "device": "cpu"},
                "edit": "not-a-dict",
                "auto": "not-a-dict",
            }

    profile_calls: list[str] = []
    auto_calls: list[object] = []

    def _apply_profile(cfg, profile):  # noqa: ARG001
        profile_calls.append(profile)
        return cfg

    def _apply_auto_adapter(cfg):
        auto_calls.append(cfg)
        return cfg

    result = _prepare_config(
        profile="release",
        edit="noop",
        tier="balanced",
        probes=3,
        event_fn=event_fn,
        load_config_fn=lambda path: Cfg(),  # noqa: ARG005
        apply_profile_fn=_apply_profile,
        apply_auto_adapter_fn=_apply_auto_adapter,
    )

    payload = result.model_dump()
    assert profile_calls == ["release"]
    assert len(auto_calls) == 1
    assert payload["edit"] == {"name": "noop"}
    assert payload["auto"]["tier"] == "balanced"
    assert payload["auto"]["probes"] == 3
    assert ("EXEC", "Edit override: noop") in events
    assert ("INIT", "Applying profile: release") in events


def test_prepare_config_for_run_uses_default_event_import_and_propagates_auto_adapter_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    def _event_fn(console, tag: str, message: str, **kwargs) -> None:  # noqa: ARG001
        events.append((tag, message))

    monkeypatch.setattr(
        "invarlock.cli.run_shell_output._event",
        _event_fn,
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_config_mod.prepare_config_for_run(
            config_path="config.yaml",
            profile=None,
            edit=None,
            tier=None,
            probes=None,
            console=_quiet_console(),
            load_config_fn=lambda path: _CfgWrap({"model": {}, "edit": {}, "auto": {}}),
            apply_profile_fn=lambda cfg, profile: cfg,  # noqa: ARG005
            apply_auto_adapter_fn=lambda cfg: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )

    assert ("INIT", "Loading configuration: config.yaml") in events


def test_prepare_config_for_run_tolerates_default_auto_adapter_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orig_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "invarlock.adapters.auto":
            raise ImportError("optional adapter unavailable")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    cfg = run_config_mod.prepare_config_for_run(
        config_path="config.yaml",
        profile=None,
        edit=None,
        tier=None,
        probes=None,
        console=_quiet_console(),
        event_fn=lambda *args, **kwargs: None,
        load_config_fn=lambda path: _default_config(),
        apply_profile_fn=lambda cfg, profile: cfg,  # noqa: ARG005
    )

    assert cfg.model_dump()["auto"] == {}


def test_prepare_config_for_run_propagates_default_auto_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(cfg):
        raise RuntimeError("auto adapter failed")

    monkeypatch.setattr(
        "invarlock.adapters.auto.apply_auto_adapter_if_needed",
        _boom,
    )

    with pytest.raises(RuntimeError, match="auto adapter failed"):
        run_config_mod.prepare_config_for_run(
            config_path="config.yaml",
            profile=None,
            edit=None,
            tier=None,
            probes=None,
            console=_quiet_console(),
            event_fn=lambda *args, **kwargs: None,
            load_config_fn=lambda path: _default_config(),
            apply_profile_fn=lambda cfg, profile: cfg,  # noqa: ARG005
        )


@pytest.mark.parametrize(
    ("load_config_fn", "apply_profile_fn", "profile", "tier", "probes", "code"),
    [
        (
            lambda path: (_ for _ in ()).throw(ValueError("bad config")),
            lambda cfg, profile: cfg,
            None,
            None,
            None,
            2,
        ),
        (
            lambda path: _CfgWrap({"model": {}, "edit": {}, "auto": {}}),
            lambda cfg, profile: (_ for _ in ()).throw(RuntimeError("profile boom")),
            "release",
            None,
            None,
            None,
        ),
        (
            lambda path: _CfgWrap({"model": {}, "edit": {}, "auto": {}}),
            lambda cfg, profile: cfg,
            None,
            "invalid",
            None,
            1,
        ),
        (
            lambda path: _CfgWrap({"model": {}, "edit": {}, "auto": {}}),
            lambda cfg, profile: cfg,
            None,
            None,
            11,
            1,
        ),
    ],
)
def test_prepare_config_for_run_error_paths(
    load_config_fn,
    apply_profile_fn,
    profile,
    tier,
    probes,
    code,
) -> None:
    if code is None:
        with pytest.raises(RuntimeError, match="profile boom"):
            _prepare_config(
                profile=profile,
                tier=tier,
                probes=probes,
                load_config_fn=load_config_fn,
                apply_profile_fn=apply_profile_fn,
            )
        return

    with pytest.raises(typer.Exit) as excinfo:
        _prepare_config(
            profile=profile,
            tier=tier,
            probes=probes,
            load_config_fn=load_config_fn,
            apply_profile_fn=apply_profile_fn,
        )

    assert excinfo.value.exit_code == code


def test_prepare_config_for_run_shell_profile_failure_emits_fail_event() -> None:
    events, event_fn = _events()

    with pytest.raises(typer.Exit) as excinfo:
        _prepare_config(
            profile="release",
            event_fn=event_fn,
            apply_profile_fn=lambda cfg, profile: (_ for _ in ()).throw(
                ValueError("profile boom")
            ),
        )

    assert excinfo.value.exit_code == 1
    assert ("FAIL", "profile boom") in events


def test_prepare_config_for_run_non_shell_profile_failure_raises_validation_error() -> (
    None
):
    with pytest.raises(Exception) as excinfo:
        run_config_mod.prepare_config_for_run(
            config_path="config.yaml",
            profile="release",
            edit=None,
            tier=None,
            probes=None,
            console=None,
            event_fn=None,
            invarlock_config_cls=_CfgWrap,
            load_config_fn=lambda path: _CfgWrap({"model": {}, "edit": {}, "auto": {}}),
            apply_profile_fn=lambda cfg, profile: (_ for _ in ()).throw(
                ValueError("profile boom")
            ),
            apply_auto_adapter_fn=lambda cfg: cfg,
        )

    assert getattr(excinfo.value, "code", None) == "E003"


def test_prepare_config_for_run_shell_edit_override_failure_emits_fail_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, event_fn = _events()

    monkeypatch.setattr(
        run_config_mod,
        "_resolve_requested_edit_name",
        lambda _edit: (_ for _ in ()).throw(ValueError("bad edit")),
    )

    with pytest.raises(typer.Exit) as excinfo:
        _prepare_config(
            edit="quant_rtn",
            event_fn=event_fn,
        )

    assert excinfo.value.exit_code == 2
    assert ("FAIL", "bad edit") in events


def test_prepare_config_for_run_tier_probe_override_handles_model_dump_type_error() -> (
    None
):
    class _Cfg:
        def model_dump(self) -> dict:
            raise TypeError("bad dump")

    result = _prepare_config(
        tier="balanced",
        probes=2,
        load_config_fn=lambda path: _Cfg(),
    )

    assert result.model_dump()["auto"] == {"tier": "balanced", "probes": 2}


def test_resolve_device_and_output_uses_cfg_defaults_and_rejects_bad_device() -> None:
    class _Cfg:
        model = SimpleNamespace()
        output = SimpleNamespace(dir="custom-runs")

    resolved, output_dir = run_config_mod.resolve_device_and_output(
        _Cfg(),
        device=None,
        out=None,
        console=_quiet_console(),
        format_kv_line_fn=lambda label, value: f"{label}: {value}",
        device_resolution_note_fn=lambda target, resolved: "note",
        resolve_device_fn=lambda target: "cpu",
        validate_device_fn=lambda device: (True, ""),
    )

    assert resolved == "cpu"
    assert output_dir == Path("custom-runs")


def test_resolve_device_and_output_falls_back_to_runs_and_rejects_invalid_device() -> (
    None
):
    class _Cfg:
        model = SimpleNamespace()
        output = SimpleNamespace()

    resolved, output_dir = run_config_mod.resolve_device_and_output(
        _Cfg(),
        device=None,
        out=None,
        console=_quiet_console(),
        format_kv_line_fn=lambda label, value: f"{label}: {value}",
        device_resolution_note_fn=lambda target, resolved: "note",
        resolve_device_fn=lambda target: "cpu",
        validate_device_fn=lambda device: (True, ""),
    )

    assert resolved == "cpu"
    assert output_dir == Path("runs")

    with pytest.raises(typer.Exit):
        run_config_mod.resolve_device_and_output(
            _Cfg(),
            device="cuda",
            out="outdir",
            console=_quiet_console(),
            format_kv_line_fn=lambda label, value: f"{label}: {value}",
            device_resolution_note_fn=lambda target, resolved: "note",
            resolve_device_fn=lambda target: "cuda",
            validate_device_fn=lambda device: (False, "bad device"),
        )


def test_resolve_device_and_output_uses_default_shell_helpers_and_propagates_cfg_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "invarlock.cli.run_shell_output._format_kv_line",
        lambda label, value: printed.append((label, value)) or f"{label}: {value}",
    )
    monkeypatch.setattr(
        "invarlock.cli.run_shell_output._device_resolution_note",
        lambda target, resolved: f"{target}->{resolved}",
    )

    class _Cfg:
        @property
        def model(self):
            raise RuntimeError("missing model")

        output = SimpleNamespace(dir="custom-runs")

    with pytest.raises(RuntimeError, match="missing model"):
        run_config_mod.resolve_device_and_output(
            _Cfg(),
            device=None,
            out=None,
            console=_quiet_console(),
            resolve_device_fn=lambda target: "cpu",
            validate_device_fn=lambda device: (True, ""),
        )

    assert printed == []


def test_resolve_device_and_output_handles_missing_cfg_device_attribute() -> None:
    class _Cfg:
        @property
        def model(self):
            raise AttributeError("no model")

        output = SimpleNamespace(dir="custom-runs")

    resolved, output_dir = run_config_mod.resolve_device_and_output(
        _Cfg(),
        device=None,
        out=None,
        console=_quiet_console(),
        format_kv_line_fn=lambda label, value: f"{label}: {value}",
        device_resolution_note_fn=lambda target, resolved: "note",
        resolve_device_fn=lambda target: "cpu",
        validate_device_fn=lambda device: (True, ""),
    )

    assert resolved == "cpu"
    assert output_dir == Path("custom-runs")


def test_resolve_provider_and_split_ignores_emit_and_available_split_fallback() -> None:
    calls: list[tuple[str, dict]] = []
    emit_sentinel = object()

    class Provider:
        def available_splits(self):
            return ["validation"]

    def _get_provider(name, **kwargs):  # noqa: ARG001
        calls.append((name, dict(kwargs)))
        return Provider()

    provider, split, used = run_config_mod.resolve_provider_and_split(
        SimpleNamespace(
            dataset=SimpleNamespace(provider=None, split="val"),
        ),
        model_profile=SimpleNamespace(default_provider="wikitext2"),
        get_provider_fn=_get_provider,
        choose_dataset_split_fn=lambda **kwargs: ("validation", True),
        provider_kwargs={"existing": True},
        resolved_device="cpu",
        emit=emit_sentinel,
    )

    assert isinstance(provider, Provider)
    assert split == "validation"
    assert used is True
    assert calls[0][0] == "wikitext2"
    assert calls[0][1]["existing"] is True
    assert calls[0][1]["device_hint"] == "cpu"
    assert "emit" not in calls[0][1]


def test_resolve_provider_and_split_propagates_available_splits_failures() -> None:
    class Provider:
        def available_splits(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_config_mod.resolve_provider_and_split(
            SimpleNamespace(dataset=SimpleNamespace(provider=None, split="val")),
            model_profile=SimpleNamespace(default_provider="wikitext2"),
            get_provider_fn=lambda *_args, **_kwargs: Provider(),
            choose_dataset_split_fn=lambda **kwargs: ("validation", True),
            provider_kwargs={"existing": True},
            resolved_device="cpu",
        )


def test_resolve_provider_and_split_handles_missing_provider_split_and_split_probe_errors() -> (
    None
):
    class _Dataset:
        @property
        def provider(self):
            raise AttributeError("missing provider")

        @property
        def split(self):
            raise AttributeError("missing split")

    class Provider:
        def available_splits(self):
            raise ValueError("bad splits")

    provider, split, used = run_config_mod.resolve_provider_and_split(
        SimpleNamespace(dataset=_Dataset()),
        model_profile=SimpleNamespace(default_provider="wikitext2"),
        get_provider_fn=lambda *_args, **_kwargs: Provider(),
        choose_dataset_split_fn=lambda **kwargs: (
            "validation",
            kwargs["available"] is None,
        ),
        provider_kwargs={"existing": True},
        resolved_device="cpu",
    )

    assert isinstance(provider, Provider)
    assert split == "validation"
    assert used is True


def test_resolve_provider_and_split_handles_missing_dataset_attribute() -> None:
    class _Cfg:
        @property
        def dataset(self):
            raise AttributeError("missing dataset")

    class Provider:
        def available_splits(self):
            return ["validation"]

    provider, split, used = run_config_mod.resolve_provider_and_split(
        _Cfg(),
        model_profile=SimpleNamespace(default_provider="wikitext2"),
        get_provider_fn=lambda *_args, **_kwargs: Provider(),
        choose_dataset_split_fn=lambda **kwargs: (
            "validation",
            kwargs["requested"] is None,
        ),
        provider_kwargs=None,
        resolved_device="cpu",
    )

    assert isinstance(provider, Provider)
    assert split == "validation"
    assert used is True


def test_resolve_provider_and_split_uses_default_provider_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    class Provider:
        pass

    def _get_provider(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        return Provider()

    monkeypatch.setattr("invarlock.eval.data.get_provider", _get_provider)

    provider, split, used = run_config_mod.resolve_provider_and_split(
        SimpleNamespace(dataset=SimpleNamespace(provider="custom", split=None)),
        model_profile=SimpleNamespace(default_provider=None),
        choose_dataset_split_fn=lambda **kwargs: ("test", False),
    )

    assert isinstance(provider, Provider)
    assert split == "test"
    assert used is False
    assert calls == [("custom", {})]


def test_extract_model_load_kwargs_handles_model_dump_failure_and_removed_keys() -> (
    None
):
    class _CfgFail:
        def model_dump(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_config_mod.extract_model_load_kwargs(_CfgFail())

    class _CfgRemoved:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "load_in_8bit": True,
                    "load_in_4bit": False,
                }
            }

    with pytest.raises(Exception) as excinfo:
        run_config_mod.extract_model_load_kwargs(_CfgRemoved())

    assert getattr(excinfo.value, "code", None) == "E007"


def test_extract_model_load_kwargs_returns_empty_on_recoverable_model_dump_error() -> (
    None
):
    class _CfgTypeError:
        def model_dump(self):
            raise TypeError("boom")

    assert run_config_mod.extract_model_load_kwargs(_CfgTypeError()) == {}


def test_extract_model_load_kwargs_rejects_remote_code_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_config_mod, "remote_code_allowed", lambda: False)

    class _CfgRemoteCode:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "trust_remote_code": True,
                }
            }

    with pytest.raises(Exception) as excinfo:
        run_config_mod.extract_model_load_kwargs(_CfgRemoteCode())

    assert getattr(excinfo.value, "code", None) == "E008"


def test_extract_model_load_kwargs_preserves_non_alias_dtype_strings() -> None:
    class _CfgCustomDtype:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "dtype": "float8_e4m3fn",
                }
            }

    kwargs = run_config_mod.extract_model_load_kwargs(_CfgCustomDtype())

    assert kwargs["dtype"] == "float8_e4m3fn"


def test_extract_model_load_kwargs_drops_blank_dtype_strings() -> None:
    class _CfgBlankDtype:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "dtype": "   ",
                }
            }

    kwargs = run_config_mod.extract_model_load_kwargs(_CfgBlankDtype())

    assert kwargs["dtype"] == "   "
