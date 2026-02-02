# ruff: noqa: I001,E402,F811
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from invarlock.cli.commands import run as run_mod


def test_resolve_device_and_output_prefers_out(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        "invarlock.cli.device.resolve_device",
        lambda requested: calls.append(str(requested)) or "cpu",
    )
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda _d: (True, "")
    )

    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        output=SimpleNamespace(dir=str(tmp_path / "cfg_runs")),
    )
    resolved, out_dir = run_mod._resolve_device_and_output(
        cfg, device=None, out=str(tmp_path / "explicit_out"), console=Console()
    )
    assert resolved == "cpu"
    assert out_dir == tmp_path / "explicit_out"
    assert calls == ["cpu"]


def test_resolve_device_and_output_uses_cfg_output_dir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda _d: (True, "")
    )

    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        output=SimpleNamespace(dir=str(tmp_path / "cfg_runs")),
    )
    _, out_dir = run_mod._resolve_device_and_output(
        cfg, device=None, out=None, console=Console()
    )
    assert out_dir == tmp_path / "cfg_runs"


def test_resolve_device_and_output_ignores_out_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda _d: (True, "")
    )

    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        out=SimpleNamespace(dir=str(tmp_path / "ignored_runs")),
    )
    _, out_dir = run_mod._resolve_device_and_output(
        cfg, device=None, out=None, console=Console()
    )
    assert out_dir == Path("runs")


def test_resolve_device_and_output_falls_back_to_runs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda _d: (True, "")
    )

    cfg = SimpleNamespace(model=SimpleNamespace(device="cpu"))
    _, out_dir = run_mod._resolve_device_and_output(
        cfg, device=None, out=None, console=Console()
    )
    assert out_dir == Path("runs")
    assert (tmp_path / "runs").exists()


def test_resolve_device_and_output_handles_cfg_device_getattr_failure(
    tmp_path: Path, monkeypatch
) -> None:
    class _ModelRaises:
        def __getattr__(self, _name: str):
            raise RuntimeError("boom")

    seen: list[str] = []
    monkeypatch.setattr(
        "invarlock.cli.device.resolve_device",
        lambda requested: seen.append(str(requested)) or "cpu",
    )
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda _d: (True, "")
    )

    cfg = SimpleNamespace(
        model=_ModelRaises(), output=SimpleNamespace(dir=str(tmp_path / "o"))
    )
    run_mod._resolve_device_and_output(
        cfg, device=None, out=str(tmp_path / "o2"), console=Console()
    )
    assert seen and seen[0] == "auto"


def test_resolve_device_and_output_raises_on_invalid_device(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda _d: (False, "bad")
    )

    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        output=SimpleNamespace(dir=str(tmp_path / "o")),
    )
    with pytest.raises(typer.Exit):
        run_mod._resolve_device_and_output(
            cfg, device=None, out=str(tmp_path / "o2"), console=Console()
        )


@dataclass
class _ProviderConfigObj:
    kind: str
    file: str | None = None

    def items(self):
        return {"kind": self.kind, "file": self.file, "empty": ""}.items()


def test_resolve_provider_and_split_injects_device_hint_for_wikitext2(
    monkeypatch,
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(provider="wikitext2", split="validation")
    )
    model_profile = SimpleNamespace(default_provider=None)

    seen: dict[str, object] = {}

    class _Provider:
        def available_splits(self):
            return ["validation", "train"]

    def _get_provider(name: str, **kwargs):
        seen["name"] = name
        seen["kwargs"] = dict(kwargs)
        return _Provider()

    provider, split, used_fallback = run_mod._resolve_provider_and_split(
        cfg,
        model_profile,
        get_provider_fn=_get_provider,
        provider_kwargs={},
        console=Console(),
        resolved_device="cpu",
    )
    assert isinstance(provider, _Provider)
    assert split == "validation"
    assert used_fallback is False
    assert seen["name"] == "wikitext2"
    assert isinstance(seen["kwargs"], dict)
    assert seen["kwargs"].get("device_hint") == "cpu"


def test_resolve_provider_and_split_supports_object_provider_config(
    monkeypatch,
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider=_ProviderConfigObj(kind="local_jsonl", file="x.jsonl")
        )
    )
    model_profile = SimpleNamespace(default_provider="synthetic")

    class _Provider:
        def available_splits(self):
            return ["validation"]

    seen: dict[str, object] = {}

    def _get_provider(name: str, **kwargs):
        seen["name"] = name
        seen["kwargs"] = dict(kwargs)
        return _Provider()

    _provider, split, used_fallback = run_mod._resolve_provider_and_split(
        cfg,
        model_profile,
        get_provider_fn=_get_provider,
        provider_kwargs={},
        console=Console(),
        resolved_device=None,
    )
    assert split == "validation"
    assert used_fallback is True
    assert seen["name"] == "local_jsonl"
    assert isinstance(seen["kwargs"], dict)
    assert seen["kwargs"].get("file") == "x.jsonl"


def test_extract_model_load_kwargs_filters_core_fields() -> None:
    class _Cfg:
        def __init__(self):
            self.model = SimpleNamespace(id="m")

        def model_dump(self):
            return {
                "model": {
                    "id": "m",
                    "adapter": "a",
                    "device": "cpu",
                    "dtype": "float16",
                    "foo": None,
                }
            }

    out = run_mod._extract_model_load_kwargs(_Cfg())
    assert out == {"dtype": "float16"}


def test_load_model_with_cfg_passes_filtered_kwargs() -> None:
    class _Cfg:
        def __init__(self):
            self.model = SimpleNamespace(id="m")

        def model_dump(self):
            return {"model": {"id": "m", "adapter": "a", "device": "cpu", "foo": 1}}

    calls: dict[str, object] = {}

    class _Adapter:
        def load_model(self, model_id: str, device: str, **kwargs):  # noqa: ANN001
            calls["model_id"] = model_id
            calls["device"] = device
            calls["kwargs"] = dict(kwargs)
            return object()

    run_mod._load_model_with_cfg(_Adapter(), _Cfg(), device="cpu")
    assert calls["model_id"] == "m"
    assert calls["device"] == "cpu"
    assert isinstance(calls["kwargs"], dict)
    assert calls["kwargs"].get("foo") == 1


def test_extract_model_load_kwargs_covers_unknown_dtype_normalization() -> None:
    class _Cfg:
        def model_dump(self):
            return {"model": {"id": "m", "adapter": "a", "dtype": "Float8"}}

    out = run_mod._extract_model_load_kwargs(_Cfg())
    assert out == {"dtype": "float8"}


def test_extract_model_load_kwargs_keeps_blank_dtype_string() -> None:
    class _Cfg:
        def model_dump(self):
            return {"model": {"id": "m", "adapter": "a", "dtype": " "}}

    out = run_mod._extract_model_load_kwargs(_Cfg())
    assert out == {"dtype": " "}


def test_load_model_with_cfg_passes_allowed_kwargs_for_strict_signature() -> None:
    class _Cfg:
        model = SimpleNamespace(id="m")

        def model_dump(self):
            return {"model": {"id": "m", "adapter": "a", "revision": "main"}}

    calls: dict[str, object] = {}

    class _Adapter:
        def load_model(  # noqa: ANN001
            self, model_id: str, device: str, revision: str | None = None
        ):
            calls["model_id"] = model_id
            calls["device"] = device
            calls["revision"] = revision
            return object()

    run_mod._load_model_with_cfg(_Adapter(), _Cfg(), device="cpu")
    assert calls == {"model_id": "m", "device": "cpu", "revision": "main"}


def test_load_model_with_cfg_uses_model_dump_fallback_when_cfg_model_id_raises() -> (
    None
):
    class _Model:
        @property
        def id(self) -> str:
            raise RuntimeError("boom")

    class _Cfg:
        model = _Model()

        def model_dump(self):
            return {"model": {"id": "m", "adapter": "a"}}

    calls: dict[str, object] = {}

    class _Adapter:
        def load_model(self, model_id: str, device: str):  # noqa: ANN001
            calls["model_id"] = model_id
            calls["device"] = device
            return object()

    run_mod._load_model_with_cfg(_Adapter(), _Cfg(), device="cpu")
    assert calls == {"model_id": "m", "device": "cpu"}


def test_load_model_with_cfg_raises_when_model_dump_fails() -> None:
    class _Model:
        @property
        def id(self) -> str:
            raise RuntimeError("boom")

    class _Cfg:
        model = _Model()

        def model_dump(self):
            raise RuntimeError("boom")

    class _Adapter:
        def load_model(self, model_id: str, device: str):  # noqa: ANN001,ARG002
            return object()

    with pytest.raises(ValueError, match="model.id"):
        run_mod._load_model_with_cfg(_Adapter(), _Cfg(), device="cpu")


def test_load_model_with_cfg_raises_when_model_id_missing() -> None:
    class _Cfg:
        model = SimpleNamespace()

        def model_dump(self):
            return {"model": {}}

    class _Adapter:
        def load_model(self, model_id: str, device: str):  # noqa: ANN001,ARG002
            return object()

    with pytest.raises(ValueError, match="model.id"):
        run_mod._load_model_with_cfg(_Adapter(), _Cfg(), device="cpu")


def test_load_model_with_cfg_falls_back_when_signature_introspection_fails(
    monkeypatch,
) -> None:
    class _Cfg:
        model = SimpleNamespace(id="m")

        def model_dump(self):
            return {"model": {"id": "m", "adapter": "a", "foo": 1}}

    calls: list[tuple[str, str]] = []

    class _Adapter:
        def load_model(self, model_id: str, device: str):  # noqa: ANN001
            calls.append((model_id, device))
            return object()

    monkeypatch.setattr(run_mod.inspect, "signature", lambda _fn: 1 / 0)
    run_mod._load_model_with_cfg(_Adapter(), _Cfg(), device="cpu")
    assert calls == [("m", "cpu")]


def test_execute_guarded_run_skip_model_load_uses_stub() -> None:
    class DummyRunner:
        def __init__(self) -> None:
            self.model = None

        def execute(self, **kwargs):  # noqa: D401, ARG002
            self.model = kwargs.get("model")
            return {"status": "ok"}

    runner = DummyRunner()
    report, model = run_mod._execute_guarded_run(
        runner=runner,
        adapter=SimpleNamespace(),
        model=None,
        cfg=SimpleNamespace(model=SimpleNamespace(id="dummy")),
        edit_op=SimpleNamespace(name="noop"),
        run_config=SimpleNamespace(),
        guards=[],
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        restore_fn=None,
        resolved_device="cpu",
        console=Console(file=io.StringIO()),
        skip_model_load=True,
    )
    assert isinstance(report, dict)
    assert getattr(model, "name", "") == "guarded_stub_model"
    assert runner.model is model


def test_execute_guarded_run_sets_snapshot_provenance(monkeypatch) -> None:
    sentinel = object()

    class DummyRunner:
        def __init__(self) -> None:
            self.model = None

        def execute(self, **kwargs):  # noqa: D401, ARG002
            self.model = kwargs.get("model")
            return {"status": "ok"}

    monkeypatch.setattr(run_mod, "_load_model_with_cfg", lambda *_a, **_k: sentinel)

    snapshot = {}
    runner = DummyRunner()
    report, model = run_mod._execute_guarded_run(
        runner=runner,
        adapter=SimpleNamespace(),
        model=None,
        cfg=SimpleNamespace(model=SimpleNamespace(id="dummy")),
        edit_op=SimpleNamespace(name="noop"),
        run_config=SimpleNamespace(),
        guards=[],
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        restore_fn=None,
        resolved_device="cpu",
        console=Console(file=io.StringIO()),
        snapshot_provenance=snapshot,
    )
    assert isinstance(report, dict)
    assert model is sentinel
    assert runner.model is sentinel
    assert snapshot.get("reload_path_used") is True


def test_execute_guarded_run_loads_model_without_snapshot_provenance(
    monkeypatch,
) -> None:
    sentinel = object()

    class DummyRunner:
        def __init__(self) -> None:
            self.model = None

        def execute(self, **kwargs):  # noqa: D401, ARG002
            self.model = kwargs.get("model")
            return {"status": "ok"}

    monkeypatch.setattr(run_mod, "_load_model_with_cfg", lambda *_a, **_k: sentinel)

    runner = DummyRunner()
    report, model = run_mod._execute_guarded_run(
        runner=runner,
        adapter=SimpleNamespace(),
        model=None,
        cfg=SimpleNamespace(model=SimpleNamespace(id="dummy")),
        edit_op=SimpleNamespace(name="noop"),
        run_config=SimpleNamespace(),
        guards=[],
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        restore_fn=None,
        resolved_device="cpu",
        console=Console(file=io.StringIO()),
    )
    assert isinstance(report, dict)
    assert model is sentinel
    assert runner.model is sentinel


def test_postprocess_and_summarize_includes_event_path(
    monkeypatch, tmp_path: Path
) -> None:
    captured = io.StringIO()
    console = Console(file=captured)
    run_config = SimpleNamespace(event_path=str(tmp_path / "events.jsonl"))

    monkeypatch.setattr(
        run_mod,
        "_emit_run_artifacts",
        lambda **_k: {"json": str(tmp_path / "report.json")},
    )

    saved = run_mod._postprocess_and_summarize(
        report={},
        run_dir=tmp_path,
        run_config=run_config,
        window_plan=None,
        dataset_meta={},
        match_fraction=None,
        overlap_fraction=None,
        console=console,
    )
    assert saved["json"].endswith("report.json")
    assert "Events:" in captured.getvalue()


def test_postprocess_and_summarize_omits_event_path_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    captured = io.StringIO()
    console = Console(file=captured)
    run_config = SimpleNamespace(event_path=None)

    monkeypatch.setattr(
        run_mod,
        "_emit_run_artifacts",
        lambda **_k: {"json": str(tmp_path / "report.json")},
    )

    saved = run_mod._postprocess_and_summarize(
        report={},
        run_dir=tmp_path,
        run_config=run_config,
        window_plan=None,
        dataset_meta={},
        match_fraction=None,
        overlap_fraction=None,
        console=console,
    )
    assert saved["json"].endswith("report.json")
    assert "Events:" not in captured.getvalue()
