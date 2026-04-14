from __future__ import annotations

import builtins
import importlib.machinery
import json
import sys
import types
from types import SimpleNamespace

import pytest
import typer

from invarlock.cli.commands import doctor as doctor_mod


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(arg) for arg in args))


def _install_fake_torch(monkeypatch, *, cuda_available: bool) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    class FakeProps:
        def __init__(self, total_memory=8 * 1e9, name="FakeGPU"):
            self.total_memory = total_memory
            self.device_name = name
            self.memory_total = f"{total_memory / 1e9:.1f} GB"

    class FakeCuda:
        def is_available(self):
            return cuda_available

        def device_count(self):
            return 1

        def get_device_properties(self, idx):
            return FakeProps()

    torch_mod.__version__ = "0.0.0"
    torch_mod.cuda = FakeCuda()
    torch_mod.version = SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def _patch_minimal_doctor_env(monkeypatch) -> None:
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.adapters",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(name=name),
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {"auto_selected": "cpu", "cpu": {"available": True, "info": "ok"}},
        raising=False,
    )


def _mk_report(
    *,
    tokenizer=None,
    masking=None,
    split=None,
    pm_kind=None,
    counts_source=None,
    estimated=None,
) -> dict:
    prov: dict[str, object] = {}
    if tokenizer is not None or masking is not None:
        prov["provider_digest"] = {}
        if tokenizer is not None:
            prov["provider_digest"]["tokenizer_sha256"] = tokenizer
        if masking is not None:
            prov["provider_digest"]["masking_sha256"] = masking
    if split is not None:
        prov["dataset_split"] = split
    metrics = {}
    if pm_kind is not None:
        metrics = {"primary_metric": {"kind": pm_kind}}
        if counts_source is not None:
            metrics["primary_metric"]["counts_source"] = counts_source
        if estimated is not None:
            metrics["primary_metric"]["estimated"] = estimated
    return {"provenance": prov, "metrics": metrics}


def _setup_config_env(monkeypatch, cfg):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)

    def _fake_tokenizer():
        return SimpleNamespace(__class__=SimpleNamespace(__name__="FakeTokenizer"))

    monkeypatch.setattr(
        doctor_mod,
        "load_config",
        lambda path: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod,
        "apply_profile",
        lambda cfg_obj, profile: cfg_obj,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config",
        lambda path: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
        lambda cfg_obj, profile: cfg_obj,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda model_id, adapter: SimpleNamespace(default_loss="mlm"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda profile: (_fake_tokenizer(), "tok-hash"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda cfg_obj, model_profile, resolved_loss_type=None: (
            "mlm",
            "synthetic",
            {},
        ),
        raising=False,
    )

    class _ProviderStub:
        def estimate_capacity(self, **kwargs):
            return {
                "available_nonoverlap": 2,
                "tokens_available": 10,
                "examples_available": 1,
            }

    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda kind: _ProviderStub(), raising=False
    )


def test_doctor_config_capacity_floors(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "wikitext2", "workers": 0, "deterministic_shards": True},
            seq_len=128,
            stride=64,
            preview_n=100,
            final_n=100,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=400)),
    )

    class CapacityProvider:
        def estimate_capacity(self, **kwargs):
            return {
                "available_nonoverlap": 5,
                "tokens_available": 100,
                "examples_available": 5,
            }

    def fake_tokenizer():
        return SimpleNamespace(__class__=SimpleNamespace(__name__="FakeTokenizer"))

    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.apply_profile",
        lambda cfg, profile: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
        lambda cfg, profile: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda model_id, adapter: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda profile: (fake_tokenizer(), "tok-hash"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda cfg, model_profile, resolved_loss_type=None: (
            "ppl_causal",
            "synthetic",
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda kind: CapacityProvider(),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_device_info",
        lambda: {
            "cpu": {"available": True, "info": "Available"},
            "cuda": {"available": False, "info": "Missing"},
            "auto_selected": "cpu",
        },
        raising=False,
    )
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.adapters",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers", lambda: ["synthetic"], raising=False
    )

    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(json_out=True, config=str(tmp_path / "cfg.yaml"))
    assert exc.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert {"D007", "D008"}.issubset(codes)


def test_doctor_non_json_device_and_optional_paths(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)

    dummy_console = DummyConsole()
    monkeypatch.setattr(doctor_mod, "console", dummy_console, raising=False)

    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {
            "auto_selected": "cuda",
            "cpu": {"available": True, "info": "Available"},
            "cuda": {
                "available": True,
                "device_count": 1,
                "device_name": "FakeGPU",
                "memory_total": "8 GB",
            },
            "mps": {"available": False, "info": "Missing"},
        },
        raising=False,
    )

    fake_registry = SimpleNamespace(
        list_adapters=lambda: ["hf_causal", "hf_bnb"],
        list_edits=lambda: ["quant_rtn"],
        list_guards=lambda: ["spectral"],
        get_plugin_info=lambda name, kind: {
            "module": f"invarlock.{kind}.{name}",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry", lambda: fake_registry, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers",
        lambda: ["synthetic"],
        raising=False,
    )

    def fake_find_spec(name):
        if name in {"auto_gptq", "bitsandbytes"}:
            return None
        return types.SimpleNamespace(name=name)

    monkeypatch.setattr(
        doctor_mod.importlib.util, "find_spec", fake_find_spec, raising=False
    )
    monkeypatch.delenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", raising=False)
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(__version__="1.0")
    )

    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(profile="ci")
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 0
    assert any("Optional Dependencies" in line for line in dummy_console.lines)
    assert any("Plugin Registry" in line for line in dummy_console.lines)
    assert all("Legend:" not in line for line in dummy_console.lines)
    assert all("Hints:" not in line for line in dummy_console.lines)


def test_doctor_determinism_warning_prints(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch, cuda_available=True)
    dummy_console = DummyConsole()
    monkeypatch.setattr(doctor_mod, "console", dummy_console, raising=False)

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "wikitext2", "workers": 4, "deterministic_shards": False},
            seq_len=16,
            stride=8,
            preview_n=2,
            final_n=2,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=400)),
    )

    class ProviderNoEstimate:
        pass

    def fake_tokenizer():
        return SimpleNamespace(__class__=SimpleNamespace(__name__="FakeTokenizer"))

    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.apply_profile",
        lambda cfg, profile: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
        lambda cfg, profile: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda model_id, adapter: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda profile: (fake_tokenizer(), "tok-hash"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda cfg, model_profile, resolved_loss_type=None: (
            "accuracy",
            "synthetic",
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda kind: ProviderNoEstimate(),
        raising=False,
    )
    monkeypatch.setattr("invarlock.eval.data.list_providers", lambda: [], raising=False)
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.adapters",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(name=name),
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {"auto_selected": "cpu", "cpu": {"available": True, "info": "ok"}},
        raising=False,
    )

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("dummy", encoding="utf-8")

    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(config=str(cfg_path))
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 0
    assert any(
        doctor_mod.DETERMINISM_SHARDS_WARNING in line for line in dummy_console.lines
    )


def test_doctor_handles_missing_torch(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {"auto_selected": "cpu", "cpu": {"available": True, "info": "ok"}},
        raising=False,
    )
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.adapters",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry", lambda: fake_registry, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(name=name),
        raising=False,
    )

    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(json_out=True)
    assert exc.value.exit_code == 1
