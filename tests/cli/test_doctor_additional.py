from __future__ import annotations

import builtins
import io
import json
import sys
import types
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from invarlock.cli.commands import doctor as doctor_mod


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(arg) for arg in args))


def _install_fake_torch(monkeypatch, *, cuda_available: bool) -> None:
    torch_mod = types.ModuleType("torch")

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
        "invarlock.cli.config.load_config",
        lambda path: cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile",
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
        "invarlock.cli.commands.run._resolve_metric_and_provider",
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


def test_doctor_json_optional_deps_and_registry(monkeypatch, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {
            "cpu": {"available": True, "info": "Available"},
            "cuda": {"available": False, "info": "Missing"},
            "mps": {"available": False, "info": "Missing"},
            "auto_selected": "cpu",
        },
        raising=False,
    )

    orig_find_spec = doctor_mod.importlib.util.find_spec
    orig_find_spec = doctor_mod.importlib.util.find_spec
    overrides = {
        "datasets": SimpleNamespace(),
        "transformers": SimpleNamespace(),
        "bitsandbytes": SimpleNamespace(),
    }

    def fake_find_spec(name):
        base = name.replace("-", "_")
        if base in overrides:
            return overrides[base]
        return orig_find_spec(name)

    monkeypatch.setattr(
        doctor_mod.importlib.util, "find_spec", fake_find_spec, raising=False
    )

    fake_registry = SimpleNamespace(
        list_adapters=lambda: ["hf_bnb", "hf_gptq"],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.plugins.fake",
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
        lambda name, kind: "⚠️ missing invarlock[gptq]" if name == "hf_gptq" else "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.list_providers",
        lambda: ["synthetic"],
        raising=False,
    )
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(json_out=True)
    assert exc.value.exit_code == 0
    lines = [
        line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    payload = json.loads(lines[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D006" in codes  # plugin discovery disabled note


def test_doctor_config_preflight_findings(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)

    missing_path = tmp_path / "missing.jsonl"
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={
                "kind": "unknown_kind",
                "file": str(missing_path),
                "text_field": "",
                "workers": 2,
                "deterministic_shards": False,
            },
            seq_len=32,
            stride=16,
            preview_n=2,
            final_n=2,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cuda"),
        runner=SimpleNamespace(device="cuda"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=10)),
    )

    class FakeProvider:
        def estimate_capacity(self, **kwargs):
            return {
                "available_nonoverlap": 2,
                "tokens_available": 10,
                "examples_available": 1,
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
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
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
        "invarlock.cli.commands.run._resolve_metric_and_provider",
        lambda cfg, model_profile, resolved_loss_type=None: (
            "accuracy",
            "synthetic",
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda kind: FakeProvider(), raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_device_info",
        lambda: {
            "cpu": {"available": True, "info": "Available"},
            "cuda": {"available": False, "info": "Missing"},
            "mps": {"available": False, "info": "Missing"},
            "auto_selected": "cpu",
        },
        raising=False,
    )
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.plugins.fake",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers", lambda: ["synthetic"], raising=False
    )

    tiny_report = tmp_path / "tiny.json"
    tiny_report.write_text(json.dumps({"auto": {"tiny_relax": True}}), encoding="utf-8")

    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            config=str(tmp_path / "cfg.yaml"),
            baseline_report=str(tiny_report),
            tier="balanced",
        )
    assert exc.value.exit_code == 1  # errors present
    lines = [
        line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    payload = json.loads(lines[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert {"D001", "D002", "D004", "D013"}.issubset(codes)


def test_doctor_config_provider_string_kind_error(monkeypatch, capsys, tmp_path):
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider="mystery_provider",
            seq_len=16,
            stride=16,
            preview_n=1,
            final_n=1,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=1)),
    )
    _setup_config_env(monkeypatch, cfg)
    with pytest.raises(typer.Exit):
        doctor_mod.doctor_command(json_out=True, config=str(tmp_path / "cfg.yaml"))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    codes = {item["code"] for item in payload.get("findings", [])}
    assert "D001" in codes


def test_doctor_config_local_jsonl_object_branches(monkeypatch, capsys, tmp_path):
    class LocalProvider:
        kind = "local_jsonl"
        file = tmp_path / "missing.jsonl"
        text_field = ""
        workers = 1
        deterministic_shards = False

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider=LocalProvider(),
            seq_len=32,
            stride=16,
            preview_n=1,
            final_n=1,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=1)),
    )
    _setup_config_env(monkeypatch, cfg)
    with pytest.raises(typer.Exit):
        doctor_mod.doctor_command(json_out=True, config=str(tmp_path / "cfg.yaml"))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    codes = {item["code"] for item in payload.get("findings", [])}
    assert "D011" in codes


def test_doctor_config_hf_text_object_branch(monkeypatch, capsys, tmp_path):
    class HFProvider:
        kind = "hf_text"
        text_field = ""

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider=HFProvider(),
            seq_len=32,
            stride=16,
            preview_n=1,
            final_n=1,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=1)),
    )
    _setup_config_env(monkeypatch, cfg)
    with pytest.raises(typer.Exit):
        doctor_mod.doctor_command(json_out=True, config=str(tmp_path / "cfg.yaml"))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    codes = {item["code"] for item in payload.get("findings", [])}
    assert "D004" in codes


def test_doctor_config_hf_text_dict_branch(monkeypatch, capsys, tmp_path):
    provider = {
        "kind": "hf_text",
        "text_field": "instruction",
    }
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider=provider,
            seq_len=32,
            stride=16,
            preview_n=1,
            final_n=1,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=1)),
    )
    _setup_config_env(monkeypatch, cfg)
    with pytest.raises(typer.Exit):
        doctor_mod.doctor_command(json_out=True, config=str(tmp_path / "cfg.yaml"))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["resolution"]["exit_code"] == 1


def test_doctor_config_baseline_split_console(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "wikitext2"},
            seq_len=16,
            stride=16,
            preview_n=1,
            final_n=1,
        ),
        model=SimpleNamespace(adapter="hf_fake", device="cpu"),
        runner=SimpleNamespace(device="cpu"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=1)),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"provenance": {"split_fallback": True}}), encoding="utf-8"
    )
    _setup_config_env(monkeypatch, cfg)
    dummy_console = DummyConsole()
    monkeypatch.setattr(doctor_mod, "console", dummy_console, raising=False)
    with pytest.raises((SystemExit, typer.Exit)):
        doctor_mod.doctor_command(
            json_out=False,
            config=str(tmp_path / "cfg.yaml"),
            baseline=str(baseline),
        )
    warning_text = doctor_mod.DATASET_SPLIT_FALLBACK_WARNING
    assert any(warning_text in line for line in dummy_console.lines)


def test_doctor_baseline_quick_check_split_warning(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"provenance": {"split_fallback": True}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {
            "auto_selected": "cpu",
            "cpu": {"available": True, "info": "Always"},
        },
        raising=False,
    )
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.adapters",
            "entry_point": "",
        },
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr("invarlock.eval.data.list_providers", lambda: [], raising=False)
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(json_out=True, baseline=str(baseline))
    assert exc.value.exit_code == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D003" in codes


def test_doctor_baseline_quick_check_missing_path(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    missing = tmp_path / "missing.json"
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(json_out=True, baseline=str(missing))
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 0


def test_doctor_baseline_split_warning_console(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"provenance": {"split_fallback": True}}), encoding="utf-8"
    )
    dummy_console = DummyConsole()
    monkeypatch.setattr(doctor_mod, "console", dummy_console, raising=False)
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(json_out=False, baseline=str(baseline))
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 0
    warning_text = doctor_mod.DATASET_SPLIT_FALLBACK_WARNING
    assert any(warning_text in line for line in dummy_console.lines)


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
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
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
        "invarlock.cli.commands.run._resolve_metric_and_provider",
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
        list_adapters=lambda: ["hf_gpt2", "hf_bnb"],
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
    monkeypatch.delenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", raising=False)
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(__version__="1.0")
    )

    with pytest.raises(SystemExit) as exc:
        doctor_mod.doctor_command(profile="ci")
    assert exc.value.code == 0
    assert any("Optional Dependencies" in line for line in dummy_console.lines)
    assert any("Plugin Registry" in line for line in dummy_console.lines)


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
        "invarlock.cli.config.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, profile: cfg, raising=False
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
        "invarlock.cli.commands.run._resolve_metric_and_provider",
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

    with pytest.raises(SystemExit) as exc:
        doctor_mod.doctor_command(config=str(cfg_path))
    assert exc.value.code == 0
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


def test_doctor_cross_checks_tokenizer_mismatch(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(
        json.dumps(_mk_report(tokenizer="tokA", split="validation")), encoding="utf-8"
    )
    subject.write_text(
        json.dumps(_mk_report(tokenizer="tokB", split="validation")), encoding="utf-8"
    )
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
        )
    assert exc.value.exit_code == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D009" in codes


def test_doctor_cross_checks_mask_missing(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(
        json.dumps(_mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm")),
        encoding="utf-8",
    )
    subject.write_text(
        json.dumps(_mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm")),
        encoding="utf-8",
    )
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
        )
    assert exc.value.exit_code == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D010" in codes


def test_doctor_cross_checks_split_mismatch_strict(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(split="validation")), encoding="utf-8")
    subject.write_text(json.dumps(_mk_report(split="test")), encoding="utf-8")
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
            strict=True,
        )
    assert exc.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    entries = [f for f in payload.get("findings", []) if f.get("code") == "D011"]
    assert entries and entries[0]["severity"] == "error"


def test_doctor_cross_checks_accuracy_pseudo_counts(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(split="validation")), encoding="utf-8")
    subject.write_text(
        json.dumps(
            _mk_report(
                split="validation",
                pm_kind="accuracy",
                counts_source="pseudo_config",
                estimated=True,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
            profile="ci",
        )
    assert exc.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    entries = [f for f in payload.get("findings", []) if f.get("code") == "D012"]
    assert entries and entries[0]["severity"] == "error"


def _run_cross_check(tmp_path, baseline_payload, subject_payload, **kwargs):
    baseline = tmp_path / "baseline_cc.json"
    subject = tmp_path / "subject_cc.json"
    baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    subject.write_text(json.dumps(subject_payload), encoding="utf-8")
    calls: list[tuple[str, str]] = []
    console = Console(file=io.StringIO())

    def _capture(code, severity, message, **extra):
        calls.append((code, severity))

    had_error = doctor_mod._cross_check_reports(
        str(baseline),
        str(subject),
        cfg_metric_kind=kwargs.get("cfg_metric_kind"),
        strict=kwargs.get("strict", False),
        profile=kwargs.get("profile"),
        json_out=True,
        console=console,
        add_fn=_capture,
    )
    return had_error, calls


def test_cross_checks_d009_tokenizer(tmp_path):
    had_error, calls = _run_cross_check(
        tmp_path,
        _mk_report(tokenizer="tokA", split="validation"),
        _mk_report(tokenizer="tokB", split="validation"),
    )
    assert not had_error
    assert ("D009", "warning") in calls


def test_cross_checks_d010_missing_mask(tmp_path):
    had_error, calls = _run_cross_check(
        tmp_path,
        _mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm"),
        _mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm"),
        cfg_metric_kind="ppl_mlm",
    )
    assert not had_error
    assert ("D010", "warning") in calls


def test_cross_checks_d011_strict(tmp_path):
    had_error, calls = _run_cross_check(
        tmp_path,
        _mk_report(tokenizer="tokA", split="validation"),
        _mk_report(tokenizer="tokA", split="test"),
        strict=True,
    )
    assert had_error
    assert ("D011", "error") in calls


def test_cross_checks_d012_profile(tmp_path):
    # Dev profile → warning
    had_error_dev, calls_dev = _run_cross_check(
        tmp_path,
        _mk_report(split="validation"),
        _mk_report(
            split="validation",
            pm_kind="accuracy",
            counts_source="pseudo_config",
            estimated=True,
        ),
        profile=None,
    )
    assert not had_error_dev
    assert ("D012", "warning") in calls_dev

    # CI profile → error
    had_error_ci, calls_ci = _run_cross_check(
        tmp_path,
        _mk_report(split="validation"),
        _mk_report(
            split="validation",
            pm_kind="accuracy",
            counts_source="pseudo_config",
            estimated=True,
        ),
        profile="ci",
    )
    assert had_error_ci
    assert ("D012", "error") in calls_ci
