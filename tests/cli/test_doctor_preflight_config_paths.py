from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import typer

from tests.cli._support_doctor import (
    DummyConsole,
    _install_fake_torch,
    _patch_minimal_doctor_env,
    doctor_mod,
)


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
        "invarlock.core.registry.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor._check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[gptq]" if name == "hf_gptq" else "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[gptq]" if name == "hf_gptq" else "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.list_providers",
        lambda: ["synthetic"],
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers",
        lambda: ["synthetic"],
        raising=False,
    )
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(json_out=True)
    assert exc.value.exit_code == 0
    lines = [
        line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    payload = json.loads(lines[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D006" in codes  # third-party plugin discovery explicitly enabled note


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
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
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
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
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
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
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
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
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
    tiny_report.write_text(
        json.dumps({"context": {"run": {"tiny_relax": True}}}),
        encoding="utf-8",
    )

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
    assert {"D001", "D002", "D004"}.issubset(codes)


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


def test_doctor_config_local_jsonl_object_paths(monkeypatch, capsys, tmp_path):
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


def test_doctor_config_hf_text_object_path(monkeypatch, capsys, tmp_path):
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


def test_doctor_config_hf_text_dict_path(monkeypatch, capsys, tmp_path):
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
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 1


def test_doctor_baseline_quick_check_missing_path_emits_d014(
    monkeypatch, tmp_path, capsys
):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    missing = tmp_path / "missing.json"
    with pytest.raises((SystemExit, typer.Exit)):
        doctor_mod.doctor_command(json_out=True, baseline=str(missing))

    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D014" in codes


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
