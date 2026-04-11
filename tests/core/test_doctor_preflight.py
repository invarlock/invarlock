from __future__ import annotations

import json
from types import SimpleNamespace

from invarlock.core.doctor_findings import DATASET_SPLIT_FALLBACK_WARNING
from invarlock.core.doctor_preflight import (
    run_doctor_config_preflight,
)


def test_run_doctor_config_preflight_collects_findings(monkeypatch, tmp_path) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={
                "kind": "unknown_kind",
                "file": str(tmp_path / "missing.jsonl"),
                "text_field": "",
                "workers": 2,
                "deterministic_shards": False,
            },
            seq_len=32,
            stride=16,
            preview_n=2,
            final_n=2,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model", device="cuda"),
        runner=SimpleNamespace(device="cuda"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=10)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
        lambda cfg_obj, profile: cfg_obj,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **kwargs: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda profile: (
            SimpleNamespace(__class__=SimpleNamespace(__name__="Tok")),
            "tok",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda cfg_obj, model_profile, resolved_loss_type=None: (
            "accuracy",
            "synthetic",
            {},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda kind: SimpleNamespace(
            estimate_capacity=lambda **kwargs: {
                "available_nonoverlap": 2,
                "tokens_available": 10,
                "examples_available": 1,
            }
        ),
        raising=False,
    )

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    result = run_doctor_config_preflight(
        config_path=str(tmp_path / "cfg.yaml"),
        profile="dev",
        tier="balanced",
        baseline=None,
    )

    codes = {finding.code for finding in result.findings}
    assert {"D001", "D002", "D004", "D007", "D008"}.issubset(codes)
    assert result.had_error is True
    assert result.metric_kind == "accuracy"
    assert any("Metric: accuracy" in line for line in result.lines)


def test_run_doctor_config_preflight_handles_baseline_split_and_metric_failures(
    monkeypatch, tmp_path
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"provenance": {"split_fallback": True}}),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "synthetic", "workers": "many"},
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=400)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_loader.apply_profile",
        lambda cfg_obj, profile: cfg_obj,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )

    result = run_doctor_config_preflight(
        config_path=str(tmp_path / "cfg.yaml"),
        profile="dev",
        tier="balanced",
        baseline=str(baseline),
    )

    codes = {finding.code for finding in result.findings}
    assert "D003" in codes
    assert "D016" in codes
    assert "D017" in codes
    assert result.metric_kind is None
    assert result.had_error is True
    assert DATASET_SPLIT_FALLBACK_WARNING in result.lines


def test_run_doctor_config_preflight_notes_missing_capacity_estimator(
    monkeypatch, tmp_path
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "synthetic"},
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=400)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **kwargs: SimpleNamespace(default_loss="classification"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda profile: (
            SimpleNamespace(__class__=SimpleNamespace(__name__="Tok")),
            "tok",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *args, **kwargs: ("accuracy", "synthetic", {}),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda kind: SimpleNamespace(),
        raising=False,
    )

    result = run_doctor_config_preflight(config_path=str(tmp_path / "cfg.yaml"))

    assert any(
        "Provider does not expose estimate_capacity()" in line for line in result.lines
    )


def test_run_doctor_config_preflight_tolerates_capacity_path_exceptions(
    monkeypatch, tmp_path
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "synthetic"},
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=400)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **kwargs: SimpleNamespace(default_loss="classification"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda profile: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *args, **kwargs: ("accuracy", "synthetic", {}),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda kind: SimpleNamespace(estimate_capacity=lambda **kwargs: {}),
        raising=False,
    )

    result = run_doctor_config_preflight(config_path=str(tmp_path / "cfg.yaml"))

    assert result.metric_kind == "accuracy"
    assert any("Metric: accuracy" in line for line in result.lines)
    assert {finding.code for finding in result.findings} >= {"D018"}
    assert result.had_error is True


def test_run_doctor_config_preflight_handles_worker_profile_and_metric_failures(
    monkeypatch, tmp_path
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={
                "kind": "synthetic",
                "workers": object(),
                "deterministic_shards": False,
            },
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=10)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )

    result = run_doctor_config_preflight(config_path=str(tmp_path / "cfg.yaml"))

    assert result.metric_kind is None
    assert result.had_error is True
    assert {finding.code for finding in result.findings} >= {"D015", "D016", "D017"}
    assert "D002" not in {finding.code for finding in result.findings}


def test_run_doctor_config_preflight_skips_split_warning_for_clean_baseline(
    monkeypatch, tmp_path
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"provenance": {}}), encoding="utf-8")
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "synthetic"},
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=10)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *args, **kwargs: ("accuracy", None, {}),
        raising=False,
    )

    result = run_doctor_config_preflight(
        config_path=str(tmp_path / "cfg.yaml"),
        baseline=str(baseline),
    )

    assert DATASET_SPLIT_FALLBACK_WARNING not in result.lines


def test_run_doctor_config_preflight_tolerates_provider_lookup_exceptions(
    monkeypatch, tmp_path
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "synthetic"},
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model"),
        eval=SimpleNamespace(bootstrap=SimpleNamespace(replicates=10)),
    )

    monkeypatch.setattr(
        "invarlock.core.config_loader.load_config", lambda path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **kwargs: SimpleNamespace(default_loss="classification"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *args, **kwargs: ("accuracy", "synthetic", {}),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda kind: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )

    result = run_doctor_config_preflight(config_path=str(tmp_path / "cfg.yaml"))

    assert result.metric_kind == "accuracy"
    assert any("Metric: accuracy" in line for line in result.lines)
    assert {finding.code for finding in result.findings} >= {"D018"}
    assert result.had_error is True
