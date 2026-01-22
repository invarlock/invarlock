from __future__ import annotations

from invarlock.cli.commands.run import _resolve_metric_and_provider
from invarlock.cli.config import InvarLockConfig
from invarlock.model_profile import detect_model_profile


def test_config_provider_kind_mapping_from_dict():
    cfg = InvarLockConfig(
        {
            "model": {"id": "t5-small", "adapter": "hf_seq2seq"},
            "dataset": {"provider": {"kind": "seq2seq"}},
            "eval": {"metric": {"kind": "ppl_seq2seq"}},
        }
    )
    profile = detect_model_profile(cfg.model.id, adapter=cfg.model.adapter)
    mk, pk, _ = _resolve_metric_and_provider(cfg, profile, resolved_loss_type="seq2seq")
    assert pk == "seq2seq"
    assert mk == "ppl_seq2seq"


def test_metric_resolution_defaults_to_profile_when_auto():
    cfg = InvarLockConfig(
        {
            "model": {"id": "bert-base-uncased", "adapter": "hf_mlm"},
            "dataset": {"provider": {"kind": "text_lm"}},
            "eval": {"metric": {"kind": "auto"}},
        }
    )
    profile = detect_model_profile(cfg.model.id, adapter=cfg.model.adapter)
    mk, pk, _ = _resolve_metric_and_provider(cfg, profile, resolved_loss_type="mlm")
    assert mk == "ppl_mlm"  # profile default
    assert pk == "text_lm"


def test_config_overrides_env_flag_for_metric_kind(monkeypatch):
    monkeypatch.setenv("INVARLOCK_METRIC_V1", "1")
    cfg = InvarLockConfig(
        {
            "model": {"id": "gpt2", "adapter": "hf_causal"},
            "dataset": {"provider": "text_lm"},
            "eval": {"metric": {"kind": "accuracy", "reps": 1234, "ci_level": 0.9}},
        }
    )
    profile = detect_model_profile(cfg.model.id, adapter=cfg.model.adapter)
    mk, pk, opts = _resolve_metric_and_provider(
        cfg, profile, resolved_loss_type="causal"
    )
    assert mk == "accuracy"  # config wins
    assert pk == "text_lm"
    assert int(opts.get("reps", 0)) == 1234
    assert abs(float(opts.get("ci_level", 0.0)) - 0.9) < 1e-9
