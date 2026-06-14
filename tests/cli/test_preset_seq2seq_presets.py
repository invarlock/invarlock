from __future__ import annotations

from pathlib import Path

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_seq2seq_family_presets_load() -> None:
    root = _repo_root()
    synthetic_presets = {
        "synth_64.yaml": 64,
        "synth_128.yaml": 128,
    }
    for name, seq_len in synthetic_presets.items():
        cfg = load_config(root / "configs/presets/seq2seq" / name)
        assert cfg.require_section("model")["id"] == "t5-small"
        assert cfg.require_section("model")["adapter"] == "hf_seq2seq"
        assert cfg.require_section("dataset")["provider"] == "seq2seq"
        assert cfg.require_section("dataset")["seq_len"] == seq_len
        assert cfg.require_section("eval")["metric"]["kind"] == "ppl_seq2seq"
        assert cfg.require_section("eval")["loss"]["type"] == "seq2seq"

    cfg = load_config(
        root / "configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml"
    )
    assert cfg.require_section("model")["id"] == "google/flan-t5-base"
    assert cfg.require_section("model")["adapter"] == "hf_seq2seq"
    assert cfg.require_section("model")["revision"]
    provider = cfg.require_section("dataset")["provider"]
    assert provider["kind"] == "hf_seq2seq"
    assert provider["dataset_name"] == "abisee/cnn_dailymail"
    assert provider["config_name"] == "3.0.0"
    assert provider["revision"]
    assert provider["src_prefix"] == "summarize: "
    assert cfg.require_section("dataset")["seq_len"] == 256
    assert cfg.require_section("eval")["metric"]["kind"] == "ppl_seq2seq"
    assert cfg.require_section("eval")["loss"]["type"] == "seq2seq"
