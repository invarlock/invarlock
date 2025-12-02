import json
from pathlib import Path

from invarlock.cli.adapter_auto import (
    apply_auto_adapter_if_needed,
    resolve_auto_adapter,
)
from invarlock.cli.config import InvarLockConfig


def _write_cfg(tmp_path: Path, model_type: str, arch: str) -> Path:
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [arch]}),
        encoding="utf-8",
    )
    return d


def test_resolve_auto_adapter_llama(tmp_path):
    model_dir = _write_cfg(tmp_path, "llama", "LlamaForCausalLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_llama"


def test_resolve_auto_adapter_bert(tmp_path):
    model_dir = _write_cfg(tmp_path, "bert", "BertForMaskedLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_bert"


def test_resolve_auto_adapter_gpt_fallback(tmp_path):
    model_dir = _write_cfg(tmp_path, "gpt2", "GPT2LMHeadModel")
    assert resolve_auto_adapter(str(model_dir)) == "hf_gpt2"


def test_apply_auto_adapter_if_needed_updates_cfg(tmp_path):
    model_dir = _write_cfg(tmp_path, "llama", "LlamaForCausalLM")
    cfg = InvarLockConfig(
        {
            "model": {"id": str(model_dir), "adapter": "auto", "device": "cpu"},
            "dataset": {"provider": "synthetic", "seq_len": 32, "stride": 32},
            "eval": {},
            "guards": {"order": ["invariants"]},
            "output": {"dir": str(tmp_path / "runs")},
            "edit": {"name": "quant_rtn", "plan": {"bits": 8}},
        }
    )
    new_cfg = apply_auto_adapter_if_needed(cfg)
    assert new_cfg.model.adapter == "hf_llama"
