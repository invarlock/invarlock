import json
from pathlib import Path
from types import SimpleNamespace

from invarlock.core.adapter_auto import (
    apply_auto_adapter_if_needed,
    resolve_auto_adapter,
)


def _write_cfg(tmp_path: Path, model_type: str, arch: str) -> Path:
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [arch]}),
        encoding="utf-8",
    )
    return d


def test_resolve_auto_adapter_mistral(tmp_path):
    model_dir = _write_cfg(tmp_path, "mistral", "MistralForCausalLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_resolve_auto_adapter_qwen(tmp_path):
    model_dir = _write_cfg(tmp_path, "qwen2", "Qwen2ForCausalLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_resolve_auto_adapter_mixtral(tmp_path):
    model_dir = _write_cfg(tmp_path, "mixtral", "MixtralForCausalLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_resolve_auto_adapter_llama(tmp_path):
    model_dir = _write_cfg(tmp_path, "llama", "LlamaForCausalLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_resolve_auto_adapter_gemma3(tmp_path):
    model_dir = _write_cfg(tmp_path, "gemma3", "Gemma3ForConditionalGeneration")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_resolve_auto_adapter_gemma4(tmp_path):
    model_dir = _write_cfg(tmp_path, "gemma4", "Gemma4ForConditionalGeneration")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_resolve_auto_adapter_bert(tmp_path):
    model_dir = _write_cfg(tmp_path, "bert", "BertForMaskedLM")
    assert resolve_auto_adapter(str(model_dir)) == "hf_mlm"


def test_resolve_auto_adapter_gpt_fallback(tmp_path):
    model_dir = _write_cfg(tmp_path, "gpt2", "GPT2LMHeadModel")
    assert resolve_auto_adapter(str(model_dir)) == "hf_causal"


def test_apply_auto_adapter_if_needed_updates_cfg(tmp_path):
    model_dir = _write_cfg(tmp_path, "mistral", "MistralForCausalLM")

    class _Cfg:
        def __init__(self, data: dict):
            self._data = data
            model_data = data.get("model", {})
            self.model = SimpleNamespace(
                id=model_data.get("id"),
                adapter=model_data.get("adapter"),
                device=model_data.get("device"),
            )

        def model_dump(self) -> dict:
            return json.loads(json.dumps(self._data))

    cfg = _Cfg(
        {
            "model": {"id": str(model_dir), "adapter": "auto", "device": "cpu"},
            "dataset": {"provider": "synthetic", "seq_len": 32, "stride": 32},
            "eval": {},
            "guards": {"order": ["invariants"]},
            "output": {"dir": str(tmp_path / "runs")},
            "edit": {"name": "quant_rtn", "plan": {"bitwidth": 8}},
        }
    )
    new_cfg = apply_auto_adapter_if_needed(cfg)
    assert new_cfg.model.adapter == "hf_causal"
