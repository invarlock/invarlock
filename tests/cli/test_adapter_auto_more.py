from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.adapter_auto import (
    _read_local_hf_config,
    apply_auto_adapter_if_needed,
    resolve_auto_adapter,
)


def test_read_local_hf_config_variants(tmp_path: Path):
    # Non-path value → None
    assert _read_local_hf_config("/no/such/path/hopefully") is None
    # Directory with no config.json → None
    d = tmp_path / "empty"
    d.mkdir()
    assert _read_local_hf_config(d) is None
    # Invalid JSON → None
    (d / "config.json").write_text("not-json", encoding="utf-8")
    assert _read_local_hf_config(d) is None
    # Valid mapping → dict
    (d / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
    cfg = _read_local_hf_config(d)
    assert isinstance(cfg, dict) and cfg.get("model_type") == "gpt2"


def test_resolve_auto_adapter_from_config_and_string(tmp_path: Path):
    # From config.json (LLaMA)
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    assert resolve_auto_adapter(llama) == "hf_llama"

    # From config.json (BERT)
    bert = tmp_path / "bert"
    bert.mkdir()
    (bert / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertForMaskedLM"]}),
        encoding="utf-8",
    )
    assert resolve_auto_adapter(bert) == "hf_bert"

    # From config.json (GPT-like)
    gpt = tmp_path / "gpt"
    gpt.mkdir()
    (gpt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    assert resolve_auto_adapter(gpt) == "hf_gpt2"

    # String heuristic (name contains bert)
    assert resolve_auto_adapter("some/roberta-checkpoint") == "hf_bert"
    # String heuristic (default)
    assert resolve_auto_adapter("unknown/thing") == "hf_gpt2"


def test_apply_auto_adapter_if_needed_changes_only_when_auto(tmp_path: Path):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )

    class Cfg:
        def __init__(self, data):
            self.model = type(
                "M",
                (),
                {"adapter": data["model"]["adapter"], "id": data["model"]["id"]},
            )()

        def model_dump(self):
            return {"model": {"adapter": self.model.adapter, "id": self.model.id}}

        def __class__(self, data):  # type: ignore[override]
            return Cfg(data)

    cfg_auto = Cfg({"model": {"adapter": "auto", "id": str(llama)}})
    out = apply_auto_adapter_if_needed(cfg_auto)
    assert out.model.adapter == "hf_llama"

    cfg_noauto = Cfg({"model": {"adapter": "hf_gpt2", "id": "gpt2"}})
    out2 = apply_auto_adapter_if_needed(cfg_noauto)
    assert out2 is cfg_noauto
