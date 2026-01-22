from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.adapter_auto import resolve_auto_adapter


def _write_cfg(tmp: Path, cfg: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_resolve_auto_adapter_detects_quant_families_from_config(
    tmp_path: Path,
) -> None:
    # GPTQ
    _write_cfg(tmp_path, {"quantization_config": {"quant_method": "gptq"}})
    assert resolve_auto_adapter(str(tmp_path)) == "hf_gptq"

    # AWQ
    awq_dir = tmp_path / "awq"
    _write_cfg(awq_dir, {"quantization_config": {"quant_method": "awq"}})
    assert resolve_auto_adapter(str(awq_dir)) == "hf_awq"

    # BitsAndBytes 8-bit
    bnb_dir = tmp_path / "bnb8"
    _write_cfg(bnb_dir, {"quantization_config": {"load_in_8bit": True}})
    assert resolve_auto_adapter(str(bnb_dir)) == "hf_bnb"


def test_resolve_auto_adapter_name_heuristics() -> None:
    assert resolve_auto_adapter("company/model-gptq") == "hf_gptq"
    assert resolve_auto_adapter("org/model-awq") == "hf_awq"
    assert resolve_auto_adapter("org/model-8bit") == "hf_bnb"
    assert resolve_auto_adapter("org/decoder-7b") == "hf_causal"
    assert resolve_auto_adapter("mistralai/Mixtral-8x7B-v0.1") == "hf_causal"
    assert resolve_auto_adapter("org/awesome-bert") == "hf_mlm"
