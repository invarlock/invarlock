from __future__ import annotations

import json
from pathlib import Path

import invarlock.cli.adapter_auto as mod
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
    # From config.json (RoPE causal)
    rope_model = tmp_path / "rope"
    rope_model.mkdir()
    (rope_model / "config.json").write_text(
        json.dumps({"model_type": "mistral", "architectures": ["MistralForCausalLM"]}),
        encoding="utf-8",
    )
    assert resolve_auto_adapter(rope_model) == "hf_causal"

    # From config.json (BERT)
    bert = tmp_path / "bert"
    bert.mkdir()
    (bert / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertForMaskedLM"]}),
        encoding="utf-8",
    )
    assert resolve_auto_adapter(bert) == "hf_mlm"

    # From config.json (GPT-like)
    gpt = tmp_path / "gpt"
    gpt.mkdir()
    (gpt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    assert resolve_auto_adapter(gpt) == "hf_causal"

    # String heuristic (name contains bert)
    assert resolve_auto_adapter("some/roberta-checkpoint") == "hf_mlm"
    # String heuristic (default)
    assert resolve_auto_adapter("unknown/thing") == "hf_causal"


def test_apply_auto_adapter_if_needed_changes_only_when_auto(tmp_path: Path):
    rope_model = tmp_path / "rope"
    rope_model.mkdir()
    (rope_model / "config.json").write_text(
        json.dumps({"model_type": "mistral", "architectures": ["MistralForCausalLM"]}),
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

    cfg_auto = Cfg({"model": {"adapter": "auto", "id": str(rope_model)}})
    out = apply_auto_adapter_if_needed(cfg_auto)
    assert out.model.adapter == "hf_causal"

    cfg_noauto = Cfg({"model": {"adapter": "hf_causal", "id": "gpt2"}})
    out2 = apply_auto_adapter_if_needed(cfg_noauto)
    assert out2 is cfg_noauto


class _BadQuantConfig(dict):
    def get(self, *_args, **_kwargs):
        raise RuntimeError("broken quant config")


def test_read_local_hf_config_and_quant_detection_paths(
    tmp_path: Path, monkeypatch
) -> None:
    class _BrokenPath:
        def __call__(self, *_args, **_kwargs):
            raise TypeError("bad path")

    monkeypatch.setattr(mod, "Path", _BrokenPath())
    assert mod._read_local_hf_config("whatever") is None

    monkeypatch.setattr(mod, "Path", Path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert mod._read_local_hf_config(model_dir) is None

    assert (
        mod._detect_quant_family_from_cfg({"quantization_config": _BadQuantConfig()})
        is None
    )


def test_resolve_auto_adapter_additional_family_paths(tmp_path: Path) -> None:
    encdec = tmp_path / "encdec"
    encdec.mkdir()
    (encdec / "config.json").write_text(
        json.dumps({"is_encoder_decoder": True}), encoding="utf-8"
    )
    assert mod.resolve_auto_adapter(encdec) == "hf_seq2seq"

    seq2seq = tmp_path / "seq2seq"
    seq2seq.mkdir()
    (seq2seq / "config.json").write_text(
        json.dumps({"architectures": ["BartForConditionalGeneration"]}),
        encoding="utf-8",
    )
    assert mod.resolve_auto_adapter(seq2seq) == "hf_seq2seq"

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "config.json").write_text(
        json.dumps({"model_type": "mystery", "architectures": ["MysteryModel"]}),
        encoding="utf-8",
    )
    assert mod.resolve_auto_adapter(unknown, default="fallback") == "fallback"

    local_export = tmp_path / "onnx"
    local_export.mkdir()
    (local_export / "model.onnx").write_text("fake", encoding="utf-8")
    assert mod.resolve_auto_adapter(local_export) == "hf_causal"

    assert mod.resolve_auto_adapter("org/model-t5-small") == "hf_seq2seq"


def test_resolve_auto_adapter_causal_model_type_only_hints(tmp_path: Path) -> None:
    for model_type in ("llama", "qwen3", "qwen3_moe", "gemma3", "olmo2"):
        model_dir = tmp_path / model_type
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": model_type}),
            encoding="utf-8",
        )
        assert mod.resolve_auto_adapter(model_dir) == "hf_causal"


def test_apply_auto_adapter_if_needed_exception_path() -> None:
    class _BrokenConfig:
        @property
        def model(self):
            raise RuntimeError("broken model")

    cfg = _BrokenConfig()
    assert mod.apply_auto_adapter_if_needed(cfg) is cfg
