from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli import adapter_auto as mod


class _BadQuantConfig(dict):
    def get(self, *_args, **_kwargs):  # pragma: no cover - exercised indirectly
        raise RuntimeError("broken quant config")


def test_read_local_hf_config_and_quant_detection_edge_branches(
    tmp_path, monkeypatch
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


def test_resolve_auto_adapter_additional_family_branches(tmp_path) -> None:
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

    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_text("fake", encoding="utf-8")
    assert mod.resolve_auto_adapter(onnx_dir) == "hf_causal_onnx"

    assert mod.resolve_auto_adapter("org/model-t5-small") == "hf_seq2seq"


def test_apply_auto_adapter_if_needed_exception_branch() -> None:
    class _BrokenConfig:
        @property
        def model(self):  # pragma: no cover - exercised indirectly
            raise RuntimeError("broken model")

    cfg = _BrokenConfig()
    assert mod.apply_auto_adapter_if_needed(cfg) is cfg
