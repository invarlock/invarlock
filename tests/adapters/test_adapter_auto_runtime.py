from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_auto_module(monkeypatch) -> types.ModuleType:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "src"))

    import invarlock

    adapters_pkg = types.ModuleType("invarlock.adapters")
    adapters_pkg.__path__ = [str(root / "src" / "invarlock" / "adapters")]
    monkeypatch.setitem(sys.modules, "invarlock.adapters", adapters_pkg)
    monkeypatch.setattr(invarlock, "adapters", adapters_pkg, raising=False)

    sys.modules.pop("invarlock.adapters.auto", None)
    return importlib.import_module("invarlock.adapters.auto")


def _delegate_class(label: str):
    class _Delegate:
        def __init__(self) -> None:
            self.label = label
            self.restores: list[tuple[object, bytes]] = []

        def can_handle(self, _model) -> bool:
            return True

        def describe(self, model) -> dict[str, object]:
            return {"label": self.label, "model": model}

        def snapshot(self, model) -> bytes:
            return f"{self.label}:{model}".encode()

        def restore(self, model, blob: bytes) -> None:
            self.restores.append((model, blob))

        def load_model(self, model_id: str, device: str = "auto", **kwargs):
            return {
                "label": self.label,
                "model_id": model_id,
                "device": device,
                "kwargs": kwargs,
            }

    return _Delegate


def test_detect_quantization_from_path_variants(tmp_path: Path, monkeypatch) -> None:
    auto_mod = _load_auto_module(monkeypatch)

    assert auto_mod._detect_quantization_from_path(str(tmp_path / "missing")) is None

    no_config = tmp_path / "no-config"
    no_config.mkdir()
    assert auto_mod._detect_quantization_from_path(str(no_config)) is None

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "config.json").write_text("not-json", encoding="utf-8")
    assert auto_mod._detect_quantization_from_path(str(invalid)) is None

    non_mapping = tmp_path / "non-mapping"
    non_mapping.mkdir()
    (non_mapping / "config.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert auto_mod._detect_quantization_from_path(str(non_mapping)) is None

    empty_quant = tmp_path / "empty-quant"
    empty_quant.mkdir()
    (empty_quant / "config.json").write_text(
        json.dumps({"quantization_config": {}}), encoding="utf-8"
    )
    assert auto_mod._detect_quantization_from_path(str(empty_quant)) is None

    bad_quant_cfg = tmp_path / "bad-quant-cfg"
    bad_quant_cfg.mkdir()
    (bad_quant_cfg / "config.json").write_text(
        json.dumps({"quantization_config": ["bad"]}), encoding="utf-8"
    )
    assert auto_mod._detect_quantization_from_path(str(bad_quant_cfg)) is None

    non_string_method = tmp_path / "non-string-method"
    non_string_method.mkdir()
    (non_string_method / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": 17}}), encoding="utf-8"
    )
    assert auto_mod._detect_quantization_from_path(str(non_string_method)) is None

    awq = tmp_path / "awq"
    awq.mkdir()
    (awq / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "awq"}}),
        encoding="utf-8",
    )
    assert auto_mod._detect_quantization_from_path(str(awq)) == "hf_awq"

    gptq = tmp_path / "gptq"
    gptq.mkdir()
    (gptq / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "GPTQ"}}),
        encoding="utf-8",
    )
    assert auto_mod._detect_quantization_from_path(str(gptq)) == "hf_gptq"

    bnb = tmp_path / "bnb"
    bnb.mkdir()
    (bnb / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "bnb-4bit"}}),
        encoding="utf-8",
    )
    assert auto_mod._detect_quantization_from_path(str(bnb)) == "hf_bnb"

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "marlin"}}),
        encoding="utf-8",
    )
    assert auto_mod._detect_quantization_from_path(str(unknown)) is None


def test_detect_quantization_from_model_variants(monkeypatch) -> None:
    auto_mod = _load_auto_module(monkeypatch)

    assert auto_mod._detect_quantization_from_model(object()) is None

    loaded_in_8bit = SimpleNamespace(
        config=SimpleNamespace(quantization_config=None),
        is_loaded_in_8bit=True,
    )
    assert auto_mod._detect_quantization_from_model(loaded_in_8bit) == "hf_bnb"

    loaded_in_4bit = SimpleNamespace(
        config=SimpleNamespace(quantization_config=None),
        is_loaded_in_4bit=True,
    )
    assert auto_mod._detect_quantization_from_model(loaded_in_4bit) == "hf_bnb"

    plain = SimpleNamespace(config=SimpleNamespace(quantization_config=None))
    assert auto_mod._detect_quantization_from_model(plain) is None

    dict_awq = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "awq"})
    )
    assert auto_mod._detect_quantization_from_model(dict_awq) == "hf_awq"

    dict_gptq = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "gptq"})
    )
    assert auto_mod._detect_quantization_from_model(dict_gptq) == "hf_gptq"

    dict_bnb = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "bitsandbytes"})
    )
    assert auto_mod._detect_quantization_from_model(dict_bnb) == "hf_bnb"

    dict_unknown = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "marlin"})
    )
    assert auto_mod._detect_quantization_from_model(dict_unknown) is None

    dict_bad = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": 42})
    )
    assert auto_mod._detect_quantization_from_model(dict_bad) is None

    awq_cfg = type("AWQConfig", (), {})
    awq_model = SimpleNamespace(config=SimpleNamespace(quantization_config=awq_cfg()))
    assert auto_mod._detect_quantization_from_model(awq_model) == "hf_awq"

    gptq_cfg = type("GPTQConfig", (), {})
    gptq_model = SimpleNamespace(config=SimpleNamespace(quantization_config=gptq_cfg()))
    assert auto_mod._detect_quantization_from_model(gptq_model) == "hf_gptq"

    bnb_cfg = type("BitsAndBytesConfig", (), {})
    bnb_model = SimpleNamespace(config=SimpleNamespace(quantization_config=bnb_cfg()))
    assert auto_mod._detect_quantization_from_model(bnb_model) == "hf_bnb"

    short_bnb_cfg = type("BnbConfig", (), {})
    short_bnb_model = SimpleNamespace(
        config=SimpleNamespace(quantization_config=short_bnb_cfg())
    )
    assert auto_mod._detect_quantization_from_model(short_bnb_model) == "hf_bnb"

    other_cfg = type("UnknownConfig", (), {})
    other_model = SimpleNamespace(
        config=SimpleNamespace(quantization_config=other_cfg())
    )
    assert auto_mod._detect_quantization_from_model(other_model) is None


def test_load_adapter_dispatches_all_known_adapter_names(monkeypatch) -> None:
    auto_mod = _load_auto_module(monkeypatch)

    delegate_specs = {
        ".hf_causal": ("HF_Causal_Adapter", "hf_causal"),
        ".hf_mlm": ("HF_MLM_Adapter", "hf_mlm"),
        ".hf_multimodal": ("HF_Multimodal_Adapter", "hf_multimodal"),
        ".hf_seq2seq": ("HF_Seq2Seq_Adapter", "hf_seq2seq"),
        "invarlock.plugins.hf_bnb_adapter": ("HF_BNB_Adapter", "hf_bnb"),
        "invarlock.plugins.hf_awq_adapter": ("HF_AWQ_Adapter", "hf_awq"),
        "invarlock.plugins.hf_gptq_adapter": ("HF_GPTQ_Adapter", "hf_gptq"),
    }

    def _fake_import(name: str, package: str | None = None):
        attr_name, label = delegate_specs[name]
        return SimpleNamespace(**{attr_name: _delegate_class(label)})

    monkeypatch.setattr(auto_mod._importlib, "import_module", _fake_import)

    adapter = auto_mod.HF_Auto_Adapter()
    assert adapter.name == "hf_auto"

    assert adapter._load_adapter("hf_causal").label == "hf_causal"
    assert adapter._load_adapter("hf_mlm").label == "hf_mlm"
    assert adapter._load_adapter("hf_multimodal").label == "hf_multimodal"
    assert adapter._load_adapter("hf_seq2seq").label == "hf_seq2seq"
    assert adapter._load_adapter("hf_bnb").label == "hf_bnb"
    assert adapter._load_adapter("hf_awq").label == "hf_awq"
    assert adapter._load_adapter("hf_gptq").label == "hf_gptq"
    assert adapter._load_adapter("something-else").label == "hf_causal"


def test_ensure_delegate_from_id_prefers_cache_quantization_and_resolution(
    monkeypatch,
) -> None:
    auto_mod = _load_auto_module(monkeypatch)

    adapter = auto_mod.HF_Auto_Adapter()
    cached = _delegate_class("cached")()
    adapter._delegate = cached
    assert adapter._ensure_delegate_from_id("demo/model") is cached

    quantized = auto_mod.HF_Auto_Adapter()
    load_calls: list[str] = []
    monkeypatch.setattr(
        auto_mod,
        "_detect_quantization_from_path",
        lambda model_id: "hf_awq" if model_id == "quantized" else None,
    )
    monkeypatch.setattr(
        auto_mod, "resolve_auto_adapter", lambda model_id: f"resolved:{model_id}"
    )
    monkeypatch.setattr(
        quantized,
        "_load_adapter",
        lambda name: load_calls.append(name) or _delegate_class(name)(),
    )

    quant_delegate = quantized._ensure_delegate_from_id("quantized")
    assert quant_delegate.label == "hf_awq"
    assert load_calls == ["hf_awq"]

    resolved = auto_mod.HF_Auto_Adapter()
    monkeypatch.setattr(
        resolved,
        "_load_adapter",
        lambda name: load_calls.append(name) or _delegate_class(name)(),
    )

    resolved_delegate = resolved._ensure_delegate_from_id("plain-model")
    assert resolved_delegate.label == "resolved:plain-model"
    assert load_calls[-1] == "resolved:plain-model"


def test_ensure_delegate_from_model_prefers_quantization_and_fallbacks(
    monkeypatch,
) -> None:
    auto_mod = _load_auto_module(monkeypatch)

    adapter = auto_mod.HF_Auto_Adapter()
    cached = _delegate_class("cached")()
    adapter._delegate = cached
    assert adapter._ensure_delegate_from_model(object()) is cached

    load_calls: list[str] = []

    def _load(name: str):
        load_calls.append(name)
        return _delegate_class(name)()

    quantized = auto_mod.HF_Auto_Adapter()
    monkeypatch.setattr(
        auto_mod,
        "_detect_quantization_from_model",
        lambda model: "hf_gptq" if getattr(model, "kind", None) == "quant" else None,
    )
    monkeypatch.setattr(quantized, "_load_adapter", _load)
    quant_delegate = quantized._ensure_delegate_from_model(
        SimpleNamespace(kind="quant")
    )
    assert quant_delegate.label == "hf_gptq"

    bert_model = type("BertEncoder", (), {"config": SimpleNamespace()})()
    bert_adapter = auto_mod.HF_Auto_Adapter()
    monkeypatch.setattr(bert_adapter, "_load_adapter", _load)
    bert_delegate = bert_adapter._ensure_delegate_from_model(bert_model)
    assert bert_delegate.label == "hf_mlm"

    seq2seq_model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True))
    seq2seq_adapter = auto_mod.HF_Auto_Adapter()
    monkeypatch.setattr(seq2seq_adapter, "_load_adapter", _load)
    seq2seq_delegate = seq2seq_adapter._ensure_delegate_from_model(seq2seq_model)
    assert seq2seq_delegate.label == "hf_seq2seq"

    causal_adapter = auto_mod.HF_Auto_Adapter()
    monkeypatch.setattr(causal_adapter, "_load_adapter", _load)
    causal_delegate = causal_adapter._ensure_delegate_from_model(
        SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    )
    assert causal_delegate.label == "hf_causal"
    assert load_calls == ["hf_gptq", "hf_mlm", "hf_seq2seq", "hf_causal"]


def test_delegate_methods_forward_to_delegate_objects(monkeypatch) -> None:
    auto_mod = _load_auto_module(monkeypatch)

    eager_delegate = _delegate_class("eager")()
    adapter = auto_mod.HF_Auto_Adapter()
    adapter._delegate = eager_delegate

    assert adapter.describe("model-a") == {"label": "eager", "model": "model-a"}
    assert adapter.snapshot("model-a") == b"eager:model-a"
    adapter.restore("model-a", b"blob-a")
    assert eager_delegate.restores == [("model-a", b"blob-a")]

    lazy_delegate = _delegate_class("lazy")()
    lazy_adapter = auto_mod.HF_Auto_Adapter()
    monkeypatch.setattr(
        lazy_adapter, "_ensure_delegate_from_model", lambda model: lazy_delegate
    )
    monkeypatch.setattr(
        lazy_adapter, "_ensure_delegate_from_id", lambda model_id: lazy_delegate
    )

    assert lazy_adapter.describe("model-b") == {"label": "lazy", "model": "model-b"}
    assert lazy_adapter.snapshot("model-b") == b"lazy:model-b"
    lazy_adapter.restore("model-b", b"blob-b")
    assert lazy_delegate.restores == [("model-b", b"blob-b")]
    assert lazy_adapter.load_model("demo/model", device="cpu", revision="main") == {
        "label": "lazy",
        "model_id": "demo/model",
        "device": "cpu",
        "kwargs": {"revision": "main"},
    }
