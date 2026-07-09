from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "checks"
        / "check_local_hf_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("check_local_hf_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_local_hf_runtime_accepts_required_import_surface() -> None:
    module = _load_module()

    def fake_import_module(name: str) -> ModuleType:
        attrs = {
            "tokenizers": {"Tokenizer": object},
            "tokenizers.models": {"WordLevel": object},
            "tokenizers.pre_tokenizers": {"Whitespace": object},
            "transformers": {
                "GPT2Config": object,
                "GPT2LMHeadModel": object,
                "PreTrainedTokenizerFast": object,
            },
        }[name]
        return SimpleNamespace(**attrs)

    assert module.check_local_hf_runtime(fake_import_module) == []


def test_check_local_hf_runtime_reports_lazy_transformers_failure(capsys) -> None:
    module = _load_module()

    class BrokenTransformers:
        GPT2Config = object
        PreTrainedTokenizerFast = object

        def __getattr__(self, name: str) -> object:
            if name == "GPT2LMHeadModel":
                raise RuntimeError("operator torchvision::nms does not exist")
            raise AttributeError(name)

    def fake_import_module(name: str) -> ModuleType:
        if name == "transformers":
            return BrokenTransformers()
        attrs = {
            "tokenizers": {"Tokenizer": object},
            "tokenizers.models": {"WordLevel": object},
            "tokenizers.pre_tokenizers": {"Whitespace": object},
        }[name]
        return SimpleNamespace(**attrs)

    failures = module.check_local_hf_runtime(fake_import_module)

    assert len(failures) == 1
    assert failures[0].module == "transformers"
    assert failures[0].attr == "GPT2LMHeadModel"
    assert "torchvision::nms" in failures[0].error

    original_check = module.check_local_hf_runtime
    module.check_local_hf_runtime = lambda: failures
    try:
        assert module.main() == 1
    finally:
        module.check_local_hf_runtime = original_check

    captured = capsys.readouterr()
    assert "make local-hf-env-refresh" in captured.err
    assert "make local-hf-pipeline-smoke-locked" in captured.err
