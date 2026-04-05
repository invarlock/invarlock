from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from invarlock.runtime_security import runtime_allowances_scope


@pytest.mark.unit
def test_resolve_trust_remote_code_defaults_false(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)
    monkeypatch.delenv("INVARLOCK_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.delenv("ALLOW_REMOTE_CODE", raising=False)
    monkeypatch.delenv("TRUST_REMOTE_CODE_BOOL", raising=False)

    assert resolve_trust_remote_code({}) is False


@pytest.mark.unit
def test_resolve_trust_remote_code_ignores_env_opt_in(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    monkeypatch.setenv("INVARLOCK_TRUST_REMOTE_CODE", "1")
    assert resolve_trust_remote_code({}) is False


@pytest.mark.unit
def test_resolve_trust_remote_code_kwargs_override(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.setenv("INVARLOCK_TRUST_REMOTE_CODE", "0")
    with runtime_allowances_scope(allow_remote_code=True):
        assert resolve_trust_remote_code({"trust_remote_code": True}) is True


@pytest.mark.unit
def test_resolve_trust_remote_code_rejects_without_explicit_allow(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)
    with pytest.raises(RuntimeError, match="Remote model code is disabled by default"):
        resolve_trust_remote_code({"trust_remote_code": True})


@pytest.mark.unit
def test_default_torch_dtype_prefers_bf16_on_supported_cuda(monkeypatch):
    from invarlock.adapters.hf_loading import default_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert default_dtype() is torch.bfloat16


@pytest.mark.unit
def test_default_torch_dtype_falls_back_to_fp16_on_cuda(monkeypatch):
    from invarlock.adapters.hf_loading import default_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert default_dtype() is torch.float16


@pytest.mark.unit
def test_default_torch_dtype_uses_fp16_on_mps(monkeypatch):
    from invarlock.adapters.hf_loading import default_dtype

    if not hasattr(torch.backends, "mps"):
        pytest.skip("MPS backend not available")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert default_dtype() is torch.float16


@pytest.mark.unit
def test_default_torch_dtype_uses_fp32_on_cpu(monkeypatch):
    from invarlock.adapters.hf_loading import default_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert default_dtype() is torch.float32


@pytest.mark.unit
def test_resolve_torch_dtype_parses_strings(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_dtype({"dtype": "float16"}) is torch.float16
    assert resolve_dtype({"dtype": "bfloat16"}) is torch.bfloat16
    assert resolve_dtype({"dtype": "auto"}) == "auto"


def _write_local_config(path: Path, model_type: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
@pytest.mark.parametrize(
    ("task", "model_type", "loader_label"),
    [
        ("causal", "gpt2", "transformers.AutoModelForCausalLM"),
        ("mlm", "bert", "transformers.AutoModelForMaskedLM"),
        ("seq2seq", "t5", "transformers.AutoModelForSeq2SeqLM"),
    ],
)
def test_resolve_core_loader_strategy_defaults_to_auto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    model_type: str,
    loader_label: str,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / task, model_type)
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task=task,
        model_id=str(model_dir),
    )

    assert strategy.strategy == "auto"
    assert strategy.model_type == model_type
    assert strategy.loader_label == loader_label


@pytest.mark.unit
def test_resolve_core_loader_strategy_uses_direct_submodule_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / "causal", "gpt2")
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_dir),
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == "gpt2"
    assert (
        strategy.loader_label
        == "transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_type", "loader_label"),
    [
        (
            "gpt_oss",
            "transformers.models.gpt_oss.modeling_gpt_oss.GptOssForCausalLM",
        ),
        ("qwen3", "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM"),
        (
            "qwen3_moe",
            "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM",
        ),
        (
            "mistral3",
            "transformers.models.mistral3.modeling_mistral3.Mistral3ForConditionalGeneration",
        ),
        (
            "gemma3",
            "transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration",
        ),
        (
            "gemma4",
            "transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration",
        ),
        ("olmo2", "transformers.models.olmo2.modeling_olmo2.Olmo2ForCausalLM"),
    ],
)
def test_resolve_core_loader_strategy_supports_new_direct_submodule_families(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_type: str,
    loader_label: str,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / model_type, model_type)
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_dir),
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == model_type
    assert strategy.loader_label == loader_label


@pytest.mark.unit
def test_resolve_core_loader_strategy_supports_multimodal_gemma4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / "gemma4-mm", "gemma4")
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="multimodal",
        model_id=str(model_dir),
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == "gemma4"
    assert (
        strategy.loader_label
        == "transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration"
    )


@pytest.mark.unit
def test_resolve_core_loader_strategy_supports_multimodal_mistral3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / "mistral3-mm", "mistral3")
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="multimodal",
        model_id=str(model_dir),
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == "mistral3"
    assert (
        strategy.loader_label
        == "transformers.models.mistral3.modeling_mistral3.Mistral3ForConditionalGeneration"
    )


@pytest.mark.unit
def test_resolve_core_loader_strategy_infers_remote_model_type_from_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="mlm",
        model_id="prajjwal1/bert-tiny",
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == "bert"
    assert (
        strategy.loader_label
        == "transformers.models.bert.modeling_bert.BertForMaskedLM"
    )


@pytest.mark.unit
def test_resolve_core_loader_strategy_maps_deberta_v3_to_v2_loader_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="mlm",
        model_id="microsoft/deberta-v3-base",
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == "deberta-v2"
    assert (
        strategy.loader_label
        == "transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForMaskedLM"
    )


@pytest.mark.unit
def test_resolve_core_loader_strategy_trust_remote_code_forces_auto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / "causal_remote", "gpt2")
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )
    with runtime_allowances_scope(allow_remote_code=True):
        strategy = hf_loading.resolve_core_loader_strategy(
            task="causal",
            model_id=str(model_dir),
            kwargs={"trust_remote_code": True},
            allow_direct_submodule=True,
        )

    assert strategy.strategy == "auto"
    assert strategy.loader_label == "transformers.AutoModelForCausalLM"


@pytest.mark.unit
def test_resolve_core_loader_strategy_unknown_model_type_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / "unknown", "unknown-arch")
    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_dir),
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "auto"
    assert strategy.model_type == "unknown-arch"


@pytest.mark.unit
def test_resolve_core_loader_strategy_direct_import_failure_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    model_dir = _write_local_config(tmp_path / "direct_fallback", "gpt2")

    def _fake_import(module_path: str, symbol_name: str) -> str:
        if module_path == "transformers.models.gpt2.modeling_gpt2":
            raise ModuleNotFoundError("no direct class on this install")
        return f"{module_path}.{symbol_name}"

    monkeypatch.setattr(hf_loading, "_import_symbol", _fake_import)

    strategy = hf_loading.resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_dir),
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "auto"
    assert strategy.loader_label == "transformers.AutoModelForCausalLM"
