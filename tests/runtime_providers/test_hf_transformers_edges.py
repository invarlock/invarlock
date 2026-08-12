from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    RuntimeExecutionSettings,
    evaluation_input_parts_sha256,
)
from invarlock.runtime_providers import hf_transformers as provider


def _settings(
    *, batch_size: int = 1, max_output_tokens: int = 4
) -> RuntimeExecutionSettings:
    return RuntimeExecutionSettings(
        seed=7,
        context_length=8,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        timeout_seconds=5,
    )


def _record(*, expected_output: str | None = "x") -> EvaluationRecord:
    input_text = "prompt"
    return EvaluationRecord(
        record_id="record-1",
        input_text=input_text,
        input_sha256=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        expected_output=expected_output,
    )


class _EncodedIds:
    def __init__(self, *, ndim: int = 2, shape: tuple[int, ...] = (1, 1)) -> None:
        self.ndim = ndim
        self.shape = shape

    def to(self, _device: object) -> _EncodedIds:
        return self


def _native_model() -> object:
    model_class = type(
        "NativeModel",
        (),
        {"__module__": "transformers.models.fixture.modeling_fixture"},
    )
    return model_class()


def test_native_conversion_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _native_model()
    monkeypatch.setattr(
        provider.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(RuntimeError, match="conversion metadata is unavailable"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    conversion = SimpleNamespace(get_model_conversion_mapping=None)
    loading = SimpleNamespace(
        rename_source_key=lambda *_args: ("weight", None),
        WeightRenaming=type("Renaming", (), {}),
        WeightConverter=type("Converter", (), {}),
    )
    monkeypatch.setattr(
        provider.importlib,
        "import_module",
        lambda name: conversion if name.endswith("conversion_mapping") else loading,
    )
    with pytest.raises(RuntimeError, match="conversion metadata is unavailable"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    class Renaming:
        pass

    class Converter:
        pass

    conversion = SimpleNamespace(get_model_conversion_mapping=lambda _model: ())
    loading = SimpleNamespace(
        rename_source_key=lambda *_args: ("weight", None),
        WeightRenaming=Renaming,
        WeightConverter=Converter,
    )
    monkeypatch.setattr(
        provider.importlib,
        "import_module",
        lambda name: conversion if name.endswith("conversion_mapping") else loading,
    )
    with pytest.raises(RuntimeError, match="conversion metadata is invalid"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    conversion.get_model_conversion_mapping = lambda _model: [Renaming()]
    loading.rename_source_key = lambda *_args: (_ for _ in ()).throw(ValueError())
    with pytest.raises(RuntimeError, match="key conversion failed"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    loading.rename_source_key = lambda *_args: ("", None)
    with pytest.raises(RuntimeError, match="conversion metadata is invalid"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )


def test_legacy_gpt2_mask_helpers_validate_config_prefix_and_tensor_shape() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gpt2",
            max_position_embeddings=8,
            num_hidden_layers=2,
        )
    )
    assert provider._is_legacy_gpt2_causal_mask_key(
        "transformer.h.1.attn.bias", model=model, prefix="transformer"
    )
    assert not provider._is_legacy_gpt2_causal_mask_key(
        "h.2.attn.bias", model=model, prefix=None
    )

    model.config.num_hidden_layers = True
    assert not provider._is_legacy_gpt2_causal_mask_key(
        "h.0.attn.bias", model=model, prefix=None
    )
    model.config.num_hidden_layers = 2
    model.config.max_position_embeddings = "8"
    assert not provider._is_authenticated_legacy_gpt2_causal_mask(
        "h.0.attn.bias", object(), model=model, prefix=None
    )
    model.config.max_position_embeddings = True
    assert not provider._is_authenticated_legacy_gpt2_causal_mask(
        "h.0.attn.bias", object(), model=model, prefix=None
    )
    model.config.max_position_embeddings = 8
    assert not provider._is_authenticated_legacy_gpt2_causal_mask(
        "h.0.attn.bias", SimpleNamespace(shape=(1,)), model=model, prefix=None
    )


def test_tensor_storage_identity_and_alias_helpers_fail_closed() -> None:
    assert provider._tensor_storage_identity(object()) is None

    class EmptyStorage:
        def data_ptr(self) -> int:
            return 0

    class EmptyTensor:
        def detach(self) -> EmptyTensor:
            return self

        def untyped_storage(self) -> EmptyStorage:
            return EmptyStorage()

    empty = EmptyTensor()
    assert provider._tensor_storage_identity(empty) is None
    assert provider._tensors_share_exact_storage(empty, empty)

    class BrokenTensor(EmptyTensor):
        def untyped_storage(self) -> EmptyStorage:
            raise RuntimeError("unavailable")

    assert provider._tensor_storage_identity(BrokenTensor()) is None


def test_strict_loader_requires_complete_loading_information(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="did not return loading information"):
        provider.load_hf_model_with_strict_loading_info(
            lambda *_args, **_kwargs: object(), tmp_path
        )

    with pytest.raises(RuntimeError, match="invalid loading information"):
        provider.load_hf_model_with_strict_loading_info(
            lambda *_args, **_kwargs: (SimpleNamespace(), {}), tmp_path
        )


def test_model_input_authentication_and_tokenizer_shape_fail_closed() -> None:
    record = _record()
    settings = _settings()
    mismatched = EvaluationRecord(
        record_id=record.record_id,
        input_text=record.input_text,
        input_sha256="0" * 64,
        expected_output=record.expected_output,
    )
    with pytest.raises(ValueError, match="does not match input_sha256"):
        provider._hf_model_inputs(
            record=mismatched,
            tokenizer_call=lambda *_args, **_kwargs: {},
            settings=settings,
            device="cpu",
        )

    with pytest.raises(RuntimeError, match="did not return input_ids"):
        provider._hf_model_inputs(
            record=record,
            tokenizer_call=lambda *_args, **_kwargs: [],
            settings=settings,
            device="cpu",
        )

    with pytest.raises(RuntimeError, match="returned invalid input_ids"):
        provider._hf_model_inputs(
            record=record,
            tokenizer_call=lambda *_args, **_kwargs: {
                "input_ids": _EncodedIds(ndim=1, shape=(1,))
            },
            settings=settings,
            device="cpu",
        )

    with pytest.raises(RuntimeError, match="returned empty input_ids"):
        provider._hf_model_inputs(
            record=record,
            tokenizer_call=lambda *_args, **_kwargs: {
                "input_ids": _EncodedIds(shape=(1, 0))
            },
            settings=settings,
            device="cpu",
        )


def test_causal_logits_reject_missing_model_output() -> None:
    torch = SimpleNamespace(ones_like=lambda window: window)
    with pytest.raises(RuntimeError, match="invalid causal logits"):
        provider._hf_causal_logits(
            lambda **_kwargs: SimpleNamespace(),
            torch,
            _EncodedIds(shape=(1, 1)),
        )


def test_normalized_nll_rejects_missing_and_malformed_targets() -> None:
    input_ids = _EncodedIds(shape=(1, 1))
    common = {
        "decode": lambda *_args, **_kwargs: "prompt",
        "model": object(),
        "torch": object(),
        "input_ids": input_ids,
        "device": "cpu",
        "settings": _settings(max_output_tokens=1),
        "deadline": math.inf,
    }
    with pytest.raises(ValueError, match="requires expected_output"):
        provider._hf_normalized_nll(
            record=_record(expected_output=None),
            tokenizer_call=lambda *_args, **_kwargs: {},
            **common,
        )

    with pytest.raises(RuntimeError, match="did not return target input_ids"):
        provider._hf_normalized_nll(
            record=_record(),
            tokenizer_call=lambda *_args, **_kwargs: [],
            **common,
        )

    with pytest.raises(RuntimeError, match="invalid target input_ids"):
        provider._hf_normalized_nll(
            record=_record(),
            tokenizer_call=lambda *_args, **_kwargs: {"input_ids": object()},
            **common,
        )

    with pytest.raises(ValueError, match="target exceeds max_output_tokens"):
        provider._hf_normalized_nll(
            record=_record(),
            tokenizer_call=lambda *_args, **_kwargs: {
                "input_ids": _EncodedIds(shape=(1, 2))
            },
            **common,
        )


def test_normalized_nll_rejects_timeout_and_out_of_vocabulary_token() -> None:
    torch = provider.importlib.import_module("torch")
    input_ids = torch.tensor([[1]])
    target_ids = torch.tensor([[2]])

    def decode(token_ids: list[int], **_kwargs: object) -> str:
        return "prompt" if len(token_ids) == 1 else "promptx"

    common = {
        "record": _record(),
        "tokenizer_call": lambda *_args, **_kwargs: {"input_ids": target_ids},
        "decode": decode,
        "torch": torch,
        "input_ids": input_ids,
        "device": "cpu",
        "settings": _settings(),
    }
    with pytest.raises(TimeoutError, match="scoring timed out"):
        provider._hf_normalized_nll(
            model=object(),
            deadline=-1.0,
            **common,
        )

    def model(**kwargs: object) -> SimpleNamespace:
        window = kwargs["input_ids"]
        return SimpleNamespace(logits=torch.zeros((1, window.shape[1], 2)))

    with pytest.raises(RuntimeError, match="outside the model vocabulary"):
        provider._hf_normalized_nll(
            model=model,
            deadline=math.inf,
            **common,
        )


def test_causal_scorer_rejects_non_unit_batch_before_runtime_import() -> None:
    record = _record()
    batch = EvaluationBatch(schedule_sha256="a" * 64, records=(record,))
    scorer = provider.HFTransformersCausalScorer(
        model=object(),
        tokenizer=object(),
        artifact_identity_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="requires batch_size=1"):
        scorer(batch, _settings(batch_size=2))


class _FakeQwenTextConfig:
    model_type = "qwen3_5_text"

    def __init__(self) -> None:
        self.dtype = "bfloat16"
        self.num_hidden_layers = 1
        self.layer_types = ["linear_attention"]


class _FakeQwenConfig:
    model_type = "qwen3_5"

    def __init__(self) -> None:
        self.dtype = "bfloat16"
        self.text_config = _FakeQwenTextConfig()
        self.quantization_config = None


class _FakeQwenNativeModel:
    def __init__(self) -> None:
        self.config = _FakeQwenConfig()
        self.is_quantized = False
        self.hf_quantizer = None
        self._get_dtype_plan = lambda _dtype: {}


class _FakeQwenCausalModel:
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="qwen3_5_text")
        self.named_parameters = lambda: ()
        self.named_buffers = lambda: ()
        self.named_modules = lambda: ()


def _install_fake_qwen_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(
        Qwen3_5ForCausalLM=_FakeQwenCausalModel,
        Qwen3_5ForConditionalGeneration=_FakeQwenNativeModel,
    )
    monkeypatch.setattr(provider.importlib, "import_module", lambda _name: module)


@pytest.mark.parametrize("native_api", ["missing", "malformed"])
def test_qwen_nonexecuting_profile_requires_introspectable_native_state(
    monkeypatch: pytest.MonkeyPatch,
    native_api: str,
) -> None:
    _install_fake_qwen_module(monkeypatch)
    model = _FakeQwenCausalModel()
    if native_api == "missing":
        model.named_parameters = None
    else:
        model.named_parameters = lambda: (("valid", object()), ("malformed",))

    with pytest.raises(RuntimeError, match="native model state is unavailable"):
        provider._qwen3_5_non_executing_checkpoint_keys(
            {
                "mtp.fc.weight",
                "mtp.layers.0.input_layernorm.weight",
                "mtp.layers.0.mlp.down_proj.weight",
                "mtp.layers.0.mlp.gate_proj.weight",
                "mtp.layers.0.mlp.up_proj.weight",
                "mtp.layers.0.post_attention_layernorm.weight",
                "mtp.layers.0.self_attn.k_norm.weight",
                "mtp.layers.0.self_attn.k_proj.weight",
                "mtp.layers.0.self_attn.o_proj.weight",
                "mtp.layers.0.self_attn.q_norm.weight",
                "mtp.layers.0.self_attn.q_proj.weight",
                "mtp.layers.0.self_attn.v_proj.weight",
                "mtp.norm.weight",
                "mtp.pre_fc_norm_embedding.weight",
                "mtp.pre_fc_norm_hidden.weight",
            },
            live_state={},
            model=model,
        )


def test_qwen_native_bfloat16_profile_rejects_unbound_runtime_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qwen_module(monkeypatch)
    model = _FakeQwenNativeModel()
    authenticated = _FakeQwenConfig()

    with pytest.raises(ValueError, match="model class is unsupported"):
        provider._qwen3_5_native_float32_to_bfloat16_keys(
            SimpleNamespace(config=model.config),
            authenticated_config=authenticated,
        )
    with pytest.raises(RuntimeError, match="configuration is unavailable"):
        provider._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=None,
        )
    with pytest.raises(ValueError, match="configuration is unsupported"):
        provider._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=SimpleNamespace(
                model_type="qwen3_5",
                text_config=_FakeQwenTextConfig(),
            ),
        )

    model.config.dtype = "float32"
    with pytest.raises(ValueError, match="materialization was not preserved"):
        provider._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=authenticated,
        )
    model.config.dtype = "bfloat16"
    model._get_dtype_plan = None
    with pytest.raises(RuntimeError, match="dtype plan is unavailable"):
        provider._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=authenticated,
        )
    model._get_dtype_plan = lambda _dtype: {}
    authenticated.text_config.layer_types = ["unsupported"]
    with pytest.raises(ValueError, match="conversion profile is invalid"):
        provider._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=authenticated,
        )


def test_live_tensor_binding_rejects_many_checkpoint_keys_for_one_live_name() -> None:
    with pytest.raises(ValueError, match="conversion is not one-to-one"):
        provider._bind_authenticated_live_tensors(
            {"first", "second"},
            live_state={"shared": object()},
            model=object(),
            prefix=None,
            authoritative_targets={"first": "shared", "second": "shared"},
        )


def test_hf_model_inputs_reject_structured_non_prompt_text() -> None:
    text = "prompt"
    part = EvaluationInputPart(
        kind="text",
        role="context",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    record = EvaluationRecord(
        record_id="structured",
        input_text=text,
        input_sha256=evaluation_input_parts_sha256((part,)),
        input_parts=(part,),
    )

    with pytest.raises(ValueError, match="one prompt text input part"):
        provider._hf_model_inputs(
            record=record,
            tokenizer_call=lambda *_args, **_kwargs: {},
            settings=_settings(),
            device="cpu",
        )


def test_hf_scoring_rejects_invalid_logprob_timeout_and_unknown_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = provider.importlib.import_module("torch")
    input_ids = torch.tensor([[1]])
    target_ids = torch.tensor([[2]])

    def decode(token_ids: list[int], **_kwargs: object) -> str:
        return "prompt" if len(token_ids) == 1 else "promptx"

    def model(**kwargs: object) -> SimpleNamespace:
        window = kwargs["input_ids"]
        return SimpleNamespace(logits=torch.zeros((1, window.shape[1], 3)))

    invalid_logprob = SimpleNamespace(
        gather=lambda *_args, **_kwargs: SimpleNamespace(item=lambda: 0.1)
    )
    monkeypatch.setattr(torch, "log_softmax", lambda *_args, **_kwargs: invalid_logprob)
    with pytest.raises(RuntimeError, match="invalid target logprob"):
        provider._hf_normalized_nll(
            record=_record(),
            tokenizer_call=lambda *_args, **_kwargs: {"input_ids": target_ids},
            decode=decode,
            model=model,
            torch=torch,
            input_ids=input_ids,
            device="cpu",
            settings=_settings(),
            deadline=math.inf,
        )

    with pytest.raises(TimeoutError, match="causal scoring timed out"):
        provider._hf_exact_match_output(
            tokenizer=SimpleNamespace(eos_token_id=None),
            decode=lambda *_args, **_kwargs: "",
            model=model,
            torch=torch,
            input_ids=input_ids,
            settings=_settings(),
            deadline=-1.0,
        )

    with pytest.raises(ValueError, match="unsupported built-in HF metric"):
        provider._hf_score_record(
            record=_record(),
            metric="future_metric",
            tokenizer=object(),
            tokenizer_call=lambda *_args, **_kwargs: {"input_ids": input_ids},
            decode=lambda *_args, **_kwargs: "",
            model=model,
            torch=torch,
            device="cpu",
            settings=_settings(),
        )


def test_hf_scorer_requires_callable_apis_and_execution_tensors() -> None:
    record = _record()
    batch = EvaluationBatch(schedule_sha256="a" * 64, records=(record,))
    invalid_api = provider.HFTransformersCausalScorer(
        model=object(),
        tokenizer=object(),
        artifact_identity_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="requires model and tokenizer APIs"):
        invalid_api(batch, _settings())

    class EmptyModel:
        def __call__(self, **_kwargs: object) -> object:
            return object()

        def modules(self) -> tuple[object, ...]:
            return (SimpleNamespace(training=False),)

        def parameters(self) -> tuple[object, ...]:
            return ()

        def buffers(self) -> tuple[object, ...]:
            return ()

    class Tokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> object:
            return object()

        def decode(self, *_args: object, **_kwargs: object) -> str:
            return ""

    no_tensors = provider.HFTransformersCausalScorer(
        model=EmptyModel(),
        tokenizer=Tokenizer(),
        artifact_identity_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="no execution tensors"):
        no_tensors(batch, _settings())
