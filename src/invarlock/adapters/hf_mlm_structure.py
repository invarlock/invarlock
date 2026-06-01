"""
Structure helpers for HuggingFace masked LM adapters.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from invarlock.core.exceptions import AdapterError

_MLM_FALLBACK_TOKENS = (
    "maskedlm",
    "masked language",
    "for automodel",
    "unrecognized model",
    "model_type",
    "unrecognized configuration class",
    "is not supported for this model",
)
_HF_MLM_PROBE_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    RuntimeError,
    StopIteration,
    TypeError,
    ValueError,
)
_HF_MLM_CLASS_NAMES = {
    "BertModel",
    "BertForSequenceClassification",
    "BertForMaskedLM",
    "RobertaModel",
    "RobertaForSequenceClassification",
    "RobertaForMaskedLM",
    "DistilBertModel",
    "DistilBertForSequenceClassification",
    "DistilBertForMaskedLM",
    "DebertaV2Model",
    "DebertaV2ForSequenceClassification",
    "DebertaV2ForMaskedLM",
    "AlbertModel",
    "AlbertForSequenceClassification",
    "ElectraModel",
    "ElectraForSequenceClassification",
}
_HF_MLM_MODEL_TYPES = {
    "bert",
    "roberta",
    "distilbert",
    "deberta",
    "deberta-v2",
    "albert",
    "electra",
}


def _should_retry_mlm_loader(exc: BaseException) -> bool:
    cause = exc.__cause__ or exc
    if not isinstance(cause, (AttributeError, ImportError, OSError, ValueError)):
        return False
    message = str(cause).strip().lower()
    return any(token in message for token in _MLM_FALLBACK_TOKENS)


def _has_set_attr(obj: Any, name: str) -> bool:
    return _module_has(obj, name)


def _module_has(obj: Any, name: str) -> bool:
    if isinstance(obj, nn.Module):
        in_modules = hasattr(obj, "_modules") and name in obj._modules
        in_params = hasattr(obj, "_parameters") and name in obj._parameters
        in_buffers = hasattr(obj, "_buffers") and name in obj._buffers
        in_dict = name in getattr(obj, "__dict__", {})
        return in_modules or in_params or in_buffers or in_dict
    return name in getattr(obj, "__dict__", {})


def _first_sequence_item(values: Any) -> Any | None:
    if values is None:
        return None
    try:
        length = len(values)
        if isinstance(length, int) and length > 0:
            return values[0]
        if isinstance(length, int) and length <= 0:
            return None
    except _HF_MLM_PROBE_ERRORS:
        pass
    try:
        return next(iter(values))
    except _HF_MLM_PROBE_ERRORS:
        return None


def _has_non_empty_layers(layers: Any) -> bool:
    return _first_sequence_item(layers) is not None


def _resolve_wrapper_encoder(model: Any) -> Any | None:
    if _module_has(model, "bert") and _module_has(model.bert, "encoder"):
        return model.bert.encoder
    if _module_has(model, "roberta") and _module_has(model.roberta, "encoder"):
        return model.roberta.encoder
    if _module_has(model, "deberta") and _module_has(model.deberta, "encoder"):
        return model.deberta.encoder
    if _module_has(model, "distilbert") and _module_has(
        model.distilbert, "transformer"
    ):
        return model.distilbert.transformer
    return None


def _resolve_mlm_encoder(model: Any) -> tuple[Any | None, bool]:
    direct_encoder = getattr(model, "encoder", None)
    if _module_has(model, "encoder") and _module_has(direct_encoder, "layer"):
        return direct_encoder, False
    wrapper_encoder = _resolve_wrapper_encoder(model)
    if wrapper_encoder is not None:
        return wrapper_encoder, True
    return None, False


def _resolve_embeddings_module(model: Any) -> Any | None:
    if _module_has(model, "embeddings"):
        return model.embeddings
    if _module_has(model, "bert") and _module_has(model.bert, "embeddings"):
        return model.bert.embeddings
    if _module_has(model, "roberta") and _module_has(model.roberta, "embeddings"):
        return model.roberta.embeddings
    if _module_has(model, "deberta") and _module_has(model.deberta, "embeddings"):
        return model.deberta.embeddings
    if _module_has(model, "distilbert") and _module_has(model.distilbert, "embeddings"):
        return model.distilbert.embeddings
    return None


def _resolve_layer_by_index(layers: Any, layer_idx: int, encoder: Any) -> Any:
    try:
        return layers[layer_idx]
    except _HF_MLM_PROBE_ERRORS:
        pass

    try:
        for index, layer_candidate in enumerate(iter(layers)):
            if index == layer_idx:
                return layer_candidate
    except _HF_MLM_PROBE_ERRORS:
        pass

    if isinstance(encoder, nn.Module):
        try:
            for index, child in enumerate(encoder.children()):
                if index == layer_idx:
                    return child
        except _HF_MLM_PROBE_ERRORS:
            pass

    raise AdapterError(
        code="E202",
        message="ADAPTER-STRUCTURE-INVALID: could not access encoder layer",
        details={"layer_idx": int(layer_idx)},
    )


def _has_complete_attention_structure(layer: Any) -> bool:
    if not (
        hasattr(layer, "attention")
        and hasattr(layer, "intermediate")
        and hasattr(layer, "output")
        and hasattr(layer.attention, "self")
    ):
        return False
    return all(
        _has_set_attr(layer.attention.self, name) for name in ("query", "key", "value")
    )


def _has_distilbert_attention_structure(layer: Any) -> bool:
    if not (
        hasattr(layer, "attention")
        and hasattr(layer, "ffn")
        and _has_set_attr(layer, "sa_layer_norm")
        and _has_set_attr(layer, "output_layer_norm")
    ):
        return False
    return all(
        _has_set_attr(layer.attention, name)
        for name in ("q_lin", "k_lin", "v_lin", "out_lin")
    ) and all(_has_set_attr(layer.ffn, name) for name in ("lin1", "lin2"))


def _has_deberta_attention_structure(layer: Any) -> bool:
    if not (
        hasattr(layer, "attention")
        and hasattr(layer, "intermediate")
        and hasattr(layer, "output")
        and hasattr(layer.attention, "self")
    ):
        return False
    has_qkv = all(
        _has_set_attr(layer.attention.self, name)
        for name in ("query_proj", "key_proj", "value_proj")
    )
    has_outputs = _has_set_attr(layer.attention, "output") and _has_set_attr(
        layer.attention.output, "dense"
    )
    return bool(has_qkv and has_outputs and _has_set_attr(layer.intermediate, "dense"))


def _resolve_mlm_layer_variant(layer: Any) -> str | None:
    if _has_complete_attention_structure(layer):
        return "bert"
    if _has_distilbert_attention_structure(layer):
        return "distilbert"
    if _has_deberta_attention_structure(layer):
        return "deberta-v2"
    return None


def _prediction_head_tied_to_embeddings(
    model: Any, bert_model: Any, config: Any
) -> bool:
    decoder = getattr(
        getattr(getattr(model, "cls", None), "predictions", None), "decoder", None
    )
    decoder_weight = getattr(decoder, "weight", None)
    embeddings = getattr(
        getattr(bert_model, "embeddings", None), "word_embeddings", None
    )
    embedding_weight = getattr(embeddings, "weight", None)
    if decoder_weight is None or embedding_weight is None:
        return False
    if decoder_weight is embedding_weight:
        return True
    if getattr(config, "model_type", None) != "roberta":
        return False
    try:
        return decoder_weight.shape == embedding_weight.shape
    except _HF_MLM_PROBE_ERRORS:
        return False


def _resolve_encoder(
    model: Any,
    *,
    prefer_wrapper: bool,
) -> tuple[Any | None, bool]:
    wrapper_encoder = _resolve_wrapper_encoder(model)
    direct_encoder = (
        model.encoder
        if _module_has(model, "encoder") and _module_has(model.encoder, "layer")
        else None
    )
    if prefer_wrapper:
        if wrapper_encoder is not None and _module_has(wrapper_encoder, "layer"):
            return wrapper_encoder, True
        if direct_encoder is not None:
            return direct_encoder, False
    else:
        if direct_encoder is not None and wrapper_encoder is None:
            return direct_encoder, False
        if wrapper_encoder is not None and _module_has(wrapper_encoder, "layer"):
            return wrapper_encoder, True
        if direct_encoder is not None:
            return direct_encoder, False
    return None, False


def _require_encoder_layers(
    model: Any,
    *,
    prefer_wrapper: bool,
    message: str,
) -> tuple[Any, Any, bool]:
    encoder, from_wrapper = _resolve_encoder(model, prefer_wrapper=prefer_wrapper)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if encoder is None or layers is None:
        raise AdapterError(
            code="E202",
            message=message,
            details={"model_class": model.__class__.__name__},
        )
    return encoder, layers, from_wrapper


def _supports_hf_mlm_class(model: Any) -> bool:
    model_name = model.__class__.__name__
    if model_name not in _HF_MLM_CLASS_NAMES:
        return False
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return model_name in {
            "BertModel",
            "BertForSequenceClassification",
            "RobertaModel",
            "RobertaForSequenceClassification",
            "DistilBertModel",
            "DistilBertForSequenceClassification",
        }
    return str(model_type).lower() in _HF_MLM_MODEL_TYPES


def _resolve_parameter_device(model: Any) -> Any:
    try:
        return next(iter(model.parameters())).device
    except (AttributeError, RuntimeError, StopIteration, TypeError, ValueError):
        return torch.device("cpu")


def _count_model_parameters(model: Any) -> int:
    try:
        return int(sum(int(p.numel()) for p in model.parameters()))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def _extract_prediction_head_tying(model: Any, config: Any) -> dict[str, str]:
    if _module_has(model, "roberta"):
        bert_model = model.roberta
        base_name = "roberta"
    elif _module_has(model, "deberta"):
        bert_model = model.deberta
        base_name = "deberta"
    elif _module_has(model, "bert"):
        bert_model = model.bert
        base_name = "bert"
    elif _module_has(model, "distilbert"):
        projector = getattr(model, "vocab_projector", None)
        projector_weight = getattr(projector, "weight", None)
        embeddings = getattr(
            getattr(model.distilbert, "embeddings", None), "word_embeddings", None
        )
        embedding_weight = getattr(embeddings, "weight", None)
        if projector_weight is not None and projector_weight is embedding_weight:
            return {
                "vocab_projector.weight": (
                    "distilbert.embeddings.word_embeddings.weight"
                )
            }
        return {}
    else:
        return {}
    if not _module_has(bert_model, "embeddings"):
        return {}
    if not _prediction_head_tied_to_embeddings(model, bert_model, config):
        return {}
    return {
        "cls.predictions.decoder.weight": (
            f"{base_name}.embeddings.word_embeddings.weight"
        )
    }


def _resolve_supported_model_type(config: Any) -> str:
    model_type = str(getattr(config, "model_type", "bert") or "bert").lower()
    return model_type if model_type in _HF_MLM_MODEL_TYPES else "bert"
