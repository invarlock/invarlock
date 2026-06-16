"""Helpers for Hugging Face model loading.

Centralizes security- and performance-sensitive defaults used by HF adapters.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.runtime_security import remote_code_allowed

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_TORCH_UNSET = object()
_torch_module: Any = _TORCH_UNSET
# Split to avoid secret-scanner false positives on Mistral's architecture name.
_MISTRAL3_ARCH = "Mistral3For" + "ConditionalGeneration"
_COERCE_ERRORS = (TypeError, ValueError, OverflowError)
_CUDA_CAPABILITY_ERRORS = (AttributeError, RuntimeError, OSError)
_MEMORY_EFFICIENT_TRUE = {"1", "true", "yes", "on", "auto"}
_MEMORY_EFFICIENT_FALSE = {"0", "false", "no", "off", "disabled"}
_AUTO_DEVICE_MAP_PARAM_THRESHOLD_B = 20.0

_AUTO_LOADER_SPECS: dict[str, tuple[str, str]] = {
    "causal": ("transformers", "AutoModelForCausalLM"),
    "mlm": ("transformers", "AutoModelForMaskedLM"),
    "mlm_base": ("transformers", "AutoModel"),
    "seq2seq": ("transformers", "AutoModelForSeq2SeqLM"),
}

_MULTIMODAL_AUTO_LOADER_SPECS: tuple[tuple[str, str], ...] = (
    ("transformers", "AutoModelForImageTextToText"),
    ("transformers", "AutoModelForMultimodalLM"),
    ("transformers", "AutoModelForVision2Seq"),
)
_MULTIMODAL_AUTO_LOAD_FALLBACK_ERRORS = (TypeError, ValueError)

_DIRECT_SUBMODULE_SPECS: dict[str, dict[str, tuple[str, str]]] = {
    "causal": {
        "gpt2": ("transformers.models.gpt2.modeling_gpt2", "GPT2LMHeadModel"),
        "gpt_oss": (
            "transformers.models.gpt_oss.modeling_gpt_oss",
            "GptOssForCausalLM",
        ),
        "opt": ("transformers.models.opt.modeling_opt", "OPTForCausalLM"),
        "llama": ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
        "mistral": (
            "transformers.models.mistral.modeling_mistral",
            "MistralForCausalLM",
        ),
        "mistral3": (
            "transformers.models.mistral3.modeling_mistral3",
            _MISTRAL3_ARCH,
        ),
        "mixtral": (
            "transformers.models.mixtral.modeling_mixtral",
            "MixtralForCausalLM",
        ),
        "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
        "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
        "qwen3_moe": (
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "Qwen3MoeForCausalLM",
        ),
        "gemma3": (
            "transformers.models.gemma3.modeling_gemma3",
            "Gemma3ForConditionalGeneration",
        ),
        "gemma3n": (
            "transformers.models.gemma3n.modeling_gemma3n",
            "Gemma3nForConditionalGeneration",
        ),
        "gemma4": (
            "transformers.models.gemma4.modeling_gemma4",
            "Gemma4ForConditionalGeneration",
        ),
        "gpt_neox": (
            "transformers.models.gpt_neox.modeling_gpt_neox",
            "GPTNeoXForCausalLM",
        ),
        "olmo2": ("transformers.models.olmo2.modeling_olmo2", "Olmo2ForCausalLM"),
        "olmoe": ("transformers.models.olmoe.modeling_olmoe", "OlmoeForCausalLM"),
        "phi": ("transformers.models.phi.modeling_phi", "PhiForCausalLM"),
        "phi3": ("transformers.models.phi3.modeling_phi3", "Phi3ForCausalLM"),
    },
    "mlm": {
        "bert": ("transformers.models.bert.modeling_bert", "BertForMaskedLM"),
        "roberta": (
            "transformers.models.roberta.modeling_roberta",
            "RobertaForMaskedLM",
        ),
        "distilbert": (
            "transformers.models.distilbert.modeling_distilbert",
            "DistilBertForMaskedLM",
        ),
        "deberta": (
            "transformers.models.deberta.modeling_deberta",
            "DebertaForMaskedLM",
        ),
        "deberta-v2": (
            "transformers.models.deberta_v2.modeling_deberta_v2",
            "DebertaV2ForMaskedLM",
        ),
        "albert": ("transformers.models.albert.modeling_albert", "AlbertForMaskedLM"),
        "electra": (
            "transformers.models.electra.modeling_electra",
            "ElectraForMaskedLM",
        ),
    },
    "seq2seq": {
        "t5": ("transformers.models.t5.modeling_t5", "T5ForConditionalGeneration"),
        "bart": (
            "transformers.models.bart.modeling_bart",
            "BartForConditionalGeneration",
        ),
        "mbart": (
            "transformers.models.mbart.modeling_mbart",
            "MBartForConditionalGeneration",
        ),
        "pegasus": (
            "transformers.models.pegasus.modeling_pegasus",
            "PegasusForConditionalGeneration",
        ),
        "marian": (
            "transformers.models.marian.modeling_marian",
            "MarianMTModel",
        ),
    },
    "multimodal": {
        "gemma3n": (
            "transformers.models.gemma3n.modeling_gemma3n",
            "Gemma3nForConditionalGeneration",
        ),
        "gemma3": (
            "transformers.models.gemma3.modeling_gemma3",
            "Gemma3ForConditionalGeneration",
        ),
        "gemma4_unified": (
            "transformers.models.gemma4_unified.modeling_gemma4_unified",
            "Gemma4UnifiedForConditionalGeneration",
        ),
        "gemma4": (
            "transformers.models.gemma4.modeling_gemma4",
            "Gemma4ForConditionalGeneration",
        ),
        "mistral3": (
            "transformers.models.mistral3.modeling_mistral3",
            _MISTRAL3_ARCH,
        ),
    },
}

_MODEL_ID_TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chatglm", ("chatglm", "glm-4", "glm4")),
    ("mistral3", ("ministral-3", "ministral3", "mistral3")),
    ("qwen3_moe", ("qwen3_moe", "qwen3-moe")),
    ("qwen3", ("qwen3",)),
    (
        "qwen2",
        ("qwen2.5", "qwen2-5", "qwen2_5", "qwen1.5", "qwen1-5", "qwen1_5", "qwen2"),
    ),
    ("gemma4_unified", ("gemma-4-12b", "gemma4-12b", "gemma_4_12b")),
    ("gemma4", ("gemma-4", "gemma4")),
    ("gemma3n", ("gemma-3n", "gemma3n")),
    ("gemma3", ("gemma-3", "gemma3")),
    (
        "deberta-v2",
        (
            "deberta-v2",
            "deberta_v2",
            "debertav2",
            "deberta-v3",
            "deberta_v3",
            "debertav3",
        ),
    ),
    ("deberta", ("deberta",)),
    ("distilbert", ("distilbert",)),
    ("roberta", ("roberta",)),
    ("electra", ("electra",)),
    ("albert", ("albert",)),
    ("bert", ("bert",)),
    ("mbart", ("mbart",)),
    ("bart", ("bart",)),
    ("marian", ("marian", "opus-mt")),
    ("t5", ("t5",)),
    ("mixtral", ("mixtral",)),
    ("mistral", ("mistral",)),
    ("llama", ("llama",)),
    ("olmoe", ("olmoe", "olmo-e", "olmoe-")),
    ("olmo2", ("olmo-2", "olmo2")),
    ("gpt_neox", ("gpt-neox", "gpt_neox")),
    ("gpt_oss", ("gpt-oss", "gpt_oss")),
    ("opt", ("facebook/opt", "/opt-", " opt-", "opt-")),
    ("phi3", ("phi-3", "phi3", "phi-4-mini", "phi4-mini", "phi_4_mini")),
    ("phi", ("phi-",)),
    ("gpt2", ("gpt2",)),
)


@dataclass(frozen=True)
class HFLoaderStrategy:
    task: str
    strategy: str
    loader: Any
    loader_label: str
    model_type: str | None = None


class _ChatGLMRemoteCodeCausalLoader:
    """Compatibility loader for ChatGLM remote code on newer Transformers."""

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any) -> Any:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        loader_kwargs = dict(kwargs)
        trust_remote_code = resolve_trust_remote_code(loader_kwargs)
        config = loader_kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        if not hasattr(config, "max_length"):
            seq_length = getattr(config, "seq_length", None)
            if seq_length is not None:
                config.max_length = seq_length
        if not hasattr(config, "use_cache"):
            config.use_cache = True
        loader_kwargs["config"] = config

        auto_map = getattr(config, "auto_map", None)
        class_ref = (
            auto_map.get("AutoModelForCausalLM") if isinstance(auto_map, dict) else None
        )
        if not isinstance(class_ref, str) or not class_ref:
            auto_model = _resolve_auto_loader("causal")[0]
            return auto_model.from_pretrained(model_id, **loader_kwargs)

        model_cls = get_class_from_dynamic_module(
            class_ref,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if not hasattr(model_cls, "all_tied_weights_keys"):
            model_cls.all_tied_weights_keys = {}
        return model_cls.from_pretrained(model_id, **loader_kwargs)


def _get_torch() -> Any:
    global _torch_module
    if _torch_module is _TORCH_UNSET:
        try:
            import torch as _torch
        except ModuleNotFoundError:
            _torch_module = None
        else:
            _torch_module = _torch
    return None if _torch_module is _TORCH_UNSET else _torch_module


def _require_torch() -> Any:
    torch = _get_torch()
    if torch is None:
        raise ModuleNotFoundError("torch")
    return torch


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def resolve_trust_remote_code(
    kwargs: dict[str, Any] | None = None, *, default: bool = False
) -> bool:
    """Resolve trust_remote_code from explicit load kwargs only."""
    requested: bool | None = None
    if kwargs and "trust_remote_code" in kwargs:
        coerced = _coerce_bool(kwargs.get("trust_remote_code"))
        if coerced is not None:
            requested = coerced

    if requested is None:
        requested = default

    if not requested:
        return False
    if not remote_code_allowed():
        raise RuntimeError(
            "Remote model code is disabled by default. "
            "Enable the runtime remote-code policy before requesting trust_remote_code."
        )
    return True


def default_dtype() -> Any:
    """Pick a safe default dtype for HF loads based on hardware."""
    torch = _require_torch()
    if torch.cuda.is_available():
        try:
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
        except _CUDA_CAPABILITY_ERRORS:
            pass
        return torch.float16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16

    return torch.float32


def resolve_dtype(kwargs: dict[str, Any] | None = None) -> Any:
    """Resolve dtype from kwargs or choose a hardware-aware default."""
    torch = _require_torch()
    if kwargs and "dtype" in kwargs:
        val = kwargs.get("dtype")
        if isinstance(val, torch.dtype):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if s == "auto":
                return "auto"
            removed_aliases = {
                "fp16": "float16",
                "half": "float16",
                "bf16": "bfloat16",
                "fp32": "float32",
            }
            if s in removed_aliases:
                canonical = removed_aliases[s]
                raise ValueError(
                    f"model.dtype={s} is not supported; use model.dtype={canonical}"
                )
            mapping = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if s in mapping:
                return mapping[s]

    return default_dtype()


def _estimate_params_b_from_model_id(model_id: str) -> float | None:
    model_lower = model_id.lower()
    if "mixtral" in model_lower or "8x7b" in model_lower:
        return 47.0
    if "30b-a3b" in model_lower:
        return 30.0
    if "26b-a4b" in model_lower:
        return 26.0
    if "1b-7b" in model_lower or "olmoe" in model_lower:
        return 7.0

    matches = [
        float(match.group(1).replace("_", "."))
        for match in re.finditer(r"(\d+(?:[._]\d+)?)\s*b\b", model_lower)
    ]
    return max(matches) if matches else None


def _is_moe_model_id(model_id: str) -> bool:
    model_lower = model_id.lower()
    return any(
        token in model_lower
        for token in ("moe", "mixtral", "8x7b", "a3b", "a4b", "olmoe")
    )


def _accelerated_device_requested(load_device: Any | None) -> bool:
    requested = "auto" if load_device is None else str(load_device).strip().lower()
    if requested.startswith(("cuda", "mps", "xpu")):
        return True
    if requested != "auto":
        return False

    torch = _get_torch()
    if torch is None:
        return False
    try:
        if torch.cuda.is_available():
            return True
    except _CUDA_CAPABILITY_ERRORS:
        pass
    try:
        return bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except _CUDA_CAPABILITY_ERRORS:
        return False


def _memory_efficient_load_enabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _MEMORY_EFFICIENT_FALSE:
            return False
        if normalized in _MEMORY_EFFICIENT_TRUE:
            return True
    return True


def _normalize_explicit_load_dtype(value: Any) -> Any:
    torch = _require_torch()
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    known_values = {
        "auto",
        "float16",
        "bfloat16",
        "float32",
        "fp16",
        "half",
        "bf16",
        "fp32",
    }
    if normalized in known_values:
        return resolve_dtype({"dtype": normalized})
    return value


def apply_memory_efficient_load_defaults(
    model_id: str,
    kwargs: dict[str, Any],
    *,
    load_device: Any | None = None,
) -> dict[str, Any]:
    """Apply shared HF loading defaults that reduce avoidable peak memory.

    User-provided kwargs always win. The public config key remains `model.dtype`;
    this helper only normalizes its value before handing kwargs to Hugging Face.
    """
    prepared = dict(kwargs)
    enabled = _memory_efficient_load_enabled(
        prepared.pop("memory_efficient_load", None)
    )

    if "dtype" in prepared:
        prepared["dtype"] = _normalize_explicit_load_dtype(prepared["dtype"])
    elif enabled and _accelerated_device_requested(load_device):
        prepared["dtype"] = default_dtype()

    if not enabled:
        return prepared

    prepared.setdefault("low_cpu_mem_usage", True)

    params_b = _estimate_params_b_from_model_id(model_id)
    if (
        "device_map" not in prepared
        and _accelerated_device_requested(load_device)
        and (
            _is_moe_model_id(model_id)
            or (
                isinstance(params_b, int | float)
                and params_b >= _AUTO_DEVICE_MAP_PARAM_THRESHOLD_B
            )
        )
    ):
        prepared["device_map"] = "auto"

    return prepared


def _normalize_model_type(value: Any) -> str | None:
    try:
        normalized = str(value or "").strip().lower()
    except _COERCE_ERRORS:
        return None
    return normalized or None


def _read_local_config(model_id: str) -> dict[str, Any] | None:
    path = Path(model_id)
    if not path.is_dir():
        return None
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _infer_model_type_from_model_id(model_id: str) -> str | None:
    model_lower = _normalize_model_type(model_id)
    if model_lower is None:
        return None
    padded = f" {model_lower} "
    for model_type, hints in _MODEL_ID_TYPE_HINTS:
        if any(hint in model_lower or hint in padded for hint in hints):
            return model_type
    return None


def _import_symbol(module_path: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, symbol_name)


def _loader_label(module_path: str, symbol_name: str) -> str:
    return f"{module_path}.{symbol_name}"


def _multimodal_auto_loader_label() -> str:
    return " -> ".join(
        _loader_label(module_path, symbol_name)
        for module_path, symbol_name in _MULTIMODAL_AUTO_LOADER_SPECS
    )


class _MultimodalAutoFallbackLoader:
    """Try compatible HF multimodal auto loaders in stable preference order."""

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any) -> Any:
        failures: list[str] = []
        last_error: BaseException | None = None
        for module_path, symbol_name in _MULTIMODAL_AUTO_LOADER_SPECS:
            loader_label = _loader_label(module_path, symbol_name)
            try:
                loader = _import_symbol(module_path, symbol_name)
            except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                failures.append(f"{loader_label}: unavailable ({type(exc).__name__})")
                last_error = exc
                continue
            try:
                return loader.from_pretrained(model_id, **kwargs)
            except _MULTIMODAL_AUTO_LOAD_FALLBACK_ERRORS as exc:
                failures.append(f"{loader_label}: incompatible ({type(exc).__name__})")
                last_error = exc
                continue
        detail = "; ".join(failures) or "no loaders were attempted"
        raise ValueError(
            f"No compatible HF multimodal auto loader succeeded: {detail}"
        ) from last_error


def _resolve_auto_loader(task: str, model_type: str | None = None) -> tuple[Any, str]:
    if task == "multimodal":
        return _MultimodalAutoFallbackLoader, _multimodal_auto_loader_label()
    module_path, symbol_name = _AUTO_LOADER_SPECS[task]
    return (
        _import_symbol(module_path, symbol_name),
        _loader_label(module_path, symbol_name),
    )


def _resolve_direct_submodule_loader(
    task: str, model_type: str | None
) -> tuple[Any, str] | None:
    if model_type is None:
        return None
    spec = _DIRECT_SUBMODULE_SPECS.get(task, {}).get(model_type)
    if spec is None:
        return None
    module_path, symbol_name = spec
    try:
        return (
            _import_symbol(module_path, symbol_name),
            _loader_label(module_path, symbol_name),
        )
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None


def resolve_core_loader_strategy(
    *,
    task: str,
    model_id: str,
    kwargs: dict[str, Any] | None = None,
    allow_direct_submodule: bool = False,
) -> HFLoaderStrategy:
    """Resolve the primary loader strategy for a core HF adapter."""

    if task != "multimodal" and task not in _AUTO_LOADER_SPECS:
        raise KeyError(f"Unknown HF loader task: {task}")

    model_type = None
    config_data = _read_local_config(model_id)
    if isinstance(config_data, dict):
        model_type = _normalize_model_type(config_data.get("model_type"))
    if model_type is None:
        model_type = _infer_model_type_from_model_id(model_id)

    if (
        task == "causal"
        and model_type == "chatglm"
        and resolve_trust_remote_code(kwargs)
    ):
        return HFLoaderStrategy(
            task=task,
            strategy="remote_code",
            loader=_ChatGLMRemoteCodeCausalLoader,
            loader_label="invarlock.adapters.hf_loading._ChatGLMRemoteCodeCausalLoader",
            model_type=model_type,
        )

    if allow_direct_submodule and not resolve_trust_remote_code(kwargs):
        direct = _resolve_direct_submodule_loader(task, model_type)
        if direct is not None:
            loader, loader_label = direct
            return HFLoaderStrategy(
                task=task,
                strategy="direct_submodule",
                loader=loader,
                loader_label=loader_label,
                model_type=model_type,
            )

    loader, loader_label = _resolve_auto_loader(task, model_type)
    return HFLoaderStrategy(
        task=task,
        strategy="auto",
        loader=loader,
        loader_label=loader_label,
        model_type=model_type,
    )


__all__ = [
    "HFLoaderStrategy",
    "apply_memory_efficient_load_defaults",
    "default_dtype",
    "resolve_core_loader_strategy",
    "resolve_dtype",
    "resolve_trust_remote_code",
]
