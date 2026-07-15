"""
Tests for Model Capabilities
============================

TDD tests for the capabilities module including:
- QuantizationConfig creation and detection
- ModelCapabilities factory methods
- Detection from model config and instances
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from invarlock.adapters.capabilities import (
    ModelCapabilities,
    QuantizationConfig,
    QuantizationMethod,
    detect_capabilities_from_model,
    detect_quantization_from_config,
)


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""

    def test_default_is_not_quantized(self):
        """Default config should be FP16/not quantized."""
        cfg = QuantizationConfig()
        assert cfg.method == QuantizationMethod.NONE
        assert cfg.bits == 16
        assert cfg.is_quantized() is False
        assert cfg.is_bnb() is False

    def test_bnb_8bit_config(self):
        """BNB 8-bit config should be correctly identified."""
        cfg = QuantizationConfig(
            method=QuantizationMethod.BNB_8BIT,
            bits=8,
            from_checkpoint=True,
        )
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is True
        assert cfg.bits == 8

    def test_bnb_4bit_config(self):
        """BNB 4-bit config should be correctly identified."""
        cfg = QuantizationConfig(
            method=QuantizationMethod.BNB_4BIT,
            bits=4,
            from_checkpoint=True,
            double_quant=True,
        )
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is True
        assert cfg.double_quant is True

    def test_awq_config(self):
        """AWQ config should be correctly identified."""
        cfg = QuantizationConfig(
            method=QuantizationMethod.AWQ,
            bits=4,
            group_size=128,
            from_checkpoint=True,
        )
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is False
        assert cfg.group_size == 128

    def test_gptq_config(self):
        """GPTQ config should be correctly identified."""
        cfg = QuantizationConfig(
            method=QuantizationMethod.GPTQ,
            bits=4,
            group_size=128,
            from_checkpoint=True,
        )
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is False

    def test_torchao_int8_config(self):
        """torchao int8 config should be correctly identified."""
        cfg = QuantizationConfig(method=QuantizationMethod.TORCHAO_INT8, bits=8)
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is False

    def test_hqq_config(self):
        """HQQ config should be correctly identified."""
        cfg = QuantizationConfig(method=QuantizationMethod.HQQ, bits=4, group_size=64)
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is False

    def test_quanto_config(self):
        """Quanto config should be correctly identified."""
        cfg = QuantizationConfig(method=QuantizationMethod.QUANTO, bits=8)
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is False

    def test_compressed_tensors_config(self):
        """compressed-tensors config should be correctly identified."""
        cfg = QuantizationConfig(
            method=QuantizationMethod.COMPRESSED_TENSORS,
            bits=8,
            from_checkpoint=True,
        )
        assert cfg.is_quantized() is True
        assert cfg.is_bnb() is False

    def test_frozen_immutable(self):
        """QuantizationConfig should be immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        cfg = QuantizationConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.bits = 8


class TestModelCapabilities:
    """Tests for ModelCapabilities dataclass."""

    def test_default_capabilities(self):
        """Default capabilities should be for FP16 movable model."""
        caps = ModelCapabilities()
        assert caps.device_movable is True
        assert caps.quantization.is_quantized() is False
        assert caps.primary_metric_kind == "ppl_causal"

    def test_for_fp16_model(self):
        """Factory for FP16 model should create movable capabilities."""
        caps = ModelCapabilities.for_fp16_model()
        assert caps.device_movable is True
        assert caps.quantization.method == QuantizationMethod.NONE
        assert caps.quantization.bits == 16

    def test_for_bnb_8bit(self):
        """Factory for BNB 8-bit should create non-movable capabilities."""
        caps = ModelCapabilities.for_bnb_8bit(from_checkpoint=True)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.BNB_8BIT
        assert caps.quantization.bits == 8
        assert caps.quantization.from_checkpoint is True

    def test_for_bnb_4bit(self):
        """Factory for BNB 4-bit should create non-movable capabilities."""
        caps = ModelCapabilities.for_bnb_4bit(double_quant=True)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.BNB_4BIT
        assert caps.quantization.double_quant is True

    def test_for_awq(self):
        """Factory for AWQ should create non-movable capabilities."""
        caps = ModelCapabilities.for_awq(group_size=64)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.AWQ
        assert caps.quantization.group_size == 64

    def test_for_gptq(self):
        """Factory for GPTQ should create non-movable capabilities."""
        caps = ModelCapabilities.for_gptq(bits=8, group_size=64)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.GPTQ
        assert caps.quantization.bits == 8

    def test_for_torchao_int8(self):
        """Factory for torchao int8 should create non-movable capabilities."""
        caps = ModelCapabilities.for_torchao_int8()
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.TORCHAO_INT8
        assert caps.quantization.bits == 8

    def test_for_hqq(self):
        """Factory for HQQ should create non-movable capabilities."""
        caps = ModelCapabilities.for_hqq(bits=4, group_size=64)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.HQQ
        assert caps.quantization.bits == 4
        assert caps.quantization.group_size == 64

    def test_for_quanto(self):
        """Factory for Quanto should create non-movable capabilities."""
        caps = ModelCapabilities.for_quanto(bits=8)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.QUANTO
        assert caps.quantization.bits == 8

    def test_for_compressed_tensors(self):
        """Factory for compressed-tensors should create non-movable capabilities."""
        caps = ModelCapabilities.for_compressed_tensors(bits=4)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.COMPRESSED_TENSORS
        assert caps.quantization.bits == 4
        assert caps.quantization.from_checkpoint is True


class TestDetectQuantizationFromConfig:
    """Tests for detect_quantization_from_config function."""

    def test_none_config(self):
        """None config should return default (no quantization)."""
        cfg = detect_quantization_from_config(None)
        assert cfg.method == QuantizationMethod.NONE

    def test_config_without_quantization(self):
        """Config without quantization_config should return default."""
        mock_config = MagicMock(spec=[])
        mock_config.quantization_config = None
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.NONE

    def test_dict_style_bnb_8bit(self):
        """Dict-style BNB 8-bit config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "bitsandbytes",
            "bits": 8,
        }
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.BNB_8BIT
        assert cfg.bits == 8
        assert cfg.from_checkpoint is True

    def test_dict_style_bnb_4bit(self):
        """Dict-style BNB 4-bit config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "bitsandbytes",
            "bits": 4,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16",
        }
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.BNB_4BIT
        assert cfg.bits == 4
        assert cfg.double_quant is True
        assert cfg.compute_dtype == "float16"

    def test_dict_style_awq(self):
        """Dict-style AWQ config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.AWQ
        assert cfg.bits == 4
        assert cfg.group_size == 128

    def test_dict_style_gptq(self):
        """Dict-style GPTQ config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
        }
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.GPTQ
        assert cfg.bits == 4
        assert cfg.group_size == 128

    def test_dict_style_torchao(self):
        """Dict-style torchao config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {"quant_method": "torchao"}
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.TORCHAO_INT8
        assert cfg.bits == 8

    def test_dict_style_hqq(self):
        """Dict-style HQQ config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "hqq",
            "nbits": 4,
            "group_size": 64,
        }
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.HQQ
        assert cfg.bits == 4
        assert cfg.group_size == 64

    def test_dict_style_quanto(self):
        """Dict-style Quanto config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {"quant_method": "quanto", "weights": "int8"}
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.QUANTO
        assert cfg.bits == 8

    def test_dict_style_quanto_int4_weights(self):
        """Dict-style Quanto weight precision should determine bit width."""
        mock_config = MagicMock()
        mock_config.quantization_config = {"quant_method": "quanto", "weights": "int4"}
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.QUANTO
        assert cfg.bits == 4

    def test_dict_style_compressed_tensors(self):
        """Dict-style compressed-tensors config should be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                    }
                }
            },
        }
        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.COMPRESSED_TENSORS
        assert cfg.bits == 4

    def test_dict_style_quantization_config_tolerates_malformed_values(self):
        """Malformed serialized config fields should not crash detection."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": 7,
            "bits": "4",
            "group_size": "128",
        }

        cfg = detect_quantization_from_config(mock_config)

        assert cfg.method == QuantizationMethod.NONE

    def test_dict_style_quantization_config_normalizes_numeric_strings(self):
        """Serialized string numeric fields should still be detected."""
        mock_config = MagicMock()
        mock_config.quantization_config = {
            "quant_method": "gptq",
            "bits": "4",
            "group_size": "128",
        }

        cfg = detect_quantization_from_config(mock_config)

        assert cfg.method == QuantizationMethod.GPTQ
        assert cfg.bits == 4
        assert cfg.group_size == 128

    def test_object_style_bnb_8bit(self):
        """Object-style BitsAndBytesConfig should be detected."""
        mock_quant_cfg = MagicMock()
        mock_quant_cfg.__class__.__name__ = "BitsAndBytesConfig"
        mock_quant_cfg.load_in_8bit = True
        mock_quant_cfg.load_in_4bit = False

        mock_config = MagicMock()
        mock_config.quantization_config = mock_quant_cfg

        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.BNB_8BIT

    def test_object_style_bnb_4bit(self):
        """Object-style BitsAndBytesConfig 4-bit should be detected."""
        mock_quant_cfg = MagicMock()
        mock_quant_cfg.__class__.__name__ = "BitsAndBytesConfig"
        mock_quant_cfg.load_in_8bit = False
        mock_quant_cfg.load_in_4bit = True
        mock_quant_cfg.bnb_4bit_use_double_quant = True

        mock_config = MagicMock()
        mock_config.quantization_config = mock_quant_cfg

        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.BNB_4BIT
        assert cfg.double_quant is True

    def test_object_style_hqq(self):
        """Object-style HqqConfig should be detected."""
        mock_quant_cfg = MagicMock()
        mock_quant_cfg.__class__.__name__ = "HqqConfig"
        mock_quant_cfg.nbits = 4
        mock_quant_cfg.group_size = 64

        mock_config = MagicMock()
        mock_config.quantization_config = mock_quant_cfg

        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.HQQ
        assert cfg.bits == 4
        assert cfg.group_size == 64

    def test_object_style_quanto(self):
        """Object-style QuantoConfig should be detected."""
        mock_quant_cfg = MagicMock()
        mock_quant_cfg.__class__.__name__ = "QuantoConfig"

        mock_config = MagicMock()
        mock_config.quantization_config = mock_quant_cfg

        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.QUANTO
        assert cfg.bits == 8

    def test_object_style_compressed_tensors(self):
        """Object-style compressed-tensors config should be detected."""
        mock_quant_cfg = MagicMock()
        mock_quant_cfg.__class__.__name__ = "CompressedTensorsConfig"
        mock_quant_cfg.config_groups = {
            "group_0": {"weights": {"num_bits": 4}},
        }
        mock_quant_cfg.to_dict.return_value = {
            "config_groups": mock_quant_cfg.config_groups,
        }

        mock_config = MagicMock()
        mock_config.quantization_config = mock_quant_cfg

        cfg = detect_quantization_from_config(mock_config)
        assert cfg.method == QuantizationMethod.COMPRESSED_TENSORS
        assert cfg.bits == 4

    def test_dynamic_nested_config_fails_closed_without_unbounded_recursion(self):
        mock_quant_cfg = MagicMock()
        mock_quant_cfg.__class__.__name__ = "CompressedTensorsConfig"
        mock_quant_cfg.to_dict.return_value = None

        mock_config = MagicMock()
        mock_config.quantization_config = mock_quant_cfg

        cfg = detect_quantization_from_config(mock_config)

        assert cfg.method == QuantizationMethod.COMPRESSED_TENSORS
        assert cfg.bits is None


class TestDetectCapabilitiesFromModel:
    """Tests for detect_capabilities_from_model function."""

    def test_fp16_model(self):
        """FP16 model should have movable capabilities."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = ["MistralForCausalLM"]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is True
        assert caps.quantization.is_quantized() is False
        assert caps.primary_metric_kind == "ppl_causal"

    def test_bnb_8bit_model(self):
        """BNB 8-bit model should have non-movable capabilities."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = {
            "quant_method": "bitsandbytes",
            "bits": 8,
        }
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = ["MistralForCausalLM"]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.BNB_8BIT

    def test_awq_model(self):
        """AWQ model should have non-movable capabilities."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        mock_model.config.model_type = "mistral"

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.AWQ

    def test_torchao_module_model(self):
        """torchao module markers should have non-movable capabilities."""

        class TorchAoLinear:
            __module__ = "torchao.dtypes.affine_quantized_tensor"

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = ["MistralForCausalLM"]
        mock_model.modules.return_value = [TorchAoLinear()]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.TORCHAO_INT8

    def test_hqq_module_model(self):
        """HQQ module markers should have non-movable capabilities."""

        class HQQLinear:
            __module__ = "hqq.core.quantize"

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = ["MistralForCausalLM"]
        mock_model.modules.return_value = [HQQLinear()]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.HQQ

    def test_quanto_module_model(self):
        """Quanto module markers should have non-movable capabilities."""

        class QLinear:
            __module__ = "optimum.quanto.nn.qlinear"

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = ["MistralForCausalLM"]
        mock_model.modules.return_value = [QLinear()]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.QUANTO

    def test_compressed_tensors_module_model(self):
        """compressed-tensors module markers should have non-movable capabilities."""

        class CompressedLinear:
            __module__ = "compressed_tensors.quantization.linear"

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = ["MistralForCausalLM"]
        mock_model.modules.return_value = [CompressedLinear()]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is False
        assert caps.quantization.method == QuantizationMethod.COMPRESSED_TENSORS

    def test_bert_model_metric(self):
        """BERT model should use MLM metric."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "bert"
        mock_model.config.architectures = ["BertForMaskedLM"]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.primary_metric_kind == "ppl_mlm"

    def test_t5_model_metric(self):
        """T5 model should use seq2seq metric."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "t5"
        mock_model.config.architectures = ["T5ForConditionalGeneration"]

        caps = detect_capabilities_from_model(mock_model)
        assert caps.primary_metric_kind == "ppl_seq2seq"

    def test_weight_tying_embed_tokens(self):
        """Weight tying should be detected for embed_tokens-style models."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "mistral"
        mock_model.config.architectures = []

        # Create shared weight tensor
        import torch

        shared_weight = torch.randn(32000, 4096)
        mock_model.lm_head.weight = shared_weight
        mock_model.model.embed_tokens.weight = shared_weight

        caps = detect_capabilities_from_model(mock_model)
        assert "lm_head.weight" in caps.weight_tied
        assert caps.weight_tied["lm_head.weight"] == "model.embed_tokens.weight"

    def test_model_without_config(self):
        """Model without config should return default capabilities."""
        mock_model = MagicMock(spec=[])

        caps = detect_capabilities_from_model(mock_model)
        assert caps.device_movable is True
        assert caps.quantization.is_quantized() is False


class TestSafeDeviceMovement:
    """Tests for safe device movement based on capabilities."""

    def test_fp16_can_move(self):
        """FP16 model capabilities should allow device movement."""
        caps = ModelCapabilities.for_fp16_model()
        assert caps.device_movable is True

    def test_bnb_cannot_move(self):
        """BNB model capabilities should not allow device movement."""
        caps = ModelCapabilities.for_bnb_8bit()
        assert caps.device_movable is False

    def test_awq_cannot_move(self):
        """AWQ model capabilities should not allow device movement."""
        caps = ModelCapabilities.for_awq()
        assert caps.device_movable is False

    def test_gptq_cannot_move(self):
        """GPTQ model capabilities should not allow device movement."""
        caps = ModelCapabilities.for_gptq()
        assert caps.device_movable is False
