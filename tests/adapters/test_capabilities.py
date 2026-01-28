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

    def test_frozen_immutable(self):
        """QuantizationConfig should be immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        cfg = QuantizationConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.bits = 8  # type: ignore[misc]


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
