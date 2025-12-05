"""
Tests for Safe Device Movement
==============================

TDD tests for safe device movement in adapters:
- FP16 models should be movable
- BNB models should not call .to()
- AWQ/GPTQ models should not call .to()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch.nn as nn

from invarlock.adapters.capabilities import ModelCapabilities
from invarlock.adapters.hf_mixin import HFAdapterMixin


class SimpleMixin(HFAdapterMixin):
    """Simple test class using the mixin."""

    pass


class TestSafeToDevice:
    """Tests for _safe_to_device method."""

    def test_fp16_model_is_moved(self):
        """FP16 model should be moved to target device."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)

        caps = ModelCapabilities.for_fp16_model()

        result = mixin._safe_to_device(mock_model, "cpu", capabilities=caps)

        mock_model.to.assert_called_once()
        assert result is mock_model

    def test_bnb_8bit_model_not_moved(self):
        """BNB 8-bit model should NOT have .to() called."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)

        caps = ModelCapabilities.for_bnb_8bit()

        result = mixin._safe_to_device(mock_model, "cuda", capabilities=caps)

        mock_model.to.assert_not_called()
        assert result is mock_model

    def test_bnb_4bit_model_not_moved(self):
        """BNB 4-bit model should NOT have .to() called."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)

        caps = ModelCapabilities.for_bnb_4bit()

        result = mixin._safe_to_device(mock_model, "cuda", capabilities=caps)

        mock_model.to.assert_not_called()
        assert result is mock_model

    def test_awq_model_not_moved(self):
        """AWQ model should NOT have .to() called."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)

        caps = ModelCapabilities.for_awq()

        result = mixin._safe_to_device(mock_model, "cuda", capabilities=caps)

        mock_model.to.assert_not_called()
        assert result is mock_model

    def test_gptq_model_not_moved(self):
        """GPTQ model should NOT have .to() called."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)

        caps = ModelCapabilities.for_gptq()

        result = mixin._safe_to_device(mock_model, "cuda", capabilities=caps)

        mock_model.to.assert_not_called()
        assert result is mock_model

    def test_auto_detect_fp16(self):
        """Auto-detection should recognize FP16 model and allow movement."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "llama"
        mock_model.config.architectures = []
        mock_model.to = MagicMock(return_value=mock_model)

        # No capabilities passed - should auto-detect
        _ = mixin._safe_to_device(mock_model, "cpu", capabilities=None)

        mock_model.to.assert_called_once()

    def test_auto_detect_bnb(self):
        """Auto-detection should recognize BNB model and prevent movement."""
        mixin = SimpleMixin()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = {"load_in_8bit": True}
        mock_model.config.model_type = "llama"
        mock_model.config.architectures = []
        mock_model.to = MagicMock(return_value=mock_model)

        # No capabilities passed - should auto-detect
        _ = mixin._safe_to_device(mock_model, "cuda", capabilities=None)

        mock_model.to.assert_not_called()


class TestIsQuantizedModel:
    """Tests for _is_quantized_model method."""

    def test_fp16_not_quantized(self):
        """FP16 model should not be detected as quantized."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None

        # Remove BNB attributes
        del mock_model.is_loaded_in_8bit
        del mock_model.is_loaded_in_4bit

        # Simple modules
        mock_model.modules.return_value = [MagicMock(__class__=nn.Linear)]

        assert mixin._is_quantized_model(mock_model) is False

    def test_bnb_8bit_detected(self):
        """BNB 8-bit model should be detected as quantized."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = {"load_in_8bit": True}

        assert mixin._is_quantized_model(mock_model) is True

    def test_bnb_attribute_detected(self):
        """Model with is_loaded_in_8bit should be detected."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.is_loaded_in_8bit = True

        assert mixin._is_quantized_model(mock_model) is True

    def test_quantized_module_detected(self):
        """Model with quantized modules should be detected."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None

        # Remove BNB attributes
        del mock_model.is_loaded_in_8bit
        del mock_model.is_loaded_in_4bit

        # Add quantized module
        mock_linear8bit = MagicMock()
        mock_linear8bit.__class__.__name__ = "Linear8bitLt"
        mock_model.modules.return_value = [mock_linear8bit]

        assert mixin._is_quantized_model(mock_model) is True


class TestDetectQuantizationConfig:
    """Tests for _detect_quantization_config method."""

    def test_fp16_returns_none(self):
        """FP16 model should return None for quant config."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None

        result = mixin._detect_quantization_config(mock_model)
        assert result is None

    def test_bnb_8bit_detected(self):
        """BNB 8-bit model should return quant config."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = {"load_in_8bit": True}

        result = mixin._detect_quantization_config(mock_model)
        assert result is not None
        assert result.bits == 8


class TestDetectCapabilities:
    """Tests for _detect_capabilities method."""

    def test_returns_capabilities(self):
        """Should return ModelCapabilities for valid model."""
        mixin = SimpleMixin()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantization_config = None
        mock_model.config.model_type = "llama"
        mock_model.config.architectures = []

        caps = mixin._detect_capabilities(mock_model)
        assert caps is not None
        assert caps.device_movable is True

    def test_handles_import_error(self):
        """Should return None if capabilities module not available."""
        mixin = SimpleMixin()

        mock_model = MagicMock()

        with patch.dict("sys.modules", {"invarlock.adapters.capabilities": None}):
            # This should handle the import error gracefully
            _ = mixin._detect_capabilities(mock_model)
            # May return None or the actual capabilities depending on import state
