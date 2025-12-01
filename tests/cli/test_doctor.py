import sys
from unittest.mock import Mock, patch

import pytest

from invarlock.cli.commands.doctor import doctor_command


def test_doctor_command_healthy():
    with (
        patch("invarlock.core.registry.get_registry") as mock_registry,
        patch("invarlock.cli.device.get_device_info") as mock_device_info,
    ):
        mock_device_info.return_value = {
            "cpu": {"available": True, "info": "Always available"},
            "cuda": {
                "available": True,
                "device_count": 1,
                "device_name": "RTX 4090",
                "memory_total": "24.0 GB",
            },
            "mps": {"available": False, "info": "Not available"},
            "auto_selected": "cuda",
        }
        mock_reg = Mock()
        mock_reg.list_adapters.return_value = ["hf_gpt2", "hf_causal_auto"]
        mock_reg.list_edits.return_value = ["structured", "gptq"]
        mock_reg.list_guards.return_value = ["invariants", "spectral"]
        mock_reg.get_plugin_info.side_effect = lambda name, kind: {
            "module": "invarlock.adapters"
            if kind == "adapters"
            else f"invarlock.{kind}",
            "entry_point": name,
        }
        mock_registry.return_value = mock_reg

        with patch("invarlock.cli.commands.doctor.console"):
            with patch("torch.__version__", "2.0.0"):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.get_device_properties") as mock_props:
                        mock_props.return_value.total_memory = 24 * 1024**3
                        with pytest.raises(SystemExit) as exc_info:
                            doctor_command()
                        assert exc_info.value.code == 0


def test_doctor_command_missing_core():
    with patch(
        "invarlock.core.registry.get_registry", side_effect=ImportError("No core")
    ):
        with patch("invarlock.cli.commands.doctor.console"):
            with pytest.raises(SystemExit) as exc_info:
                doctor_command()
            assert exc_info.value.code == 1


def test_doctor_command_no_torch():
    with patch("invarlock.cli.commands.doctor.console"):
        torch_was_loaded = "torch" in sys.modules
        if torch_was_loaded:
            torch_module = sys.modules.pop("torch")

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return Mock()

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(SystemExit) as exc_info:
                    doctor_command()
                assert exc_info.value.code == 1
        finally:
            if torch_was_loaded:
                sys.modules["torch"] = torch_module


def test_doctor_command_optional_deps():
    with patch("invarlock.cli.device.get_device_info") as mock_device_info:
        mock_device_info.return_value = {
            "cpu": {"available": True, "info": "Always available"},
            "auto_selected": "cpu",
        }

        with patch("invarlock.cli.commands.doctor.console") as mock_console:
            with patch("torch.__version__", "2.0.0"):

                def mock_import(name, *args, **kwargs):
                    if name == "datasets":
                        return Mock()
                    else:
                        raise ImportError(f"No module named '{name}'")

                with patch("builtins.__import__", side_effect=mock_import):
                    try:
                        doctor_command()
                    except SystemExit:
                        pass
                    # Should reference datasets in console output
                    calls = mock_console.print.call_args_list
                    call_text = " ".join([str(call) for call in calls])
                    assert "datasets" in call_text
                    # Adapters table header may be skipped when core registry/imports are unavailable
