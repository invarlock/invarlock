from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from invarlock.cli.commands.run import (
    run_command,
)
from invarlock.core.run_policy import GUARD_METRIC_DEGRADATION_LIMIT


def test_run_command_file_not_found():
    with (
        patch("invarlock.cli.run_execution.console"),
    ):
        with pytest.raises((SystemExit, typer.Exit)) as exc_info:
            run_command(
                config="nonexistent.yaml",
                device=None,
                profile=None,
                out=None,
                edit=None,
                tier=None,
                probes=None,
            )
        # Exit code 1
        if hasattr(exc_info.value, "code"):
            assert exc_info.value.code == 1
        else:
            assert exc_info.value.exit_code == 1


@patch("invarlock.core.config_loader.load_config")
@patch("invarlock.cli.run_config._resolve_requested_edit_name")
def test_run_command_invalid_edit_kind(mock_resolve_edit, mock_load):
    mock_config = Mock()
    mock_config.model.device = "auto"
    mock_config.output.dir = Path("./output")
    mock_config.eval.spike_threshold = 2.0
    mock_load.return_value = mock_config
    mock_resolve_edit.side_effect = ValueError("Invalid edit kind")

    with patch("invarlock.cli.run_execution.console"):
        with pytest.raises((SystemExit, typer.Exit)) as exc_info:
            run_command(
                config="test.yaml",
                device=None,
                profile=None,
                out=None,
                edit="invalid_kind",
                tier=None,
                probes=None,
            )
        if hasattr(exc_info.value, "code"):
            assert exc_info.value.code == 2
        else:
            assert exc_info.value.exit_code == 2


@patch("invarlock.core.config_loader.load_config")
def test_run_command_invalid_config_key_exits_2(mock_load):
    mock_load.side_effect = ValueError(
        "edit.parameters is not supported; use edit.plan."
    )

    with patch("invarlock.cli.run_execution.console"):
        with pytest.raises((SystemExit, typer.Exit)) as exc_info:
            run_command(
                config="test.yaml",
                device=None,
                profile=None,
                out=None,
                edit=None,
                tier=None,
                probes=None,
            )
        if hasattr(exc_info.value, "code"):
            assert exc_info.value.code == 2
        else:
            assert exc_info.value.exit_code == 2


@patch("invarlock.core.config_loader.load_config")
def test_run_command_invalid_tier(mock_load):
    mock_config = Mock()
    mock_config.model.device = "auto"
    mock_config.output.dir = Path("./output")
    mock_config.eval.spike_threshold = 2.0
    mock_load.return_value = mock_config

    with patch("invarlock.cli.run_execution.console"):
        with pytest.raises((SystemExit, typer.Exit)) as exc_info:
            run_command(
                config="test.yaml",
                device=None,
                profile=None,
                out=None,
                edit=None,
                tier="invalid_tier",
                probes=None,
            )
        if hasattr(exc_info.value, "code"):
            assert exc_info.value.code == 1
        else:
            assert exc_info.value.exit_code == 1


@patch("invarlock.core.config_loader.load_config")
def test_run_command_invalid_probes(mock_load):
    mock_config = Mock()
    mock_config.model.device = "auto"
    mock_config.output.dir = Path("./output")
    mock_config.eval.spike_threshold = 2.0
    mock_load.return_value = mock_config

    with patch("invarlock.cli.run_execution.console"):
        with pytest.raises((SystemExit, typer.Exit)) as exc_info:
            run_command(
                config="test.yaml",
                device=None,
                profile=None,
                out=None,
                edit=None,
                tier=None,
                probes=15,
            )
        if hasattr(exc_info.value, "code"):
            assert exc_info.value.code == 1
        else:
            assert exc_info.value.exit_code == 1


@patch("invarlock.cli.run_runtime_exec.validate_guard_metric_impact")
@patch("invarlock.core.config_loader.load_config")
@patch("invarlock.core.config_loader.apply_profile")
@patch("invarlock.cli.device.resolve_device", return_value="cpu")
@patch("invarlock.cli.device.validate_device_for_config", return_value=(True, ""))
@patch("invarlock.core.registry.get_registry")
@patch("invarlock.core.runner.CoreRunner")
@patch("invarlock.reporting.report_bundle.save_report")
@patch("invarlock.eval.data.get_provider")
@patch("pathlib.Path.mkdir")
def test_run_command_fails_when_guard_metric_impact_exceeds_budget(
    mock_mkdir,
    mock_get_provider,
    mock_save_report,
    mock_runner,
    mock_registry,
    mock_validate_device,
    mock_resolve_device,
    mock_apply_profile,
    mock_load,
    mock_validate_guard_metric_impact,
):
    mock_config = Mock()
    mock_config.model.device = "auto"
    mock_config.model.adapter = "hf_causal"
    mock_config.model.id = "gpt2"
    mock_config.output.dir = Path("./output")
    mock_config.eval.spike_threshold = 2.0
    mock_config.eval.loss = Mock()
    mock_config.eval.loss.type = "ce"
    mock_config.dataset.provider = None
    mock_config.dataset.split = "test"
    mock_config.dataset.seq_len = 512
    mock_config.dataset.preview_n = 10
    mock_config.dataset.final_n = 20
    mock_config.dataset.stride = 256
    mock_config.guards.order = ["spectral"]
    mock_config.edit.name = "structured"

    mock_load.return_value = mock_config
    mock_apply_profile.return_value = mock_config

    mock_adapter = Mock()
    mock_adapter.name = "hf_causal"
    bare_model = Mock(name="bare_model")
    guarded_model = Mock(name="guarded_model")
    mock_adapter.load_model.side_effect = [bare_model, guarded_model]

    mock_edit = Mock()
    mock_edit.name = "structured"
    mock_guard = Mock()
    mock_guard.name = "spectral"

    mock_reg = Mock()
    mock_reg.get_adapter.return_value = mock_adapter
    mock_reg.get_edit.return_value = mock_edit
    mock_reg.get_guard.return_value = mock_guard
    mock_registry.return_value = mock_reg

    bare_report = Mock()
    bare_report.metrics = {"ppl_preview": 9.7, "ppl_final": 9.8}
    bare_report.edit = {"deltas": {"params_changed": 0}}
    bare_report.guards = {}
    bare_report.context = {}
    bare_report.evaluation_windows = {}
    bare_report.status = "success"

    guarded_report = Mock()
    guarded_report.metrics = {
        "ppl_ratio": 1.04,
        "ppl_preview": 10.0,
        "ppl_final": 10.2,
        "window_overlap_fraction": 0.0,
        "window_match_fraction": 1.0,
        "loss_type": "ce",
    }
    guarded_report.edit = {"deltas": {"params_changed": 10}}
    guarded_report.guards = {"spectral": {"metrics": {}, "passed": True}}
    guarded_report.context = {"dataset_meta": {}}
    guarded_report.evaluation_windows = {
        "preview": {
            "window_ids": [0, 1],
            "logloss": [3.0, 3.1],
            "input_ids": [[1, 2], [3, 4]],
            "attention_masks": [[1, 1], [1, 1]],
        },
        "final": {
            "window_ids": [2, 3],
            "logloss": [3.2, 3.3],
            "input_ids": [[5, 6], [7, 8]],
            "attention_masks": [[1, 1], [1, 1]],
        },
    }

    bare_runner_instance = Mock()
    bare_runner_instance.execute.return_value = bare_report
    main_runner_instance = Mock()
    main_runner_instance.execute.return_value = guarded_report
    mock_runner.side_effect = [bare_runner_instance, main_runner_instance]

    mock_validate_guard_metric_impact.return_value = Mock(
        passed=False,
        degradation=1.1,
        display_value=10.0,
        degradation_limit=GUARD_METRIC_DEGRADATION_LIMIT,
        errors=["too slow"],
    )

    with patch("invarlock.cli.run_execution.console"):
        with pytest.raises((SystemExit, typer.Exit)) as exc_info:
            run_command(
                config="test.yaml",
                device=None,
                profile="release",
                out=None,
                edit=None,
                tier=None,
                probes=None,
            )
        # Exit on budget exceeded
        if hasattr(exc_info.value, "code"):
            assert exc_info.value.code == 1
        else:
            assert exc_info.value.exit_code == 1
