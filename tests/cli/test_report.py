import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from invarlock.cli.commands.report import report_command


@patch("invarlock.reporting.report.save_report")
@patch("invarlock.cli.commands.report._load_run_report")
def test_report_command_basic(mock_load, mock_save):
    mock_report = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.05},
    }
    mock_load.return_value = mock_report
    mock_save.return_value = {"json": "report.json"}

    with patch("invarlock.cli.commands.report.console"):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir) / "run.json"
            run_path.write_text(json.dumps(mock_report))

            report_command(
                run=str(run_path),
                format="json",
                compare=None,
                baseline=None,
                output=None,
            )
            mock_save.assert_called_once()


@patch("invarlock.cli.commands.report._load_run_report")
def test_report_command_with_comparison(mock_load):
    r1 = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.05},
    }
    r2 = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "gptq"},
        "metrics": {"ppl_ratio": 1.08},
    }

    def mock_side(path):
        return r1 if "run1" in path else r2

    mock_load.side_effect = mock_side
    with (
        patch("invarlock.reporting.report.save_report") as mock_save,
        patch("invarlock.cli.commands.report.console"),
    ):
        mock_save.return_value = {"json": "report.json"}
        report_command(
            run="run1.json",
            format="json",
            compare="run2.json",
            baseline=None,
            output=None,
        )
        assert mock_load.call_count == 2
        mock_save.assert_called_once()


@patch("invarlock.cli.commands.report._load_run_report")
def test_report_command_certificate_no_baseline(mock_load):
    mock_report = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.05},
    }
    mock_load.return_value = mock_report
    with patch("invarlock.cli.commands.report.console"):
        with pytest.raises((SystemExit, Exception)):
            report_command(
                run="run.json", format="cert", compare=None, baseline=None, output=None
            )


@patch("invarlock.reporting.report.save_report")
@patch("invarlock.cli.commands.report._load_run_report")
@patch("invarlock.reporting.certificate.make_certificate")
@patch("invarlock.reporting.certificate.validate_certificate")
def test_report_command_certificate_with_baseline(
    mock_validate, mock_cert, mock_load, mock_save
):
    run = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.05},
    }
    baseline = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "baseline"},
        "metrics": {"ppl_ratio": 1.0},
    }

    def side(path):
        return baseline if "baseline" in path else run

    mock_load.side_effect = side
    mock_save.return_value = {"cert": "certificate.json"}
    mock_cert.return_value = {"validation": {"safety_check": True}}
    mock_validate.return_value = True

    with patch("invarlock.cli.commands.report.console"):
        report_command(
            run="run.json",
            format="cert",
            compare=None,
            baseline="baseline.json",
            output=None,
        )
        mock_cert.assert_called_once()
        mock_validate.assert_called_once()


def test_load_run_report_file(tmp_path: Path):
    from invarlock.cli.commands.report import _load_run_report

    payload = {"test": "data"}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(payload))
    assert _load_run_report(str(p)) == payload


def test_load_run_report_directory(tmp_path: Path):
    from invarlock.cli.commands.report import _load_run_report

    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"x": 1}))
    assert _load_run_report(str(tmp_path)) == {"x": 1}


def test_load_run_report_not_found():
    from invarlock.cli.commands.report import _load_run_report

    with pytest.raises(FileNotFoundError):
        _load_run_report("nonexistent.json")
