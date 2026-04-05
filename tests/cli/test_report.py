import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from invarlock.reporting.report_contract import generate_reports


@patch("invarlock.reporting.report_contract.save_report")
@patch("invarlock.reporting.report_contract.load_report_payload")
def test_report_command_basic(mock_load, mock_save):
    mock_report = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.05},
    }
    mock_load.return_value = mock_report
    mock_save.return_value = {"json": "report.json"}

    with tempfile.TemporaryDirectory() as temp_dir:
        run_path = Path(temp_dir) / "run.json"
        run_path.write_text(json.dumps(mock_report))

        generate_reports(
            run=str(run_path),
            format="json",
            compare=None,
            baseline=None,
            output=None,
        )
        mock_save.assert_called_once()


@patch("invarlock.reporting.report_contract.load_report_payload")
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
    with patch("invarlock.reporting.report_contract.save_report") as mock_save:
        mock_save.return_value = {"json": "report.json"}
        generate_reports(
            run="run1.json",
            format="json",
            compare="run2.json",
            baseline=None,
            output=None,
        )
        assert mock_load.call_count == 2
        mock_save.assert_called_once()


@patch("invarlock.reporting.report_contract.load_report_payload")
def test_report_command_evaluation_report_no_baseline(mock_load):
    mock_report = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.05},
    }
    mock_load.return_value = mock_report
    with pytest.raises(ValueError, match="requires --baseline"):
        generate_reports(
            run="run.json",
            format="report",
            compare=None,
            baseline=None,
            output=None,
        )


@patch("invarlock.reporting.report_contract.save_evaluation_bundle")
@patch("invarlock.reporting.report_contract.load_report_payload")
@patch("invarlock.reporting.report_contract.make_report")
@patch("invarlock.reporting.report_contract.validate_report")
def test_report_command_evaluation_report_with_baseline(
    mock_validate, mock_cert, mock_load, mock_save_bundle
):
    run = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {
            "ppl_ratio": 1.05,
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.5,
                "ratio_vs_baseline": 1.05,
            },
        },
    }
    baseline = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "baseline"},
        "metrics": {
            "ppl_ratio": 1.0,
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
        },
    }

    def side(path):
        return baseline if "baseline" in path else run

    mock_load.side_effect = side
    mock_save_bundle.return_value = {"report": "evaluation.report.json"}
    mock_cert.return_value = {
        "validation": {"safety_check": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.5,
            "ratio_vs_baseline": 1.05,
        },
    }
    mock_validate.return_value = True

    generate_reports(
        run="run.json",
        format="report",
        compare=None,
        baseline="baseline.json",
        output=None,
    )
    mock_cert.assert_called_once()
    mock_validate.assert_called_once()


@patch("invarlock.reporting.report_contract.load_report_payload")
@patch("invarlock.reporting.report_contract.make_report")
def test_report_command_rejects_non_finite_subject_primary_metric(
    mock_make_report, mock_load
):
    run = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": float("nan"),
                "final": float("nan"),
                "ratio_vs_baseline": float("nan"),
            }
        },
    }
    baseline = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "baseline"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            }
        },
    }

    mock_load.side_effect = lambda path: baseline if "baseline" in path else run

    with pytest.raises(
        ValueError,
        match="subject run report with non-finite primary metric field 'preview'",
    ):
        generate_reports(
            run="run.json",
            format="report",
            compare=None,
            baseline="baseline.json",
            output=None,
        )

    mock_make_report.assert_not_called()


@patch("invarlock.reporting.report_contract.load_report_payload")
@patch("invarlock.reporting.report_contract.make_report")
def test_report_command_rejects_failed_subject_run_report(
    mock_make_report, mock_load
):
    run = {
        "meta": {"model_id": "gpt2"},
        "status": "failed",
        "edit": {"name": "quant_rtn"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.5,
                "ratio_vs_baseline": 1.05,
            }
        },
    }
    baseline = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "baseline"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            }
        },
    }

    mock_load.side_effect = lambda path: baseline if "baseline" in path else run

    with pytest.raises(
        ValueError,
        match="subject run report with status 'failed'",
    ):
        generate_reports(
            run="run.json",
            format="report",
            compare=None,
            baseline="baseline.json",
            output=None,
        )

    mock_make_report.assert_not_called()


@patch("invarlock.reporting.report_contract.load_report_payload")
@patch("invarlock.reporting.report_contract.save_evaluation_bundle")
@patch("invarlock.reporting.report_contract.make_report")
def test_report_command_rejects_non_finite_generated_evaluation_report(
    mock_make_report, mock_save_bundle, mock_load
):
    run = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.5,
                "ratio_vs_baseline": 1.05,
            }
        },
    }
    baseline = {
        "meta": {"model_id": "gpt2"},
        "edit": {"name": "baseline"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            }
        },
    }

    mock_load.side_effect = lambda path: baseline if "baseline" in path else run
    mock_make_report.return_value = {
        "validation": {"safety_check": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": float("nan"),
            "ratio_vs_baseline": float("nan"),
        },
    }

    with pytest.raises(
        ValueError,
        match="Generated evaluation report contains non-finite primary metric field",
    ):
        generate_reports(
            run="run.json",
            format="report",
            compare=None,
            baseline="baseline.json",
            output=None,
        )

    mock_save_bundle.assert_not_called()


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


def test_load_run_report_directory_ambiguous(tmp_path: Path):
    from invarlock.cli.commands.report import _load_run_report

    (tmp_path / "report.json").write_text(json.dumps({"x": 1}))
    (tmp_path / "evaluation.report.json").write_text(json.dumps({"x": 2}))

    assert _load_run_report(str(tmp_path)) == {"x": 1}


def test_load_run_report_directory_requires_canonical_filename(tmp_path: Path):
    from invarlock.cli.commands.report import _load_run_report

    (tmp_path / "subject_report.json").write_text(json.dumps({"x": 1}))

    with pytest.raises(ValueError, match="canonical run report file"):
        _load_run_report(str(tmp_path))


def test_load_run_report_not_found():
    from invarlock.cli.commands.report import _load_run_report

    with pytest.raises(ValueError, match="Path not found"):
        _load_run_report("nonexistent.json")
