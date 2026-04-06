import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from invarlock.reporting.report_contract import (
    _assert_evaluation_report_is_finite,
    _describe_run_report_health_error,
    _extract_saved_provenance_env_flags,
    _is_non_bool_finite_number,
    generate_reports,
)


def test_extract_saved_provenance_env_flags_prefers_provenance_then_meta():
    assert _extract_saved_provenance_env_flags(None) is None
    assert (
        _extract_saved_provenance_env_flags(
            {"provenance": {"env_flags": {}}, "meta": None}
        )
        is None
    )
    assert _extract_saved_provenance_env_flags(
        {
            "provenance": {"env_flags": {"cuda_matmul_allow_tf32": False}},
            "meta": {"env_flags": {"cuda_matmul_allow_tf32": True}},
        }
    ) == {"cuda_matmul_allow_tf32": False}
    assert _extract_saved_provenance_env_flags(
        {
            "provenance": {"env_flags": {}},
            "meta": {"env_flags": {"cuda_matmul_allow_tf32": False}},
        }
    ) == {"cuda_matmul_allow_tf32": False}


def test_run_report_health_helpers_cover_edge_cases():
    assert _is_non_bool_finite_number(True) is False
    assert _is_non_bool_finite_number(object()) is False

    assert _describe_run_report_health_error(None, role="subject") is None
    assert (
        _describe_run_report_health_error(
            {"status": "completed", "metrics": []},
            role="subject",
        )
        is None
    )
    assert (
        _describe_run_report_health_error(
            {"metrics": {"primary_metric": {}}},
            role="subject",
        )
        is None
    )
    assert (
        _describe_run_report_health_error(
            {
                "metrics": {
                    "primary_metric": {
                        "degraded": True,
                        "degraded_reason": "malformed",
                    }
                }
            },
            role="subject",
        )
        == "Cannot generate evaluation report from subject run report with "
        "degraded primary metric (malformed)."
    )
    assert (
        _describe_run_report_health_error(
            {
                "metrics": {
                    "primary_metric": {
                        "preview": None,
                        "final": 10.0,
                        "ratio_vs_baseline": float("nan"),
                    }
                }
            },
            role="baseline",
        )
        is None
    )


def test_assert_evaluation_report_is_finite_covers_structure_and_optional_fields():
    with pytest.raises(ValueError, match="missing or malformed"):
        _assert_evaluation_report_is_finite(None)

    with pytest.raises(ValueError, match="missing a primary_metric block"):
        _assert_evaluation_report_is_finite({})

    with pytest.raises(ValueError, match="degraded primary metric \\(unstable\\)"):
        _assert_evaluation_report_is_finite(
            {
                "primary_metric": {
                    "degraded": True,
                    "degraded_reason": "unstable",
                }
            }
        )

    _assert_evaluation_report_is_finite({"primary_metric": {"final": 10.0}})
    _assert_evaluation_report_is_finite(
        {"primary_metric": {"preview": None, "final": 10.0}}
    )


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
    mock_cert.assert_called_once_with(
        run,
        baseline,
        provenance_env_flags=None,
    )
    mock_validate.assert_called_once()


@patch("invarlock.reporting.report_contract.save_evaluation_bundle")
@patch("invarlock.reporting.report_contract.load_report_payload")
@patch("invarlock.reporting.report_contract.make_report")
@patch("invarlock.reporting.report_contract.validate_report")
def test_report_command_reuses_saved_subject_env_flags_for_generated_report(
    mock_validate, mock_make_report, mock_load, mock_save_bundle
):
    run = {
        "meta": {
            "model_id": "gpt2",
            "env_flags": {"cuda_matmul_allow_tf32": False},
        },
        "edit": {"name": "quant_rtn"},
        "metrics": {
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
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
        },
    }

    mock_load.side_effect = lambda path: baseline if "baseline" in path else run
    mock_save_bundle.return_value = {"report": "evaluation.report.json"}
    mock_make_report.return_value = {
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

    mock_make_report.assert_called_once_with(
        run,
        baseline,
        provenance_env_flags={"cuda_matmul_allow_tf32": False},
    )
    mock_save_bundle.assert_called_once_with(
        run_report=run,
        output_dir="reports_run.json",
        evaluation_report=mock_make_report.return_value,
        source_run_path="run.json",
    )


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
def test_report_command_rejects_failed_subject_run_report(mock_make_report, mock_load):
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
@patch("invarlock.reporting.report_contract.validate_report")
def test_report_command_accepts_baseline_without_self_baseline_ratio(
    mock_validate, mock_make_report, mock_save_bundle, mock_load
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
            "window_pairing_reason": "no_baseline_reference",
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": float("nan"),
            },
        },
    }

    mock_load.side_effect = lambda path: baseline if "baseline" in path else run
    mock_make_report.return_value = {
        "validation": {"safety_check": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.5,
            "ratio_vs_baseline": 1.05,
        },
    }
    mock_save_bundle.return_value = {"report": "evaluation.report.json"}
    mock_validate.return_value = True

    generate_reports(
        run="run.json",
        format="report",
        compare=None,
        baseline="baseline.json",
        output=None,
    )

    mock_make_report.assert_called_once()
    mock_validate.assert_called_once()
    mock_save_bundle.assert_called_once()


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


@patch("invarlock.reporting.report_contract.load_report_payload")
@patch("invarlock.reporting.report_contract.make_report")
@patch("invarlock.reporting.report_contract.save_evaluation_bundle")
@patch("invarlock.reporting.report_contract.validate_report")
def test_report_command_allows_non_finite_input_ratio_without_pairing_reason(
    mock_validate, mock_save_bundle, mock_make_report, mock_load
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
                "ratio_vs_baseline": float("nan"),
            },
        },
    }

    mock_load.side_effect = lambda path: baseline if "baseline" in path else run
    mock_make_report.return_value = {
        "validation": {"safety_check": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.5,
            "ratio_vs_baseline": 1.05,
        },
    }
    mock_save_bundle.return_value = {"report": "evaluation.report.json"}
    mock_validate.return_value = True

    generate_reports(
        run="run.json",
        format="report",
        compare=None,
        baseline="baseline.json",
        output=None,
    )

    mock_make_report.assert_called_once()
    mock_validate.assert_called_once()
    mock_save_bundle.assert_called_once()


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
