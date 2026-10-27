"""Cheap report-validation checks without timing data or chart rendering."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "render_optimization_results",
    Path(__file__).parents[1] / "render_optimization_results.py",
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def row(output):
    return {
        "output_directory": output,
        "payload": {
            "result": {
                "outputs": [
                    {
                        "payload": {
                            "report": {
                                "requested_outputs": {"html": output + "/report.html"},
                                "written_outputs": {"html": output + "/report.html"},
                                "digest": "unchanged",
                            }
                        }
                    }
                ],
                "files": {"html": "sha256:unchanged"},
            }
        },
    }


def test_comparison_normalizes_only_declared_destinations():
    left, right = row("/linux/one"), row("/linux/two")
    assert report.comparison(left) == report.comparison(right)
    right["payload"]["result"]["outputs"][0]["payload"]["report"]["digest"] = "changed"
    assert report.comparison(left) != report.comparison(right)
    assert left == row("/linux/one")


def test_escaping_report_destination_is_rejected():
    value = row("/linux/one")
    value["payload"]["result"]["outputs"][0]["payload"]["report"]["written_outputs"][
        "html"
    ] = "/elsewhere/report.html"
    with pytest.raises(ValueError, match="escaped"):
        report.comparison(value)


def test_incomplete_run_stops_before_public_output(tmp_path):
    for name in ("metadata.json", "environment.json", "manifest.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "summary.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        report.validate(
            tmp_path, tmp_path / "metadata.json", tmp_path / "environment.json"
        )


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_evidence_is_rejected(tmp_path, number):
    path = tmp_path / "sample.json"
    path.write_text('{"value": ' + number + "}")
    with pytest.raises(ValueError):
        report.read(path)


def test_source_revisions_require_explicit_full_commit_identity():
    metadata = {
        "source_commit": "a" * 40,
        "before_source": {"commit": "b" * 40},
        "after_source": {"commit": "c" * 40},
    }
    report.validate_sources(metadata, "a" * 40, "b" * 40, "c" * 40)
    with pytest.raises(ValueError, match="unexpected before"):
        report.validate_sources(metadata, "a" * 40, "d" * 40, "c" * 40)
    with pytest.raises(ValueError, match="full lowercase"):
        report.validate_sources(metadata, "a" * 40, "b" * 8, "c" * 40)
