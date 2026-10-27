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


def cpu_fixture():
    samples, summary = [], []
    for candidate, users, systems, wall, rss in (
        ("reference", [1, 1, 1, 101], [101, 1, 1, 1], 10, 100),
        ("rust-ffi", [4] * 4, [1] * 4, 10, 80),
        ("rust-standalone", [60] * 4, [1] * 4, 8, 120),
    ):
        for round_index, (user, system) in enumerate(zip(users, systems, strict=True)):
            samples.append(
                {
                    "attempt": f"{candidate}-{round_index}",
                    "phase": "measurement",
                    "workload": "captured-nll-1000",
                    "candidate": candidate,
                    "round": round_index,
                    "metrics": {"user_cpu_seconds": user, "system_cpu_seconds": system},
                }
            )
        summary.append(
            {
                "workload": "captured-nll-1000",
                "candidate": candidate,
                "wall_seconds": {"median": wall},
                "max_rss_bytes": {"median": rss},
            }
        )
    samples.append({"phase": "qualification", "metrics": {"user_cpu_seconds": 999}})
    return samples, summary


def test_total_cpu_sums_attempts_before_summarizing_and_preserves_old_fields():
    samples, summary = cpu_fixture()
    original = json.dumps([samples, summary], sort_keys=True)
    derived = report.derived_metrics(samples, summary)
    cells = {item["candidate"]: item for item in derived["total_cpu_summary"]}
    # Component medians are 1 + 1 = 2; the median of the actual totals is 52.
    assert cells["reference"]["total_cpu_seconds"] == {
        "median": 52,
        "minimum": 2,
        "maximum": 102,
    }
    assert len(derived["total_cpu_samples"]) == 12
    assert all(item["n"] == 4 for item in cells.values())
    assert derived["qualification_samples_included"] is False
    assert json.dumps([samples, summary], sort_keys=True) == original


def test_python_comparisons_describe_each_observed_median_without_threshold():
    samples, summary = cpu_fixture()
    derived = report.derived_metrics(samples, summary)
    cells = {item["candidate"]: item for item in derived["versus_python"]}
    ffi = cells["rust-ffi"]["metrics"]
    assert ffi["wall_seconds"]["observed_median_status"] == "tie"
    assert ffi["total_cpu_seconds"]["observed_median_status"] == "win"
    assert ffi["total_cpu_seconds"]["candidate_over_python_median_ratio"] == 5 / 52
    assert ffi["max_rss_bytes"]["candidate_over_python_median_ratio"] == 0.8
    standalone = cells["rust-standalone"]["metrics"]
    assert standalone["wall_seconds"]["observed_median_status"] == "win"
    assert standalone["total_cpu_seconds"]["observed_median_status"] == "loss"
    assert standalone["max_rss_bytes"]["observed_median_status"] == "loss"
    assert not cells["rust-ffi"]["all_three_observed_medians_lower"]
    assert not cells["rust-standalone"]["all_three_observed_medians_lower"]
    assert report.median_comparison(1 - 1e-12, 1)["observed_median_win"]


@pytest.mark.parametrize(
    "candidate,reference,status,ratio",
    [
        (0, 0, "tie", None),
        (1, 0, "loss", None),
        (0, 2, "win", 0.0),
    ],
)
def test_zero_cpu_medians_produce_no_nonfinite_json(
    candidate, reference, status, ratio
):
    result = report.median_comparison(candidate, reference)
    assert result["observed_median_status"] == status
    assert result["candidate_over_python_median_ratio"] == ratio
    assert result["ratio_unavailable_reason"] == (
        "zero_python_median" if reference == 0 else None
    )
    report.canonical(result)


@pytest.mark.parametrize("change", ["duplicate", "missing", "nonfinite"])
def test_total_cpu_rejects_incomplete_or_invalid_attempts(change):
    samples, summary = cpu_fixture()
    if change == "duplicate":
        samples.append(samples[0])
    elif change == "missing":
        samples.pop(0)
    else:
        samples[0]["metrics"]["system_cpu_seconds"] = float("inf")
    with pytest.raises(ValueError):
        report.derived_metrics(samples, summary)
