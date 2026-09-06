"""Cross-check arithmetic, uncertainty, missingness and executable CI behavior."""

import copy
import json

import pytest
from typer.testing import CliRunner

from invarlock.pipeline import PipelineError, compare_runs
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import digest
from invarlock.pipeline.templates import example_project


def test_scalar_unit_conversion_preserves_decision_and_scales_interval():
    base, candidate, policy = example_project("judge")
    for i, row in enumerate(candidate["records"]):
        row["scores"]["latency_ms"] += i % 3
    original = compare_runs(base, candidate, policy)
    p2, b2, c2 = copy.deepcopy((policy, base, candidate))
    p2["metrics"][1]["unit"] = "seconds"
    p2["metrics"][1]["accepted_provenance"]["unit"] = "seconds"
    for key in ("maximum_regression", "maximum_interval_width", "candidate_maximum"):
        p2["metrics"][1][key] /= 1000
    for run in (b2, c2):
        run["score_provenance"]["latency_ms"]["unit"] = "seconds"
        for row in run["records"]:
            row["scores"]["latency_ms"] /= 1000
    converted = compare_runs(b2, c2, p2)
    assert original["decision"] == converted["decision"]
    for key in ("lower", "upper"):
        assert converted["metrics"][1]["interval"][key] == pytest.approx(
            original["metrics"][1]["interval"][key] / 1000
        )


def test_swapping_paired_runs_reverses_delta_and_interval():
    base, candidate, policy = example_project("judge")
    for i, row in enumerate(candidate["records"]):
        row["scores"]["quality"] += (i % 5 - 2) / 100
    forward = compare_runs(base, candidate, policy)["metrics"][0]
    reverse = compare_runs(candidate, base, policy)["metrics"][0]
    assert forward["delta"] == pytest.approx(-reverse["delta"])
    assert forward["interval"]["lower"] == pytest.approx(-reverse["interval"]["upper"])
    assert forward["interval"]["upper"] == pytest.approx(-reverse["interval"]["lower"])


def test_latency_regression_rejects_candidate_below_absolute_ceiling():
    base, candidate, policy = example_project("judge")
    for row in candidate["records"]:
        row["scores"]["latency_ms"] = 150.0
    result = compare_runs(base, candidate, policy)
    latency = result["metrics"][1]
    assert latency["candidate_mean"] < policy["metrics"][1]["candidate_maximum"]
    assert latency["decision"] == result["decision"] == "regression"
    assert latency["reasons"] == ["upper interval bound exceeds allowed regression"]


def test_wide_interval_cannot_pass_an_acceptable_mean():
    base, candidate, policy = example_project("judge")
    policy["metrics"][0]["maximum_interval_width"] = 0.001
    for i, row in enumerate(candidate["records"]):
        row["scores"]["quality"] += 0.05 if i % 2 else -0.05
    result = compare_runs(base, candidate, policy)
    quality = result["metrics"][0]
    assert quality["delta"] == pytest.approx(0)
    assert quality["decision"] == result["decision"] == "insufficient_evidence"
    assert quality["reasons"] == ["interval is too wide"]


def test_invalid_reference_identifies_metric_and_record():
    base, candidate, policy = example_project("extraction")
    for run in (base, candidate):
        del run["records"][0]["expected"]["currency"]
    with pytest.raises(PipelineError, match="metric quality, record case-0: reference"):
        compare_runs(base, candidate, policy)


@pytest.mark.parametrize(
    "mutation", ["unit", "nan", "boolean", "empty_slice", "missing_score"]
)
def test_no_false_pass_from_invalid_or_incomplete_measurements(mutation):
    base, candidate, policy = example_project("judge")
    if mutation == "unit":
        candidate["score_provenance"]["quality"]["unit"] = "percent"
    elif mutation in ("nan", "boolean"):
        candidate["records"][0]["scores"]["quality"] = (
            float("nan") if mutation == "nan" else True
        )
    elif mutation == "empty_slice":
        policy["slices"][0]["where"] = {"category": "nonexistent"}
    else:
        del candidate["records"][0]["scores"]["quality"]
    if mutation in ("unit", "nan", "boolean"):
        with pytest.raises(PipelineError):
            compare_runs(base, candidate, policy)
    else:
        assert (
            compare_runs(base, candidate, policy)["decision"] == "insufficient_evidence"
        )


def test_real_cli_sign_verify_and_all_decision_exit_codes(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    assert (
        runner.invoke(app, ["init", str(project), "--example", "judge"]).exit_code == 0
    )
    key_directory = tmp_path / "keys"
    key = key_directory / "private.pem"
    generated = runner.invoke(app, ["keygen", str(key_directory)])
    assert generated.exit_code == 0, generated.output
    assert key.stat().st_mode & 0o077 == 0
    assert runner.invoke(app, ["keygen", str(key_directory)]).exit_code == 2
    compared = runner.invoke(
        app,
        [
            "compare",
            str(project / "pipeline.json"),
            "--output",
            str(project / "signed"),
            "--signing-key",
            str(key),
        ],
    )
    assert compared.exit_code == 0, compared.output
    base = json.loads((project / "baseline.json").read_text())
    candidate = json.loads((project / "candidate.json").read_text())
    args = [
        "verify",
        str(project / "signed/evidence.json"),
        "--public-key",
        str(key_directory / "public.pem"),
        "--policy",
        str(project / "policy.json"),
        "--expected-baseline",
        digest(base),
        "--expected-candidate",
        digest(candidate),
    ]
    verified = runner.invoke(app, args)
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["authenticated"] is True
    args[-1] = "sha256:" + "a" * 64
    assert runner.invoke(app, args).exit_code == 2
    for row in candidate["records"]:
        row["scores"]["quality"] = 0.1
    (project / "candidate.json").write_text(json.dumps(candidate))
    regressed = runner.invoke(
        app,
        [
            "compare",
            str(project / "pipeline.json"),
            "--output",
            str(project / "regression"),
        ],
    )
    assert regressed.exit_code == 1, regressed.output
    assert json.loads(regressed.stdout)["decision"] == "regression"
