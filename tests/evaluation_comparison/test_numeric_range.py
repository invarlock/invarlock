"""Finite observations can exceed binary64 range during paired arithmetic."""

import json

import pytest
from typer.testing import CliRunner

from invarlock.cli import app
from invarlock.evaluation_comparison.comparison import compare_runs, make_run
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from tests._evaluation_support import materialize_captured_request


def scalar_request(baseline, subject):
    provenance = {
        "kind": "measurement",
        "source": "numeric-range-test",
        "version": "1",
        "unit": "score",
        "rubric_digest": None,
    }

    def run(values):
        return make_run(
            [
                {
                    "id": str(i),
                    "input": f"question {i}",
                    "expected": "reference",
                    "output": None,
                    "scores": {"value": value},
                }
                for i, value in enumerate(values)
            ],
            source={"name": "numeric-range-test", "version": "1"},
            run_id="test",
            artifact_digest="sha256:" + "a" * 64,
            score_provenance={"value": provenance},
        )

    policy = {
        "format": "invarlock/comparison-policy-v1",
        "slices": [],
        "metrics": [
            {
                "name": "value",
                "kind": "recorded",
                "score_key": "value",
                "accepted_provenance": provenance,
                "configuration": {},
                "unit": "score",
                "direction": "higher",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 0,
                "maximum_interval_width": 2,
            }
        ],
    }
    return run(baseline), run(subject), policy


OVERFLOW_CASES = [
    pytest.param([1e308, -1e308], [-1e308, 1e308], id="opposite-infinite-differences"),
    pytest.param([-1e308, 0], [1e308, 0], id="one-infinite-difference"),
    pytest.param([-1e308], [1e308], id="constant-infinite-difference"),
    pytest.param([0, 0], [1e308, -1e308], id="finite-bootstrap-sum"),
    pytest.param([1e308, 1e308], [1e308, 1e308], id="finite-mean-sum"),
    pytest.param([1e308, 1e308, -1e308], [0, 0, 0], id="cancelling-mean-sum"),
]


@pytest.mark.parametrize("baseline,subject", OVERFLOW_CASES)
def test_public_comparison_normalizes_numeric_range_errors(baseline, subject):
    with pytest.raises(EvaluationRecordsError, match="finite numeric range"):
        compare_runs(*scalar_request(baseline, subject))


@pytest.mark.parametrize("baseline,subject", OVERFLOW_CASES[:4])
def test_cli_serializes_range_failure_without_publishing(tmp_path, baseline, subject):
    request = materialize_captured_request(tmp_path, *scalar_request(baseline, subject))
    result = CliRunner().invoke(app, ["evaluate", str(request), "--unsigned", "--json"])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == "captured"
    assert "finite numeric range" in " ".join(payload["errors"])
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize(
    "baseline,subject,delta,bounds,decision",
    [
        ([1, 2], [2, 4], 1.5, (1, 2), "pass"),
        ([0, 0], [1e307, -1e307], 0, (-1e307, 1e307), "regression"),
        ([-5e307, -5e307], [5e307, 5e307], 1e308, (1e308, 1e308), "pass"),
        ([2**60 + 1], [2**60], 0, (0, 0), "pass"),
        ([2**53, 2**53 + 2], [2**53, 2**53], 0, (-2, 0), "regression"),
    ],
)
def test_finite_results_preserve_binary64_semantics(
    baseline, subject, delta, bounds, decision
):
    result = compare_runs(*scalar_request(baseline, subject))["metrics"][0]
    assert result["delta"] == delta
    assert (result["interval"]["lower"], result["interval"]["upper"]) == bounds
    assert result["decision"] == decision
