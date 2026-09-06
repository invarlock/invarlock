"""Exercise the parser against retained fields from executed upstream SDKs."""

import copy
from pathlib import Path

import pytest

from invarlock.pipeline import PipelineError, compare_runs, load_run
from invarlock.pipeline.templates import example_project

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "filename,adapter,version",
    [
        ("inspect-0.3.254.json", "inspect-json", "0.3.254"),
        ("lm-eval-0.4.12.jsonl", "lm-eval-samples", "0.4.12"),
        ("promptfoo-0.121.19.jsonl", "promptfoo-jsonl", "0.121.19"),
    ],
)
def test_real_upstream_capture_import_and_release_changes(filename, adapter, version):
    baseline = load_run(
        FIXTURES / filename,
        adapter=adapter,
        source={"name": adapter, "version": version},
        run_id="synthetic-baseline",
        artifact_digest="sha256:" + "a" * 64,
    )
    assert len(baseline["records"]) == 40
    _, _, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    candidate = copy.deepcopy(baseline)
    candidate["artifact_digest"] = "sha256:" + "b" * 64
    for row in candidate["records"]:
        row["context"] = {"rendered_prompt": "New template", "temperature": 0.2}
        row["output"] = " YES "
    assert compare_runs(baseline, candidate, policy)["decision"] == "pass"
    changed_input = copy.deepcopy(candidate)
    changed_input["records"][0]["input"] = "different case"
    with pytest.raises(PipelineError, match="input changed"):
        compare_runs(baseline, changed_input, policy)
    for row in candidate["records"]:
        row["output"] = "wrong"
    assert compare_runs(baseline, candidate, policy)["decision"] == "regression"
