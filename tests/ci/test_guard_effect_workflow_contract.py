from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_workflow() -> dict[str, Any]:
    path = Path(".github/workflows/guard-effect-benchmark.yml")
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def test_guard_effect_workflow_is_manual_only_and_does_not_claim_pr_comments() -> None:
    workflow = _load_workflow()

    assert workflow["name"] == "Guard Overhead and Stability Benchmark"
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert all(
        step.get("name") != "Comment PR with results"
        for job in workflow["jobs"].values()
        for step in job["steps"]
    )
    assert all(
        job.get("permissions", {"contents": "read"}) == {"contents": "read"}
        for job in workflow["jobs"].values()
    )


def test_guard_effect_profiles_have_distinct_jobs_and_uploaded_outputs() -> None:
    workflow = _load_workflow()
    ci_job = workflow["jobs"]["guard-effect-benchmark"]
    release_job = workflow["jobs"]["guard-effect-benchmark-release"]

    assert ci_job["if"] == "github.event.inputs.profile != 'release'"
    assert (
        release_job["if"]
        == "github.event_name == 'workflow_dispatch' && github.event.inputs.profile == 'release'"
    )

    ci_run = next(step for step in ci_job["steps"] if step.get("id") == "benchmark-ci")
    assert "--profile ci" in ci_run["run"]
    assert "github.event.inputs.edits" in ci_run["run"]

    release_run = next(
        step
        for step in release_job["steps"]
        if step.get("name") == "Run release overhead and stability benchmark"
    )
    assert "--profile release" in release_run["run"]
    assert "github.event.inputs.edits" in release_run["run"]

    release_upload = next(
        step
        for step in release_job["steps"]
        if step.get("name") == "Upload release benchmark results"
    )
    assert "benchmarks/results/" in release_upload["with"]["path"]
    assert "docs/benchmarks/guard_effect_latest.md" in release_upload["with"]["path"]
    assert "docs/benchmarks/guard_effect_latest.json" in release_upload["with"]["path"]
    assert release_upload["with"]["name"] == "guard-overhead-stability-results-release"


def test_guard_effect_workflow_uses_the_loadable_benchmark_front_door() -> None:
    workflow = _load_workflow()
    run_commands = "\n".join(
        str(step["run"])
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "run" in step
    )

    assert "invarlock.eval.bench" not in run_commands
    assert run_commands.count("python -m invarlock.cli.bench") == 3
    assert "--adapter hf_gpt2" not in run_commands
    assert run_commands.count("--adapter hf_causal") == 2
