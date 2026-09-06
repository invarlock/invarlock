from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "change",
    ["candidate_version", "runner", "dependency", "authority", "upstream_url", None],
)
def test_current_inspect_binds_complete_profile_before_scoring(
    monkeypatch, tmp_path, change
):
    helpers = load(
        "inspect_binding_helpers",
        ROOT / "tests/examples/test_inspect_literal_execution.py",
    )
    seen = []
    mutable_cases = tmp_path / "cases.json"
    mutable_cases.write_bytes((EXAMPLE / "cases.json").read_bytes())

    async def score(state, target):
        seen.append(state.sample_id)
        # Simulate another process changing the caller's source after execution starts.
        mutable_cases.write_text('{"records":[]}')
        output = state.output.completion
        return SimpleNamespace(
            value="C" if output == target else "I",
            answer=output.strip(),
            explanation=output,
        )

    helpers._upstream(monkeypatch, score)
    runner = load(
        "inspect_bound_runner_test", EXAMPLE / "maintained/inspect_literal_runner.py"
    )
    import matrix
    import runner_support

    definition = matrix.load(EXAMPLE / "maintained/inspect-profile.json")
    profile_path = matrix.write_profile(definition, artifacts=tmp_path / "artifacts")
    profile = matrix.load(profile_path)
    if change == "candidate_version":
        profile["upstream"]["package"]["version"] = "99.0.0"
    elif change == "runner":
        profile["execution"]["runner_sha256"] = "sha256:" + "1" * 64
    elif change == "dependency":
        profile["execution"]["dependency_lock_sha256"] = "sha256:" + "1" * 64
    elif change == "authority":
        profile["authority"]["metric"] = {
            "kind": "numeric_tolerance",
            "absolute_tolerance": 10,
        }
    elif change == "upstream_url":
        profile["upstream"]["project_url"] = "https://example.invalid/candidate"
    profile_path.write_bytes(matrix.canonical_json_bytes(profile))
    args = argparse.Namespace(
        profile=profile_path,
        dependency_lock=EXAMPLE / "locks/inspect-ai.txt",
        cases=mutable_cases,
        schedule=EXAMPLE / "schedule.json",
        export=profile_path.parent / "export.json",
        raw_output=profile_path.parent / "upstream-output.json",
    )
    monkeypatch.setattr(runner, "arguments", lambda: args)
    # A matching candidate installation must not make a changed profile authoritative.
    monkeypatch.setattr(
        runner_support.importlib.metadata,
        "version",
        lambda _: "99.0.0" if change == "candidate_version" else "0.3.254",
    )
    if change:
        with pytest.raises(
            ValueError, match="current definition|dependency declaration"
        ):
            asyncio.run(runner.run())
        assert seen == [] and not args.export.exists()
    else:
        asyncio.run(runner.run())
        assert seen == ["record-1", "record-2"]
        result = matrix.qualify(
            definition, artifacts=profile_path.parent.parent, schedule=args.schedule
        ).as_dict()
        assert result["scores"] == [1.0, 0.0]
        assert json.loads(mutable_cases.read_bytes())["records"] == []
