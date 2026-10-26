"""Retained runtime judge profiles and semantic diagnostics against Python."""

import copy
import json
import os
import subprocess
from pathlib import Path

import tests.judge_measurements.test_runtime_provider_judge as fixture
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    validate_measurements,
)
from invarlock.judge_measurements.runtime_provider import (
    RuntimeProviderJudgeOptions,
    collect_runtime_provider,
)


def test_retained_runtime_source_replay(tmp_path, monkeypatch):
    fixture._strict_boundary.__wrapped__(monkeypatch)
    root = tmp_path.resolve()
    cases = []
    for profile in ["canonical", "chatml", "cancelled"]:
        plan = fixture._plan(
            runtime_format="chatml-v1" if profile == "chatml" else None
        )
        provider = (
            fixture._Provider(record_status="error")
            if profile == "cancelled"
            else fixture._Provider()
        )
        m = collect_runtime_provider(
            plan,
            provider=provider,
            spec=fixture._spec(),
            resources=fixture._resources(root / profile),
            baseline_run=fixture._load("baseline_run.json"),
            subject_run=fixture._load("subject_run.json"),
            options=RuntimeProviderJudgeOptions(),
        )
        cases.append((profile, m, plan))
        if profile == "canonical":
            base = copy.deepcopy(m)
            baseplan = copy.deepcopy(plan)
    mutations = {
        "empty_spec": lambda s: s.update(runtime_spec={}),
        "empty_artifact": lambda s: s.update(artifact_identity={}),
        "unsupported_provider": lambda s: s["runtime_spec"].update(
            provider_name="custom_runtime"
        ),
        "third_party": lambda s: s["provider_receipt"]["plugin"].update(
            distribution="custom"
        ),
        "missing_image": lambda s: s["provider_receipt"].update(
            outer_image_digest=None
        ),
        "metric_missing": lambda s: s["provider_receipt"]["capabilities"].update(
            metrics=["normalized_nll_per_utf8_byte"]
        ),
        "bad_plan_hash": lambda s: s["scoring_observation"].update(
            schedule_sha256="0" * 64
        ),
        "bad_aggregate": lambda s: s["scoring_observation"].update(
            aggregate_source_sha256="0" * 64
        ),
        "batch_bool": lambda s: s["runtime_spec"]["settings"].update(batch_size=True),
        "batch_float": lambda s: s["runtime_spec"]["settings"].update(batch_size=1.0),
        "no_attempt": lambda s: s["trials"][0].update(attempts=[]),
        "no_request": lambda s: s["trials"][0]["attempts"][0].update(request=None),
        "forged_output": lambda s: s["trials"][0]["attempts"][0]["response"].update(
            text="different"
        ),
    }
    for name, value in [
        ("batch_size", 2),
        ("context_length", 1048577),
        ("timeout_seconds", 604801),
        ("max_output_tokens", 127),
    ]:

        def mutate(s, name=name, value=value):
            s["runtime_spec"]["settings"][name] = value
            s["provider_receipt"]["execution_settings"][name] = value

        mutations[name] = mutate
    for label, mutate in mutations.items():
        m = copy.deepcopy(base)
        for source in m["sources"]:
            s = json.loads(source["content"])
            mutate(s)
            # Preserve dependent provider observation byte identity so each fixture isolates
            # the semantic boundary rather than failing first at a changed byte digest.
            s["provider_receipt"]["scoring_observation_sha256"] = fixture._sha256(
                canonical_payload(s["scoring_observation"])
            )
            raw = canonical_payload(s)
            source.update(
                content=raw.decode(), byte_size=len(raw), sha256=fixture._sha256(raw)
            )
        cases.append((label, m, baseplan))
    requests = [
        {
            "operation": "validate_measurements",
            "measurements": m,
            "plan": p,
            "baseline_run": fixture._load("baseline_run.json"),
            "subject_run": fixture._load("subject_run.json"),
        }
        for label, m, p in cases
    ]
    target = Path(
        os.environ.get(
            "CARGO_TARGET_DIR", Path(__file__).resolve().parents[2] / "native/target"
        )
    )
    binary = Path(
        os.environ.get("NATIVE_ANALYSIS_BINARY", target / "debug/examples/analysis")
    )
    assert binary.is_file(), f"Build native core analysis example: {binary}"
    result = subprocess.run(
        [str(binary)],
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    for (label, m, p), out in zip(cases, outputs, strict=True):
        try:
            validate_measurements(
                m,
                p,
                baseline_run=fixture._load("baseline_run.json"),
                subject_run=fixture._load("subject_run.json"),
            )
            expected = None
        except ValueError as e:
            expected = str(e)
        assert out.get("error") == expected, (label, expected, out)
        if expected is None:
            assert out == {"ok": {"valid": True}}
