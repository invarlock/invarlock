"""Native handoffs bind authentic bytes and explicit cases before policy replay."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from invarlock.pipeline import compare_runs

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"examples/pipeline/{name}.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


@pytest.fixture(params=["inspect", "lm-eval", "promptfoo"])
def bundle(tmp_path, request):
    capture_script = module("native_rehearsal")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    protocol_path = tmp_path / "protocol.json"
    protocol_sha = capture_script.prepare(
        model, protocol_path, "fixture/no-inference", "a" * 40
    )
    protocol = json.loads(protocol_path.read_bytes())
    evaluator = request.param
    capture = tmp_path / "capture"
    capture.mkdir()
    for kind, cases in protocol["cases"].items():
        if evaluator == "lm-eval":
            (capture / f"{kind}-dataset.json").write_bytes(raw(cases))
        for side, template in protocol["prompts"].items():
            rows, calls = [], []
            for i, case in enumerate(cases):
                prompt = template.format(input=case["input"])
                output = "fixture-only wrong answer"
                calls.append(
                    {
                        "prompt": prompt,
                        "output": output,
                        "latency_ms": 10 + i,
                        "error": None,
                    }
                )
                if evaluator == "inspect":
                    rows.append(
                        {
                            "id": case["id"],
                            "input": case["input"],
                            "target": case["expected"],
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": output},
                            ],
                            "output": {"choices": [{"message": {"content": output}}]},
                            "scores": {"match": {"value": "I"}},
                        }
                    )
                elif evaluator == "lm-eval":
                    rows.append(
                        {
                            "doc_id": i,
                            "doc": case,
                            "target": case["expected"],
                            "arguments": [
                                [
                                    prompt,
                                    {
                                        "do_sample": False,
                                        "max_gen_toks": 8,
                                        "until": [],
                                    },
                                ]
                            ],
                            "filtered_resps": [output],
                            "exact_match": 0.0,
                        }
                    )
                else:
                    rows.append(
                        {
                            "testIdx": i,
                            "promptIdx": 0,
                            "prompt": {"raw": prompt},
                            "testCase": {
                                "vars": {"input": case["input"]},
                                "metadata": {"invarlock_expected": case["expected"]},
                                "assert": [
                                    {"type": "equals", "value": case["expected"]}
                                ],
                            },
                            "response": {"output": output},
                            "score": 0,
                            "error": "Assertion failed",
                            "failureReason": 1,
                            "success": False,
                            "gradingResult": {
                                "pass": False,
                                "score": 0,
                                "reason": "Assertion failed",
                            },
                        }
                    )
            stem = f"{kind}-{side}"
            (capture / f"{stem}-calls.json").write_bytes(raw(calls))
            if evaluator == "inspect":
                value = {
                    "version": 2,
                    "status": "success",
                    "samples": rows,
                    "eval": {
                        "task": f"{kind}_{side}",
                        "model": "native_cpu_rehearsal/fixture/no-inference",
                        "packages": {
                            "inspect_ai": capture_script.VERSIONS[evaluator][1]
                        },
                        "model_generate_config": {
                            "max_connections": 1,
                            "max_tokens": 8,
                            "temperature": 0.0,
                        },
                    },
                }
                (capture / f"{stem}.json").write_bytes(raw(value))
            else:
                (capture / f"{stem}.jsonl").write_bytes(
                    b"".join(raw(row) for row in rows)
                )
            if evaluator == "promptfoo":
                (capture / f"{stem}-config.json").write_bytes(
                    raw(
                        {
                            "prompts": [template.replace("{input}", "{{input}}")],
                            "tests": [row["testCase"] for row in rows],
                        }
                    )
                )
    manifest = {
        "format": "invarlock/example-native-capture-v1",
        "evaluator": evaluator,
        "version": capture_script.VERSIONS[evaluator][1],
        "protocol_sha256": protocol_sha,
        "files": {path.name: sha(path.read_bytes()) for path in capture.iterdir()},
    }
    (capture / "capture.json").write_bytes(raw(manifest))
    return {
        "capture": capture,
        "protocol_path": protocol_path,
        "expected_protocol": protocol_sha,
        "expected_capture": sha(raw(manifest)),
        "evaluator": evaluator,
    }


def project(bundle, out):
    return module("native_handoff").project_capture(
        bundle["capture"],
        bundle["protocol_path"],
        bundle["expected_protocol"],
        bundle["expected_capture"],
        out,
    )


def edit_native(bundle, change, *, authorize_new_bytes=False):
    evaluator = bundle["evaluator"]
    suffix = "json" if evaluator == "inspect" else "jsonl"
    path = bundle["capture"] / f"classification-candidate.{suffix}"
    value = (
        json.loads(path.read_bytes())
        if evaluator == "inspect"
        else [json.loads(line) for line in path.read_bytes().splitlines()]
    )
    rows = value["samples"] if evaluator == "inspect" else value
    change(rows, evaluator)
    path.write_bytes(
        raw(value) if evaluator == "inspect" else b"".join(raw(row) for row in rows)
    )
    if authorize_new_bytes:
        manifest_path = bundle["capture"] / "capture.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"][path.name] = sha(path.read_bytes())
        manifest_path.write_bytes(raw(manifest))
        bundle["expected_capture"] = sha(raw(manifest))


def test_native_projection_keeps_failed_quality_as_measurable_results(bundle, tmp_path):
    output = tmp_path / "projected"
    project(bundle, output)
    for kind in ("classification", "extraction", "numeric"):
        directory = output / kind
        baseline, candidate, policy = [
            json.loads((directory / f"{name}.json").read_bytes())
            for name in ("baseline", "candidate", "policy")
        ]
        result = compare_runs(baseline, candidate, policy)
        assert result["decision"] == "regression"
        assert baseline["source_digest"] == bundle["expected_capture"]
        assert baseline["artifact_digest"] != candidate["artifact_digest"]
        assert all(
            row["error"] is None and row["scores"]["latency_ms"] > 0
            for row in candidate["records"]
        )
        assert candidate["records"][0]["output"] == "fixture-only wrong answer"


@pytest.mark.parametrize(
    "fault", ["output", "reference", "input", "order", "duplicate", "score", "prompt"]
)
def test_even_newly_authenticated_bytes_cannot_contradict_declared_records(
    bundle, tmp_path, fault
):
    def change(rows, evaluator):
        row = rows[0]
        if fault == "order":
            rows.reverse()
        elif fault == "duplicate":
            rows[1] = copy.deepcopy(row)
        elif fault == "output":
            if evaluator == "inspect":
                row["output"]["choices"][0]["message"]["content"] = "forged"
            elif evaluator == "lm-eval":
                row["filtered_resps"] = ["forged"]
            else:
                row["response"]["output"] = "forged"
        elif fault == "reference":
            if evaluator == "promptfoo":
                row["testCase"]["metadata"]["invarlock_expected"] = "forged"
            else:
                row["target"] = "forged"
        elif fault == "input":
            if evaluator == "inspect":
                row["input"] = "forged"
            elif evaluator == "lm-eval":
                row["doc"]["input"] = "forged"
            else:
                row["testCase"]["vars"]["input"] = "forged"
        elif fault == "score":
            if evaluator == "inspect":
                row["scores"]["match"]["value"] = "C"
            elif evaluator == "lm-eval":
                row["exact_match"] = 1.0
            else:
                row["score"] = 1
        else:
            if evaluator == "inspect":
                row["messages"][0]["content"] = "forged"
            elif evaluator == "lm-eval":
                row["arguments"][0][0] = "forged"
            else:
                row["prompt"]["raw"] = "forged"

    edit_native(bundle, change, authorize_new_bytes=True)
    with pytest.raises(ValueError):
        project(bundle, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "fault", ["source", "protocol", "capture", "calls", "symlink", "extra"]
)
def test_handoff_rejects_drift_against_independent_inputs(bundle, tmp_path, fault):
    root = bundle["capture"]
    if fault == "source":
        edit_native(bundle, lambda rows, evaluator: rows.pop())
    elif fault == "protocol":
        bundle["protocol_path"].write_bytes(b"{}")
    elif fault == "capture":
        (root / "capture.json").write_bytes(b"{}")
    elif fault == "calls":
        (root / "classification-baseline-calls.json").write_bytes(b"[]")
    elif fault == "extra":
        (root / "unbound.json").write_bytes(b"{}")
    else:
        path = root / "classification-baseline-calls.json"
        saved = tmp_path / "saved.json"
        saved.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(saved)
    with pytest.raises(ValueError):
        project(bundle, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_authenticated_native_configuration_drift_is_rejected(bundle, tmp_path):
    root = bundle["capture"]
    evaluator = bundle["evaluator"]
    name = {
        "inspect": "classification-baseline.json",
        "lm-eval": "classification-dataset.json",
        "promptfoo": "classification-baseline-config.json",
    }[evaluator]
    path = root / name
    value = json.loads(path.read_bytes())
    if evaluator == "inspect":
        value["eval"]["model_generate_config"]["max_tokens"] = 900
    elif evaluator == "lm-eval":
        value[0]["expected"] = "undeclared reference"
    else:
        value["prompts"] = ["undeclared prompt"]
    path.write_bytes(raw(value))
    manifest_path = root / "capture.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][name] = sha(path.read_bytes())
    manifest_path.write_bytes(raw(manifest))
    bundle["expected_capture"] = sha(raw(manifest))
    with pytest.raises(ValueError, match="drift"):
        project(bundle, tmp_path / "rejected")


def test_protocol_is_fixed_before_model_capture(tmp_path):
    capture_script = module("native_rehearsal")
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"fixture, not real model weights")
    protocol = tmp_path / "protocol.json"
    expected = capture_script.prepare(model, protocol, "fixture/no-inference", "a" * 40)
    assert (
        capture_script.load_protocol(protocol, expected, model)["policy"][
            "candidate_minimum"
        ]
        == 0.75
    )
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="local model differs"):
        capture_script.load_protocol(protocol, expected, model)
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        capture_script.capture(
            "inspect",
            model,
            protocol,
            "sha256:" + "0" * 64,
            tmp_path / "not-created",
            "unused",
        )
    assert not (tmp_path / "not-created").exists()
    weights.unlink()
    weights.symlink_to(protocol)
    with pytest.raises(ValueError, match="symbolic links"):
        capture_script.model_files(model)


def test_promptfoo_version_probe_uses_the_isolated_state_directory(
    tmp_path, monkeypatch
):
    import os
    from types import SimpleNamespace

    capture_script = module("native_rehearsal")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    protocol = tmp_path / "protocol.json"
    expected = capture_script.prepare(model, protocol, "fixture/no-inference", "a" * 40)
    output = tmp_path / "capture"
    monkeypatch.setattr(capture_script.os, "environ", dict(os.environ))

    def version_probe(command, **kwargs):
        assert command == ["fixture-promptfoo", "--version"]
        assert capture_script.os.environ["PROMPTFOO_CONFIG_DIR"] == str(
            output / "promptfoo-state"
        )
        assert capture_script.os.environ["PROMPTFOO_DISABLE_UPDATE"] == "1"
        return SimpleNamespace(stdout="unsupported-version")

    monkeypatch.setattr(capture_script.subprocess, "run", version_probe)
    with pytest.raises(ValueError, match="expected promptfoo"):
        capture_script.capture(
            "promptfoo", model, protocol, expected, output, "fixture-promptfoo"
        )
    assert not (output / "capture.json").exists()


def test_capture_budget_is_enforced_before_retaining_all_files(bundle, monkeypatch):
    handoff = module("native_handoff")
    monkeypatch.setattr(handoff, "MAX_CAPTURE_BYTES", 100, raising=False)
    calls = []
    original = handoff.read

    def observe(path, *args, **kwargs):
        calls.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(handoff, "read", observe)
    with pytest.raises(ValueError, match="bound"):
        handoff.inputs(
            bundle["capture"],
            bundle["protocol_path"],
            bundle["expected_protocol"],
            bundle["expected_capture"],
        )
    assert len(calls) == 3  # protocol, manifest, then the first oversized payload


def test_bounded_reader_does_not_follow_a_leaf_swapped_after_lstat(
    tmp_path, monkeypatch
):
    handoff = module("native_handoff")
    selected = tmp_path / "selected.json"
    outside = tmp_path / "outside.json"
    selected.write_bytes(b"approved")
    outside.write_bytes(b"not approved")
    original = Path.lstat

    def raced(path):
        result = original(path)
        if path == selected:
            selected.unlink()
            selected.symlink_to(outside)
        return result

    monkeypatch.setattr(Path, "lstat", raced)
    # The old lstat/open implementation read the swapped target. The descriptor
    # reader either opens the original directly or rejects a link atomically.
    assert handoff.read(selected) == b"approved"


def test_reader_rejects_growth_after_descriptor_stat(tmp_path, monkeypatch):
    handoff = module("native_handoff")
    path = tmp_path / "growing.json"
    path.write_bytes(b"a")
    original = handoff.os.fstat

    def grow(fd):
        before = original(fd)
        with path.open("ab") as stream:
            stream.write(b"extra bytes")
        return before

    monkeypatch.setattr(handoff.os, "fstat", grow)
    with pytest.raises(ValueError, match="grew"):
        handoff.read(path, max_bytes=4)


def test_reader_atomically_rejects_a_link_swapped_at_open(tmp_path, monkeypatch):
    handoff = module("native_handoff")
    selected, outside = tmp_path / "selected", tmp_path / "outside"
    selected.write_bytes(b"approved")
    outside.write_bytes(b"secret")
    original = handoff.os.open

    def swap(path, flags):
        selected.unlink()
        selected.symlink_to(outside)
        return original(path, flags)

    monkeypatch.setattr(handoff.os, "open", swap)
    with pytest.raises(ValueError, match="bounded regular"):
        handoff.read(selected)


def test_reader_fails_closed_without_no_follow_support(tmp_path, monkeypatch):
    handoff = module("native_handoff")
    monkeypatch.delattr(handoff.os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match="no-follow"):
        handoff.read(tmp_path / "unused")


@pytest.mark.parametrize("value", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_manifest_json_has_unambiguous_finite_values(value):
    with pytest.raises(ValueError):
        module("native_handoff").decode(value)


def test_handoff_cli_projects_valid_capture_and_reports_bad_independent_hash(
    bundle, tmp_path, monkeypatch, capsys
):
    import sys

    handoff = module("native_handoff")
    arguments = [
        "native_handoff.py",
        "--capture",
        str(bundle["capture"]),
        "--protocol",
        str(bundle["protocol_path"]),
        "--expected-protocol",
        bundle["expected_protocol"],
        "--expected-capture",
        bundle["expected_capture"],
        "--output",
        str(tmp_path / "projected-cli"),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    handoff.main()
    assert (
        json.loads(capsys.readouterr().out)["expected_capture"]
        == bundle["expected_capture"]
    )
    arguments[-3] = "sha256:" + "0" * 64
    with pytest.raises(SystemExit) as exc:
        handoff.main()
    assert exc.value.code == 2
    assert "Native handoff rejected" in capsys.readouterr().err
