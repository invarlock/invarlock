"""Recipient handoff rejects substituted facts using synthetic bounded captures."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from invarlock.engine import case_set_digest, freeze_case_set, load_run, run_digest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/captured-results/harness_model_handoff.py"
CAPTURE_SCRIPT = ROOT / "examples/captured-results/harness_model_comparison.py"


def module(path=SCRIPT):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def material(tmp_path):
    handoff, capture = module(), module(CAPTURE_SCRIPT)
    cases = [
        {
            "id": f"case-{i:04}",
            "input": f"Prompt {i}",
            "expected": " answer",
            "metadata": {
                "source_id": str(i),
                "source_sha256": handoff.object_digest(i),
            },
        }
        for i in range(400)
    ]
    models = {}
    for role, (name, revision) in handoff.MODELS.items():
        files = [
            {"path": path, "byte_size": 1, "sha256": handoff.object_digest(role + path)}
            for path in ("config.json", "model.safetensors", "tokenizer.json")
        ]
        models[role] = {
            "id": name,
            "revision": revision,
            "files": files,
            "artifact_digest": handoff.object_digest(files),
            "tokenizer_digest": handoff.object_digest(files[-1:]),
        }
    configuration = copy.deepcopy(capture.CONFIGURATION)
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "expected_case_set_digest": case_set_digest(freeze_case_set(cases)),
        "metrics": [
            {
                "name": "reference_nll",
                "kind": "normalized_nll_per_utf8_byte",
                "configuration": {
                    "configuration_digest": handoff.object_digest(configuration),
                    "baseline_tokenizer_digest": models["baseline"]["tokenizer_digest"],
                    "subject_tokenizer_digest": models["subject"]["tokenizer_digest"],
                },
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "aggregation": "mean",
                "minimum_count": 400,
                "ratio_max": 1.05,
                "maximum_interval_width": 0.1,
            }
        ],
        "slices": [],
    }
    protocol = {
        "format": "invarlock/harness-model-comparison-v1",
        "models": models,
        "source": handoff.SOURCE,
        "cases": cases,
        "configuration": configuration,
        "policy": policy,
        "dataset": {"name": "synthetic handoff fixture"},
        "source_implementations": capture.UPSTREAM_HASHES,
        "capture_script_sha256": handoff.digest(CAPTURE_SCRIPT.read_bytes()),
    }
    protocol_pin = handoff.object_digest(protocol)
    source_facts = [
        {
            "path": name.replace(".", "/") + ".py",
            "byte_size": 1,
            "sha256": "sha256:" + capture.UPSTREAM_HASHES[name]
            if name in capture.UPSTREAM_HASHES
            else handoff.object_digest(name),
        }
        for name in sorted(handoff.IMPLEMENTATIONS | handoff.RUNTIME_IMPLEMENTATIONS)
    ]
    directory = tmp_path / "capture"
    directory.mkdir()
    pins = {}
    for role in models:
        side = directory / role
        side.mkdir()
        tokenizations = [
            {
                "context_token_ids": [1],
                "continuation_token_ids": [2],
                "joined_token_ids": [1, 2],
                "decoded_context": case["input"],
                "decoded_joined": case["input"] + case["expected"],
                "decoded_continuation": case["expected"],
            }
            for case in cases
        ]
        records = [
            capture.capture_record(
                case,
                (-2.5, True),
                tokens,
                models[role],
                handoff.object_digest(configuration),
                index,
            )
            for index, (case, tokens) in enumerate(
                zip(cases, tokenizations, strict=True)
            )
        ]
        manifest = {
            "format": "invarlock/harness-model-comparison-manifest-v1",
            "protocol_sha256": protocol_pin,
            "role": role,
            "source": handoff.SOURCE,
            "model": models[role],
            "configuration": configuration,
            "configuration_digest": handoff.object_digest(configuration),
            "source_implementations": source_facts,
            "packages": {
                name: "0.4.12" if name == "lm-eval" else "1"
                for name in handoff.PACKAGES
            },
            "capture_script_sha256": protocol["capture_script_sha256"],
        }
        raw = {
            "format": "invarlock/harness-model-comparison-results-v1",
            "source": handoff.SOURCE,
            "protocol_sha256": protocol_pin,
            "role": role,
            "records": records,
            "metadata": {
                "status": "complete",
                "harness_loglikelihood_call_count": 400,
                "case_count": 400,
                "timings": {
                    "wall_seconds": 10,
                    "started_unix_ns": 1,
                    "finished_unix_ns": 2,
                },
            },
        }
        for name, value in (
            ("protocol", protocol),
            ("manifest", manifest),
            ("tokenizations", tokenizations),
            ("raw-results", raw),
        ):
            (side / f"{name}.json").write_bytes(handoff.canonical_json_bytes(value))
        pins[role] = handoff.object_digest(raw)
    return handoff, directory, tmp_path / "project", protocol_pin, pins


def project(material):
    handoff, capture, output, protocol, pins = material
    return handoff.project_capture(
        capture, output, protocol, pins["baseline"], pins["subject"]
    )


def change(material, name, mutate, role="baseline", repin=True):
    handoff, capture, _, _, pins = material
    path = capture / role / f"{name}.json"
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_bytes(handoff.canonical_json_bytes(value))
    if name == "raw-results" and repin:
        pins[role] = handoff.digest(path.read_bytes())


def test_public_sdk_handoff_preserves_frozen_policy_schedule_and_raw_source_pins(
    material,
):
    handoff, capture, output, protocol_pin, pins = material
    result = project(material)
    expected_files = {
        "protocol.json",
        "policy.json",
        "baseline.json",
        "subject.json",
        "request.yaml",
        "anchors.json",
    }
    assert {p.name for p in output.iterdir()} == expected_files
    protocol = json.loads((output / "protocol.json").read_bytes())
    assert json.loads((output / "policy.json").read_bytes()) == protocol["policy"]
    for role in ("baseline", "subject"):
        run = load_run(output / f"{role}.json")
        assert run["source_digest"] == pins[role]
        assert result[f"{role}_run_digest"] == run_digest(run)
        assert [
            {key: row[key] for key in ("id", "input", "expected", "metadata")}
            for row in run["records"]
        ] == protocol["cases"]
        assert run["records"][0]["likelihood"]["logprob_sum"] == -2.5
    assert result["protocol_sha256"] == protocol_pin
    assert result["source_assurance"] == "captured_inputs"
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises((ValueError, OSError)):
        project(material)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert not list(output.parent.glob(".harness-handoff-*"))


@pytest.mark.parametrize(
    "kind",
    [
        "protocol-pin",
        "baseline-pin",
        "subject-pin",
        "invalid-pin",
        "extra-root",
        "missing-role",
        "extra-role",
        "missing-file",
        "linked-file",
        "linked-role",
        "oversized",
        "duplicate-json",
    ],
)
def test_snapshot_and_independent_hash_failures_publish_nothing(
    material, kind, monkeypatch
):
    handoff, capture, output, protocol, pins = material
    if kind == "protocol-pin":
        material = handoff, capture, output, "sha256:" + "f" * 64, pins
    elif kind in {"baseline-pin", "subject-pin"}:
        pins[kind.split("-")[0]] = "sha256:" + "f" * 64
    elif kind == "invalid-pin":
        pins["baseline"] = "A" * 64
    elif kind == "extra-root":
        (capture / "extra").mkdir()
    elif kind == "missing-role":
        (capture / "subject").rename(capture.parent / "moved")
    elif kind == "extra-role":
        (capture / "baseline" / "extra.json").write_text("{}")
    elif kind == "missing-file":
        (capture / "baseline" / "tokenizations.json").unlink()
    elif kind == "linked-file":
        path = capture / "baseline" / "tokenizations.json"
        path.rename(capture.parent / "tokens.json")
        path.symlink_to(capture.parent / "tokens.json")
    elif kind == "linked-role":
        path = capture / "baseline"
        path.rename(capture.parent / "moved")
        path.symlink_to(capture.parent / "moved", target_is_directory=True)
    elif kind == "oversized":
        monkeypatch.setattr(handoff, "MAX_FILE_BYTES", 1)
    else:
        (capture / "baseline" / "manifest.json").write_text('{"format":1,"format":2}')
    with pytest.raises((ValueError, OSError)):
        project(material)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "other"),
        ("source", {"name": "other", "version": "1"}),
        ("protocol_sha256", "sha256:" + "f" * 64),
        ("role", "subject"),
    ],
)
def test_raw_declarations_cannot_drift_even_with_reapproved_raw_hash(
    material, field, value
):
    change(material, "raw-results", lambda raw: raw.__setitem__(field, value))
    with pytest.raises(ValueError, match="raw source or protocol"):
        project(material)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "order",
        "input",
        "reference",
        "metadata",
        "model",
        "generated-output",
        "logprob",
        "positive-logprob",
        "huge-logprob",
        "greedy-bool",
        "greedy-value",
        "token-count",
        "byte-count",
        "config",
        "tokenizer",
        "index-bool",
        "input-bytes",
        "token-row-drift",
        "decode-context",
        "decode-joined",
        "decode-continuation",
        "token-prefix",
        "truncation",
        "token-bool",
    ],
)
def test_record_likelihood_and_contextual_token_boundaries_fail_closed(material, fault):
    def mutate(raw):
        rows = raw["records"]
        row = rows[0]
        harness = row["context"]["harness"]
        if fault == "missing":
            rows.pop()
        elif fault == "extra":
            rows.append(copy.deepcopy(row))
        elif fault == "order":
            rows.reverse()
        elif fault == "input":
            row["input"] += "other"
        elif fault == "reference":
            row["expected"] += "other"
        elif fault == "metadata":
            row["metadata"]["source_id"] = "other"
        elif fault == "model":
            row["context"]["model_id"] = "other"
        elif fault == "generated-output":
            row["output"] = "answer"
        elif fault == "logprob":
            row["likelihood"]["logprob_sum"] = -3
        elif fault == "positive-logprob":
            harness["result"][0] = 1
        elif fault == "huge-logprob":
            harness["result"][0] = -(10**400)
        elif fault == "greedy-bool":
            harness["is_greedy"] = 1
        elif fault == "greedy-value":
            harness["is_greedy"] = False
        elif fault == "token-count":
            row["likelihood"]["token_count"] = 2
        elif fault == "byte-count":
            row["likelihood"]["utf8_byte_count"] = 2
        elif fault == "config":
            row["likelihood"]["configuration_digest"] = "sha256:" + "f" * 64
        elif fault == "tokenizer":
            row["likelihood"]["tokenizer_digest"] = "sha256:" + "f" * 64
        elif fault == "index-bool":
            harness["request_index"] = False
        elif fault == "input-bytes":
            harness["input_utf8_byte_count"] = 1
        else:
            harness["joined_token_ids"] = [3, 4]

    if fault.startswith("decode-") or fault in {
        "token-prefix",
        "truncation",
        "token-bool",
    }:

        def tokens_change(tokens):
            first = tokens[0]
            if fault.startswith("decode-"):
                first["decoded_" + fault.split("-")[1]] = "changed"
            elif fault == "token-prefix":
                first["joined_token_ids"] = [7]
            elif fault == "truncation":
                first["joined_token_ids"] = [1] * 514
            else:
                first["context_token_ids"] = [True]

        change(material, "tokenizations", tokens_change)
    else:
        change(material, "raw-results", mutate)
    with pytest.raises((ValueError, TypeError)):
        project(material)
    assert not material[2].exists()


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "count-bool",
        "calls",
        "wall",
        "start",
        "finish",
        "tokenizations-count",
        "tokenizations-type",
        "format",
        "declarations",
        "packages",
        "package-version",
        "sources-count",
        "source-path",
        "source-hash",
        "source-size",
        "runtime-drift",
    ],
)
def test_manifest_completeness_and_timing_failures(material, fault):
    if fault in {"tokenizations-count", "tokenizations-type"}:
        change(material, "tokenizations", lambda value: value.clear())
    elif fault in {"status", "count-bool", "calls", "wall", "start", "finish"}:

        def modify(raw):
            value = raw["metadata"]
            if fault == "status":
                value["status"] = "incomplete"
            elif fault == "count-bool":
                value["case_count"] = True
            elif fault == "calls":
                value["harness_loglikelihood_call_count"] = 399
            elif fault == "wall":
                value["timings"]["wall_seconds"] = -1
            elif fault == "start":
                value["timings"]["started_unix_ns"] = False
            else:
                value["timings"]["finished_unix_ns"] = 0

        change(material, "raw-results", modify)
    else:

        def modify(value):
            if fault == "format":
                value["format"] = "other"
            elif fault == "declarations":
                value["configuration_digest"] = "sha256:" + "f" * 64
            elif fault == "packages":
                value["packages"]["extra"] = "1"
            elif fault == "package-version":
                value["packages"]["torch"] = ""
            elif fault == "sources-count":
                value["source_implementations"].pop()
            elif fault == "source-path":
                value["source_implementations"][0]["path"] = "../other"
            elif fault == "source-hash":
                value["source_implementations"][0]["sha256"] = "invalid"
            elif fault == "source-size":
                value["source_implementations"][0]["byte_size"] = True
            else:
                value["packages"]["torch"] = "other"

        change(material, "manifest", modify)
    with pytest.raises(ValueError):
        project(material)
    assert not material[2].exists()


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "source",
        "models",
        "model-name",
        "model-path",
        "model-count",
        "model-hash",
        "model-inventory",
        "model-digest",
        "same-models",
        "implementations",
        "dataset",
        "configuration",
        "case-count",
        "case-reference",
        "case-context",
        "case-size",
        "case-metadata",
        "schedule-pin",
        "metric",
        "metric-configuration",
    ],
)
def test_protocol_requires_fixed_models_and_complete_frozen_case_policy(
    material, fault
):
    handoff, capture, _, _, _ = material
    value = json.loads((capture / "baseline" / "protocol.json").read_bytes())
    if fault == "shape":
        value["extra"] = 1
    elif fault == "source":
        value["source"]["version"] = "different"
    elif fault == "models":
        del value["models"]["subject"]
    elif fault == "model-name":
        value["models"]["baseline"]["id"] = "other"
    elif fault == "model-path":
        value["models"]["baseline"]["files"][0]["path"] = "../config.json"
    elif fault == "model-count":
        value["models"]["baseline"]["files"][0]["byte_size"] = False
    elif fault == "model-hash":
        value["models"]["baseline"]["files"][0]["sha256"] = "bad"
    elif fault == "model-inventory":
        value["models"]["baseline"]["files"].reverse()
    elif fault == "model-digest":
        value["models"]["baseline"]["artifact_digest"] = "sha256:" + "f" * 64
    elif fault == "same-models":
        for field in ("files", "artifact_digest", "tokenizer_digest"):
            value["models"]["subject"][field] = value["models"]["baseline"][field]
    elif fault == "implementations":
        value["source_implementations"]["lm_eval.api.model"] = 2
    elif fault == "dataset":
        value["dataset"] = {}
    elif fault == "configuration":
        value["configuration"]["truncation"] = True
    elif fault == "case-count":
        value["cases"].pop()
    elif fault == "case-reference":
        value["cases"][0]["expected"] = ""
    elif fault == "case-context":
        value["cases"][0]["input"] += " "
    elif fault == "case-size":
        value["cases"][0]["input"] = "x" * 65536
    elif fault == "case-metadata":
        value["cases"][0]["metadata"]["source_id"] = "changed"
    elif fault == "schedule-pin":
        value["policy"]["expected_case_set_digest"] = "sha256:" + "f" * 64
    elif fault == "metric":
        value["policy"]["metrics"][0]["minimum_count"] = 399
    else:
        value["policy"]["metrics"][0]["configuration"]["configuration_digest"] = (
            "sha256:" + "f" * 64
        )
    with pytest.raises(ValueError):
        handoff.validate_protocol(value)


def progress(material):
    handoff, capture, _, protocol, _ = material
    side = capture / "baseline"
    rows = json.loads((side / "raw-results.json").read_bytes())["records"]
    directory = side / "progress"
    directory.mkdir()
    for index, row in enumerate(rows):
        (directory / f"{index:06}.attempt.json").write_bytes(
            handoff.canonical_json_bytes(
                {
                    "request_index": index,
                    "case_id": row["id"],
                    "protocol_sha256": protocol,
                }
            )
        )
        (directory / f"{index:06}.json").write_bytes(handoff.canonical_json_bytes(row))
    return directory


def test_complete_admissions_and_results_are_replayed_with_exact_byte_bounds(
    material, monkeypatch
):
    handoff = material[0]
    directory = progress(material)
    original = handoff.read_regular_file_bytes
    admitted = []

    def read(path, **kwargs):
        if path.parent == directory:
            admitted.append(kwargs["max_bytes"])
            assert kwargs["max_bytes"] == path.stat().st_size
        return original(path, **kwargs)

    monkeypatch.setattr(handoff, "read_regular_file_bytes", read)
    project(material)
    assert len(admitted) == 800
    assert sum(admitted) < 2 * handoff.MAX_FILE_BYTES


@pytest.mark.parametrize(
    "fault", ["missing", "admission", "row", "whitespace", "symlink"]
)
def test_changed_progress_is_rejected_without_unbounded_reads(material, fault):
    directory = progress(material)
    if fault == "missing":
        (directory / "000000.json").unlink()
    elif fault == "admission":
        (directory / "000000.attempt.json").write_text("{}\n")
    elif fault == "row":
        (directory / "000000.json").write_text("{}\n")
    elif fault == "whitespace":
        path = directory / "000000.json"
        path.write_bytes(path.read_bytes() + b" " * 1000)
    else:
        (directory / "000000.json").unlink()
        (directory / "000000.json").symlink_to(directory / "000001.json")
    with pytest.raises((ValueError, OSError)):
        project(material)
    assert not material[2].exists()


def test_cli_success_and_actionable_rejection(material, capsys):
    handoff, capture, output, protocol, pins = material
    args = [
        "--capture",
        str(capture),
        "--output",
        str(output),
        "--protocol-sha256",
        protocol,
        "--baseline-sha256",
        pins["baseline"],
        "--subject-sha256",
        pins["subject"],
    ]
    handoff.main(args)
    assert json.loads(capsys.readouterr().out)["source_assurance"] == "captured_inputs"
    with pytest.raises(SystemExit) as failure:
        handoff.main(args)
    assert failure.value.code == 2
    assert "Harness handoff rejected" in capsys.readouterr().err


@pytest.mark.parametrize("output_kind", ["inside", "parent"])
def test_projection_output_cannot_modify_its_capture_tree(material, output_kind):
    handoff, capture, _, protocol, pins = material
    output = capture / "project" if output_kind == "inside" else capture.parent
    with pytest.raises(ValueError, match="remain separate"):
        handoff.project_capture(
            capture, output, protocol, pins["baseline"], pins["subject"]
        )
    assert {path.name for path in capture.iterdir()} == {"baseline", "subject"}


def test_protocol_implementation_pin_cannot_be_replaced_by_manifest(material):
    def mutate(value):
        for fact in value["source_implementations"]:
            if fact["path"] == "lm_eval/models/huggingface.py":
                fact["sha256"] = "sha256:" + "f" * 64

    change(material, "manifest", mutate)
    with pytest.raises(ValueError, match="upstream implementation"):
        project(material)
    assert not material[2].exists()
