"""Production capture journeys using retained upstream executions and source text."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from invarlock.evidence_pack_contract import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"
PROFILES = json.loads((EXAMPLE / "matrix.json").read_bytes())["profiles"]
EXACT = [
    item["profile_id"]
    for item in PROFILES
    if item["authority"]["mode"] == "deterministic_per_record"
]
ALL = [item["profile_id"] for item in PROFILES]


@pytest.fixture
def capture():
    spec = importlib.util.spec_from_file_location(
        "shortlist_capture", EXAMPLE / "maintained/capture.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(ecosystem):
    artifact = EXAMPLE / "artifacts" / ecosystem
    return {
        "ecosystem": ecosystem,
        "cases_path": EXAMPLE / "cases.json",
        "schedule_path": EXAMPLE / "schedule.json",
        "profile_path": artifact / "profile.json",
        "export_path": artifact / "export.json",
        "raw_output_path": artifact / "upstream-output.json",
        "run_id": "retained-capture",
        "artifact_digest": "sha256:" + "a" * 64,
    }


@pytest.mark.parametrize("ecosystem", EXACT)
def test_retained_native_exports_join_original_case_text(capture, ecosystem):
    run = capture.capture_qualification(**arguments(ecosystem))
    assert [row["input"] for row in run["records"]] == [
        "What is the capital of France?",
        "Return the release label.",
    ]
    assert [row["output"] for row in run["records"]] == ["Paris", "candidate"]
    assert [row["expected"] for row in run["records"]] == ["Paris", "approved"]
    assert run["source"] == {
        "name": capture.profiles()[ecosystem]["upstream"]["name"],
        "version": capture.profiles()[ecosystem]["upstream"]["version"],
    }
    assert all(row["scores"] == {} for row in run["records"])
    assert all(
        row["context"]["evaluator_capture"]["original_profile_id"] == ecosystem
        for row in run["records"]
    )
    assert run["source_digest"] == capture.digest(
        arguments(ecosystem)["export_path"].read_bytes()
    )


@pytest.mark.parametrize("ecosystem", ["mlflow", "garak"])
def test_observation_summaries_do_not_manufacture_case_rows(capture, ecosystem):
    with pytest.raises(ValueError, match="summaries cannot manufacture"):
        capture.capture_qualification(**arguments(ecosystem))


@pytest.mark.parametrize("ecosystem", ALL)
def test_all_ecosystems_accept_explicit_canonical_source_records(
    capture, tmp_path, ecosystem
):
    path = tmp_path / "cases.json"
    path.write_bytes(
        canonical_json_bytes(
            [
                {
                    "id": "attempt-1/output-0",
                    "input": " café\u0000\n",
                    "expected": "",
                    "output": "\n",
                },
                {
                    "id": "attempt-1/output-1",
                    "input": " café\u0000\n",
                    "expected": "ok",
                    "output": "ok",
                },
            ]
        )
    )
    run = capture.capture_records(
        ecosystem=ecosystem,
        records_path=path,
        run_id="captured",
        artifact_digest="sha256:" + "b" * 64,
        source_version="1.0",
    )
    assert [row["id"] for row in run["records"]] == [
        "attempt-1/output-0",
        "attempt-1/output-1",
    ]
    assert run["records"][0]["input"] == " café\u0000\n"
    assert run["records"][0]["output"] == "\n"
    assert run["records"][0]["expected"] == ""
    assert run["source_digest"] == capture.digest(path.read_bytes())
    from invarlock.evaluation_comparison.comparison import compare_runs

    result = compare_runs(run, run, exact_policy())
    assert result["metrics"][0]["baseline_mean"] == 0.5
    assert result["metrics"][0]["scoring_assurance"] == "recomputed"


@pytest.mark.parametrize(
    "kind",
    [
        "input",
        "input_sha256",
        "reference",
        "output",
        "record_id",
        "duplicate",
        "reorder",
        "missing",
    ],
)
def test_original_case_mutations_fail_closed(capture, tmp_path, kind):
    args = arguments("deepeval")
    cases = json.loads(args["cases_path"].read_bytes())
    if kind == "duplicate":
        cases["records"][1] = cases["records"][0]
    elif kind == "reorder":
        cases["records"].reverse()
    elif kind == "missing":
        cases["records"].pop()
    else:
        cases["records"][0][kind] = "changed"
    args["cases_path"] = tmp_path / "cases.json"
    args["cases_path"].write_bytes(canonical_json_bytes(cases))
    with pytest.raises(ValueError):
        capture.capture_qualification(**args)


@pytest.mark.parametrize(
    "binding",
    [
        "profile_sha256",
        "raw_output_sha256",
        "runner_sha256",
        "schedule_sha256",
        "dependency_lock_sha256",
    ],
)
def test_every_export_binding_is_checked(capture, tmp_path, binding):
    args = arguments("deepeval")
    exported = json.loads(args["export_path"].read_bytes())
    exported["bindings"][binding] = "sha256:" + "f" * 64
    args["export_path"] = tmp_path / "export.json"
    args["export_path"].write_bytes(canonical_json_bytes(exported))
    with pytest.raises(ValueError):
        capture.capture_qualification(**args)


@pytest.mark.parametrize("change", ["identity", "order", "score", "count"])
def test_raw_bindings_cannot_hide_native_row_contradictions(capture, tmp_path, change):
    args = arguments("deepeval")
    raw = json.loads(args["raw_output_path"].read_bytes())
    if change == "identity":
        raw["profile_id"] = "other"
    elif change == "order":
        raw["records"].reverse()
    elif change == "count":
        raw["records"].pop()
    else:
        raw["records"][0]["score"] = 0
    args["raw_output_path"] = tmp_path / "raw.json"
    body = canonical_json_bytes(raw)
    args["raw_output_path"].write_bytes(body)
    exported = json.loads(args["export_path"].read_bytes())
    exported["bindings"]["raw_output_sha256"] = capture.digest(body)
    args["export_path"] = tmp_path / "export.json"
    args["export_path"].write_bytes(canonical_json_bytes(exported))
    with pytest.raises(ValueError):
        capture.capture_qualification(**args)


@pytest.mark.parametrize(
    "records",
    [
        [],
        {"summary": {"accuracy": 0.5}},
        [{"entry_type": "eval", "passed": 2}],
        [{"id": "one", "input": "question"}],
    ],
)
def test_missing_original_records_are_not_scores(capture, tmp_path, records):
    path = tmp_path / "cases.json"
    path.write_bytes(canonical_json_bytes(records))
    with pytest.raises(ValueError):
        capture.capture_records(
            ecosystem="garak",
            records_path=path,
            run_id="captured",
            artifact_digest="sha256:" + "b" * 64,
            source_version="1.0",
        )


def test_matrix_and_capture_cli(capture, tmp_path, capsys):
    capture.main(["matrix"])
    matrix = json.loads(capsys.readouterr().out)
    assert len(matrix) == 19
    assert sum(row["retained"] == "observation summary only" for row in matrix) == 2
    assert all("have none" in row["normalized_nll_per_utf8_byte"] for row in matrix)
    args = arguments("inspect-ai")
    destination = tmp_path / "captured.json"
    command = [
        "qualification",
        "--ecosystem",
        "inspect-ai",
        "--run-id",
        args["run_id"],
        "--artifact-digest",
        args["artifact_digest"],
        "--output",
        str(destination),
    ]
    for key in ("cases", "schedule", "profile", "export", "raw_output"):
        command.extend(["--" + key.replace("_", "-"), str(args[key + "_path"])])
    capture.main(command)
    assert json.loads(destination.read_bytes())["records"][0]["output"] == "Paris"
    assert json.loads(capsys.readouterr().out)
    with pytest.raises(FileExistsError):
        capture.main(command)


def exact_policy():
    return {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "correct",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 1,
                "maximum_interval_width": 2,
            }
        ],
        "slices": [],
    }


@pytest.mark.parametrize("ecosystem", EXACT)
def test_retained_capture_reaches_invarlock_exact_scorer(capture, ecosystem):
    from copy import deepcopy

    from invarlock.evaluation_comparison.comparison import compare_runs

    run = capture.capture_qualification(**arguments(ecosystem))
    candidate = deepcopy(run)
    candidate["source_digest"] = "sha256:" + "c" * 64
    candidate["records"][0]["context"]["evaluator_capture"]["export_sha256"] = (
        "sha256:" + "c" * 64
    )
    result = compare_runs(run, candidate, exact_policy())
    assert result["metrics"][0]["baseline_mean"] == 0.5
    assert result["metrics"][0]["scoring_assurance"] == "recomputed"
    capabilities = capture.evaluator_input_capabilities(run)
    assert capabilities["exact_match"]["usable_count"] == 2
    assert capabilities["judge"]["usable_count"] == 2
    assert capabilities["normalized_nll_per_utf8_byte"]["usable_count"] == 0


@pytest.mark.parametrize("ecosystem", ALL)
def test_all_sources_require_typed_likelihood_to_reach_nll(
    capture, tmp_path, ecosystem
):
    from invarlock.evaluation_comparison.comparison import compare_runs
    from invarlock.evaluation_record_contracts.contracts import digest

    source = {
        "name": capture.profiles()[ecosystem]["upstream"]["name"],
        "version": "1.0",
    }
    record = {"id": "one", "input": "prompt", "expected": "é", "output": "é"}
    path = tmp_path / "records.json"
    kwargs = {
        "ecosystem": ecosystem,
        "records_path": path,
        "run_id": "typed",
        "artifact_digest": digest("model"),
        "source_version": source["version"],
    }
    path.write_bytes(canonical_json_bytes([record]))
    missing = capture.capture_records(**kwargs)
    assert capture.evaluator_input_capabilities(missing)[
        "normalized_nll_per_utf8_byte"
    ]["unavailable_ids"] == ["one"]
    record["likelihood"] = {
        "basis": "reference_continuation",
        "logprob_sum": -4.0,
        "token_count": 1,
        "utf8_byte_count": 2,
        "input_digest": digest(record["input"]),
        "reference_digest": digest(record["expected"]),
        "artifact_digest": digest("model"),
        "configuration_digest": digest("config"),
        "tokenizer_digest": digest("tokenizer"),
        "source": source,
    }
    path.write_bytes(canonical_json_bytes([record]))
    run = capture.capture_records(**kwargs)
    assert all(
        value["usable_count"] == 1
        for value in capture.evaluator_input_capabilities(run).values()
    )
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "nll",
                "kind": "normalized_nll_per_utf8_byte",
                "configuration": {
                    "configuration_digest": digest("config"),
                    "baseline_tokenizer_digest": digest("tokenizer"),
                    "subject_tokenizer_digest": digest("tokenizer"),
                },
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "aggregation": "mean",
                "minimum_count": 1,
                "ratio_max": 1.1,
                "maximum_interval_width": 10,
            }
        ],
        "slices": [],
    }
    result = compare_runs(run, run, policy)
    assert result["metrics"][0]["baseline_mean"] == 2.0
    assert compare_runs(missing, missing, policy)["decision"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "cases",
    [
        [],
        {},
        {"records": []},
        {"records": [None, None]},
        {
            "records": [
                {
                    "record_id": "one",
                    "input": {},
                    "reference": "Paris",
                    "output": "Paris",
                },
                {},
            ]
        },
    ],
)
def test_malformed_original_cases_fail_closed(capture, tmp_path, cases):
    args = arguments("deepeval")
    args["cases_path"] = tmp_path / "cases.json"
    args["cases_path"].write_bytes(canonical_json_bytes(cases))
    with pytest.raises(ValueError):
        capture.capture_qualification(**args)


def test_cannot_relabel_package_as_another_ecosystem(capture):
    args = arguments("deepeval")
    args["ecosystem"] = "ragas"
    with pytest.raises(ValueError, match="source package"):
        capture.capture_qualification(**args)


@pytest.mark.parametrize("project", [True, False])
def test_records_cli_preserves_source_and_explicit_projection(
    capture, tmp_path, capsys, project
):
    path, output = tmp_path / "source.json", tmp_path / "capture.json"
    path.write_bytes(
        canonical_json_bytes(
            [
                {
                    "id": "one",
                    "input": {"question": "original?"} if project else "original?",
                    "expected": None,
                    "output": "answer",
                }
            ]
        )
    )
    command = [
        "records",
        "--ecosystem",
        "garak",
        "--source-version",
        "0.15.1",
        "--records",
        str(path),
        "--run-id",
        "reviewed-attempt",
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--output",
        str(output),
    ]
    if project:
        command.extend(["--input-pointer", "/input/question"])
    capture.main(command)
    row = json.loads(output.read_bytes())["records"][0]
    assert row["input"] == "original?"
    if project:
        assert row["context"]["input_projection"]["source"]["input"] == {
            "question": "original?"
        }
    capabilities = json.loads(capsys.readouterr().out)
    assert capabilities["judge"]["usable_count"] == 1
    assert capabilities["exact_match"]["usable_count"] == 0
