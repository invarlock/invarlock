"""Authored fixtures test the handoff consumer; these tests do not run inference."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.engine import case_set_digest, compare_runs, freeze_case_set
from invarlock.evidence_pack_contract import canonical_json_bytes
from tests.cli.test_import_journey import _key

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/captured-results/harness_likelihood_handoff.py"
)


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("harness_handoff", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def material(helper):
    source = {"name": "lm-eval", "version": "0.4.12"}
    files = {"weights.bin": {"sha256": "sha256:" + "a" * 64, "bytes": 1}}
    token_files = {"tokenizer.json": {"sha256": "sha256:" + "b" * 64, "bytes": 2}}
    config = {
        "max_length": 16,
        "backend": "causal",
        "truncation": False,
        "add_bos_token": False,
        "logits_cache": False,
        "result_cache": False,
    }
    cases = [
        {"id": f"case-{i}", "input": f"Question {i}:", "expected": reference}
        for i, reference in enumerate(("x", "é1234567", "café", "yes", "red", "blue"))
    ]
    manifest = {
        "format": "invarlock/harness-likelihood-manifest-v1",
        "source": source,
        "model": {"files": files, "artifact_digest": helper.digest(files)},
        "tokenizer": {
            "files": token_files,
            "tokenizer_digest": helper.digest(token_files),
        },
        "configuration": config,
        "configuration_digest": helper.digest(config),
        "cases": cases,
        "source_implementations": {"test": "sha256:" + "c" * 64},
        "capture_script": "sha256:" + "d" * 64,
        "model_identity": {"id": "authored-fixture", "revision": "e" * 40},
    }
    records = []
    for index, case in enumerate(cases):
        total = -float((1, 90, 5, 3, 3, 4)[index])
        records.append(
            {
                **case,
                "output": None,
                "context": {
                    "harness": {
                        "context_token_ids": [10 + index],
                        "continuation_token_ids": [100 + index],
                        "joined_token_ids": [10 + index, 100 + index],
                        "decoded_continuation": case["expected"],
                        "result": [total, True],
                        "is_greedy": True,
                        "input_utf8_byte_count": len(case["input"].encode()),
                        "request_index": index,
                    }
                },
                "likelihood": {
                    "basis": "reference_continuation",
                    "logprob_sum": total,
                    "token_count": 1,
                    "utf8_byte_count": len(case["expected"].encode()),
                    "input_digest": helper.digest(case["input"]),
                    "reference_digest": helper.digest(case["expected"]),
                    "artifact_digest": manifest["model"]["artifact_digest"],
                    "configuration_digest": manifest["configuration_digest"],
                    "tokenizer_digest": manifest["tokenizer"]["tokenizer_digest"],
                    "source": source,
                },
            }
        )
    raw = {
        "format": "invarlock/harness-likelihood-results-v1",
        "source": source,
        "runs": {
            side: {
                "records": copy.deepcopy(records),
                "metadata": {"elapsed_seconds": 0},
            }
            for side in ("baseline", "subject")
        },
    }
    return manifest, raw


def _capture(helper, material):
    return helper.captured_runs(*material, "sha256:" + "f" * 64)


def _write(tmp_path, material):
    capture = tmp_path / "capture"
    capture.mkdir()
    pins = {}
    for filename, key, value in zip(
        ("manifest.json", "raw-results.json"),
        ("manifest_sha256", "results_sha256"),
        material,
        strict=True,
    ):
        raw = canonical_json_bytes(value)
        (capture / filename).write_bytes(raw)
        pins[key] = "sha256:" + hashlib.sha256(raw).hexdigest()
    return capture, pins


def test_capture_preserves_authored_raw_facts_without_model_execution_claim(
    helper, material
):
    before = copy.deepcopy(material)
    runs = _capture(helper, material)
    assert material == before
    for side, run in runs.items():
        assert run["run_id"] == side
        assert run["source_digest"] == "sha256:" + "f" * 64
        assert run["artifact_digest"] == material[0]["model"]["artifact_digest"]
        assert all(
            row["output"] is None and row["scores"] == {} for row in run["records"]
        )
        assert [row["likelihood"] for row in run["records"]] == [
            row["likelihood"] for row in material[1]["runs"][side]["records"]
        ]
        assert run["records"][2]["likelihood"]["utf8_byte_count"] == 5
        assert all(
            row["context"]["model_id"] == material[0]["model_identity"]["id"]
            and row["context"]["model_revision"]
            == material[0]["model_identity"]["revision"]
            for row in run["records"]
        )
        assert "runtime" not in run


@pytest.mark.parametrize("key", ["manifest_sha256", "results_sha256"])
def test_prepare_requires_both_independent_byte_pins_before_output(
    helper, material, tmp_path, key
):
    capture, pins = _write(tmp_path, material)
    pins[key] = "sha256:" + "0" * 64
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="independently supplied digest"):
        helper.prepare(capture, output, **pins)
    assert not output.exists()


@pytest.mark.parametrize(
    "raw", [b"[]", b"null", b'{"format":1,"format":2}', b"not json"]
)
def test_pinned_json_rejects_nonobject_and_ambiguous_bytes(helper, tmp_path, raw):
    path = tmp_path / "raw.json"
    path.write_bytes(raw)
    pin = "sha256:" + hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        helper.pinned_json(path, pin)


def test_pinned_json_is_bounded_and_rejects_symlinks(helper, tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}\n")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    pin = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="symlink"):
        helper.pinned_json(link, pin)
    monkeypatch.setattr(helper, "MAX_CAPTURE_BYTES", 2)
    with pytest.raises(ValueError, match="limit"):
        helper.pinned_json(path, pin)


@pytest.mark.parametrize(
    "target",
    [
        "manifest-format",
        "raw-format",
        "source",
        "extra-run",
        "model-files",
        "tokenizer-files",
        "configuration",
    ],
)
def test_capture_cross_binds_profiles_and_all_identity_inventories(
    helper, material, target
):
    manifest, raw = material
    if target == "manifest-format":
        manifest["format"] = "unknown"
    elif target == "raw-format":
        raw["format"] = "unknown"
    elif target == "source":
        raw["source"] = {"name": "other", "version": "1"}
    elif target == "extra-run":
        raw["runs"]["third"] = {}
    elif target == "model-files":
        manifest["model"]["files"]["weights.bin"]["bytes"] += 1
    elif target == "tokenizer-files":
        manifest["tokenizer"]["files"]["tokenizer.json"]["bytes"] += 1
    else:
        manifest["configuration"]["max_length"] += 1
    with pytest.raises(ValueError):
        _capture(helper, material)


@pytest.mark.parametrize("cases", [None, {}, [], [{"id": "same"}] * 9])
def test_case_inventory_is_bounded(helper, material, cases):
    material[0]["cases"] = cases
    with pytest.raises(ValueError, match="four to eight"):
        _capture(helper, material)


def test_case_ids_cannot_repeat(helper, material):
    material[0]["cases"][1]["id"] = material[0]["cases"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        _capture(helper, material)


@pytest.mark.parametrize("maximum", [0, 2049, True, 1.0])
def test_context_limit_is_a_bounded_exact_integer(helper, material, maximum):
    material[0]["configuration"]["max_length"] = maximum
    material[0]["configuration_digest"] = helper.digest(material[0]["configuration"])
    with pytest.raises(ValueError, match="context limit"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "change", ["nonlist", "missing", "reordered", "input", "reference"]
)
def test_raw_records_preserve_schedule_and_original_strings(helper, material, change):
    records = material[1]["runs"]["subject"]["records"]
    if change == "nonlist":
        material[1]["runs"]["subject"]["records"] = {"score": 1}
    elif change == "missing":
        records.pop()
    elif change == "reordered":
        records.reverse()
    elif change == "input":
        records[0]["input"] += "changed"
    else:
        records[0]["expected"] = "changed"
    with pytest.raises(ValueError, match="schedule|order changed"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "key,value",
    [
        ("input", ""),
        ("input", "question "),
        ("input", "question\u2003"),
        ("input", 1),
        ("expected", ""),
        ("expected", None),
    ],
)
def test_boundaries_cannot_reinterpret_original_context_or_reference(
    helper, material, key, value
):
    material[0]["cases"][0][key] = value
    material[1]["runs"]["baseline"]["records"][0][key] = value
    with pytest.raises(ValueError, match="outside this profile"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_token_ids", []),
        ("context_token_ids", "tokens"),
        ("context_token_ids", [True]),
        ("continuation_token_ids", [-1]),
        ("continuation_token_ids", [1] * 2049),
        ("continuation_token_ids", [1] * 17),
        ("context_token_ids", [1] * 17),
        ("request_index", 1),
        ("request_index", False),
        ("joined_token_ids", [1, 2]),
        ("joined_token_ids", [True, 100]),
        ("decoded_continuation", " x"),
        ("input_utf8_byte_count", 999),
        ("input_utf8_byte_count", True),
    ],
)
def test_retained_token_boundaries_and_positions_are_exact(
    helper, material, field, value
):
    material[1]["runs"]["baseline"]["records"][0]["context"]["harness"][field] = value
    with pytest.raises(ValueError, match="boundary|truncation|order|reference|UTF-8"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "result",
    [
        None,
        [1],
        [1, True],
        [True, True],
        [float("nan"), True],
        [float("inf"), True],
        [-1, 1],
        [-2, True],
        [-1, False],
    ],
)
def test_api_return_is_a_finite_sum_and_exact_boolean_pair(helper, material, result):
    material[1]["runs"]["baseline"]["records"][0]["context"]["harness"]["result"] = (
        result
    )
    with pytest.raises(ValueError, match="API result"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_count", 2),
        ("artifact_digest", "sha256:" + "0" * 64),
        ("tokenizer_digest", "sha256:" + "0" * 64),
        ("configuration_digest", "sha256:" + "0" * 64),
        ("input_digest", "sha256:" + "0" * 64),
        ("reference_digest", "sha256:" + "0" * 64),
        ("utf8_byte_count", 2),
        ("unknown", 1),
        ("basis", "generated_answer"),
    ],
)
def test_closed_sdk_likelihood_facts_remain_bound_to_original_inputs(
    helper, material, field, value
):
    material[1]["runs"]["subject"]["records"][0]["likelihood"][field] = value
    with pytest.raises(ValueError):
        _capture(helper, material)


def test_is_greedy_does_not_accept_numeric_boolean_coercion(helper, material):
    material[1]["runs"]["baseline"]["records"][0]["context"]["harness"]["is_greedy"] = 1
    with pytest.raises(ValueError, match="API result"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-manifest-field",
        "runs-not-object",
        "row-not-object",
        "context-not-object",
    ],
)
def test_malformed_capture_structure_has_a_public_value_error(helper, material, fault):
    manifest, raw = material
    if fault == "missing-manifest-field":
        del manifest["model"]
    elif fault == "runs-not-object":
        raw["runs"] = None
    elif fault == "row-not-object":
        raw["runs"]["baseline"]["records"][0] = None
    else:
        raw["runs"]["baseline"]["records"][0]["context"] = None
    with pytest.raises(ValueError):
        _capture(helper, material)


def test_invalid_capture_never_creates_output(helper, material, tmp_path):
    material[1]["runs"]["subject"]["records"][0]["likelihood"]["basis"] = (
        "generated_answer"
    )
    capture, pins = _write(tmp_path, material)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        helper.prepare(capture, output, **pins)
    assert not output.exists()


def test_prepared_policy_is_fixed_mean_ratio_and_pins_are_prepublication(
    helper, material, tmp_path
):
    capture, pins = _write(tmp_path, material)
    output = tmp_path / "prepared"
    anchors = helper.prepare(capture, output, **pins)
    policy = json.loads((output / "policy.json").read_bytes())
    assert policy["expected_case_set_digest"] == case_set_digest(
        freeze_case_set([{**case, "metadata": {}} for case in material[0]["cases"]])
    )
    metric = policy["metrics"][0]
    assert metric["ratio_max"] == 1.000001
    assert metric["maximum_interval_width"] == 0.000001
    assert metric["minimum_count"] == 6
    assert metric["aggregation"] == "mean"
    assert (
        metric["configuration"]["baseline_tokenizer_digest"]
        == metric["configuration"]["subject_tokenizer_digest"]
    )
    assert not (output / "evidence").exists()
    assert anchors == json.loads((output / "anchors.json").read_bytes())
    second = tmp_path / "recipient-preparation"
    assert helper.prepare(capture, second, **pins) == anchors
    runs = {
        side: json.loads((output / f"{side}.json").read_bytes())
        for side in ("baseline", "subject")
    }
    assert (
        compare_runs(runs["baseline"], runs["subject"], policy)["metrics"][0]["ratio"]
        == 1
    )
    with pytest.raises(FileExistsError):
        helper.prepare(capture, output, **pins)


def test_api_sums_are_normalized_per_case_not_pooled_or_token_normalized(
    helper, material
):
    # Equal extra rows retain the unequal byte lengths needed to expose pooling.
    for side in ("baseline", "subject"):
        for index, row in enumerate(material[1]["runs"][side]["records"]):
            value = (
                (1, 90, 5, 3, 3, 4)[index]
                if side == "baseline"
                else (2, 45, 5, 3, 3, 4)[index]
            )
            row["likelihood"]["logprob_sum"] = -value
            row["context"]["harness"]["result"][0] = -value
    runs = _capture(helper, material)
    from tests.evaluation_comparison.test_likelihood import policy as nll_policy

    policy = nll_policy()
    policy["metrics"][0]["configuration"] = {
        "configuration_digest": material[0]["configuration_digest"],
        "baseline_tokenizer_digest": material[0]["tokenizer"]["tokenizer_digest"],
        "subject_tokenizer_digest": material[0]["tokenizer"]["tokenizer_digest"],
    }
    metric = compare_runs(runs["baseline"], runs["subject"], policy)["metrics"][0]
    assert metric["baseline_mean"] == 15 / 6
    assert metric["subject_mean"] == 11 / 6
    assert metric["ratio"] == pytest.approx(11 / 15)
    assert metric["ratio"] != pytest.approx(62 / 106)


def test_cli_prepare_and_signed_evaluate_verify_preserve_independent_anchors(
    helper, material, tmp_path, monkeypatch, capsys
):
    capture, pins = _write(tmp_path, material)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--capture",
            str(capture),
            "--output",
            str(output),
            "--manifest-sha256",
            pins["manifest_sha256"],
            "--results-sha256",
            pins["results_sha256"],
        ],
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    anchors = json.loads(capsys.readouterr().out)
    key, signer = _key(tmp_path / "signer.pem")
    result = CliRunner().invoke(
        app,
        ["evaluate", str(output / "request.yaml"), "--signing-key", str(key), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["decision"] == "pass"
    from invarlock.captured_verification import verify_captured_evidence

    verified = verify_captured_evidence(
        output / "evidence",
        policy_path=output / "policy.json",
        expected_baseline_run=anchors["baseline_run_digest"],
        expected_subject_run=anchors["subject_run_digest"],
        expected_request_digest=anchors["request_digest"],
        expected_signer=signer,
        receipt_path=tmp_path / "receipt.json",
        verifier_signing_key_path=key,
        verifier_identity="recipient",
    )
    assert verified["ok"] and verified["replay_status"] == "completed"
    rendered = CliRunner().invoke(app, ["report", str(output / "evidence"), "--json"])
    assert rendered.exit_code == 0, rendered.output
    for option, filename in (("--html", "report.html"), ("--markdown", "report.md")):
        report_path = tmp_path / filename
        result = CliRunner().invoke(
            app, ["report", str(output / "evidence"), option, str(report_path)]
        )
        assert result.exit_code == 0, result.output
        report_text = report_path.read_text()
        assert material[0]["model_identity"]["id"] in report_text
        assert material[0]["model_identity"]["revision"] in report_text


@pytest.mark.parametrize(
    "source",
    [{"name": "other", "version": "0.4.12"}, {"name": "lm-eval", "version": "0.4.13"}],
)
def test_only_the_reviewed_source_profile_is_supported(helper, material, source):
    material[0]["source"] = source
    material[1]["source"] = source
    with pytest.raises(ValueError, match="capture profile"):
        _capture(helper, material)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "seq2seq"),
        ("truncation", True),
        ("add_bos_token", True),
        ("logits_cache", True),
        ("result_cache", True),
        ("truncation", 0),
    ],
)
def test_configuration_cannot_reinterpret_continuations_or_use_cached_results(
    helper, material, field, value
):
    material[0]["configuration"][field] = value
    material[0]["configuration_digest"] = helper.digest(material[0]["configuration"])
    with pytest.raises(ValueError, match="configuration"):
        _capture(helper, material)
