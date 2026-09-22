"""Synthetic capture ledgers test offline recipient preservation, never live execution."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location(
    "live_recipient_example", HERE / "recipient.py"
)
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)
COMMON = LIVE.common


def fixture(tmp_path, evaluator="inspect-ai", *, subject_logprob=-1.0, metadata=None):
    """Explicit synthetic observations for the recipient's file contract tests."""
    versions = {
        "inspect-ai": "0.3.254",
        "langfuse": "4.14.1",
        "lm-evaluation-harness": "0.4.12",
    }
    configuration = {"max_new_tokens": 32, "max_length": 512}
    models = {
        role: {
            "artifact_digest": "sha256:" + char * 64,
            "tokenizer_digest": "sha256:" + "c" * 64,
        }
        for role, char in zip(LIVE.ROLES, "ab", strict=True)
    }
    cases = [
        {
            "id": str(i),
            "input": f"Synthetic question {i}",
            "expected": "A",
            "metadata": {"family": "synthetic-test", **(metadata or {})},
        }
        for i in range(2)
    ]
    policies = {}
    for scorer, metric in (
        (
            "exact_match",
            {
                "kind": "exact_match",
                "direction": "higher",
                "unit": "score",
                "configuration": {},
                "maximum_regression": 0.02,
            },
        ),
        (
            "normalized_nll",
            {
                "kind": "normalized_nll_per_utf8_byte",
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "configuration": {
                    "configuration_digest": COMMON.digest(configuration),
                    "baseline_tokenizer_digest": models["baseline"]["tokenizer_digest"],
                    "subject_tokenizer_digest": models["subject"]["tokenizer_digest"],
                },
                "ratio_max": 1.05,
            },
        ),
    ):
        policies[scorer] = {
            "format": "invarlock/comparison-policy-v1",
            "metrics": [
                {
                    "name": scorer,
                    "aggregation": "mean",
                    "minimum_count": 2,
                    "maximum_interval_width": 0.1,
                    **metric,
                }
            ],
            "slices": [],
        }
    protocol = {
        "format": "invarlock/live-evaluator-protocol-v1",
        "cases": cases,
        "models": models,
        "configuration": configuration,
        "evaluators": [evaluator],
        "versions": {evaluator: versions[evaluator]},
        "acceptance": policies,
    }
    protocol_path = tmp_path / "protocol.json"
    COMMON.write(protocol_path, protocol)
    for role in LIVE.ROLES:
        capture = tmp_path / role
        (capture / "tasks").mkdir(parents=True)
        role_protocol = {**protocol, "role": role}
        COMMON.write(capture / "protocol.json", role_protocol)
        samples = []
        for case in cases:
            request = {
                "evaluator": evaluator,
                "case_id": case["id"],
                "protocol_digest": COMMON.digest(role_protocol),
            }
            measured = -1.0 if role == "baseline" else subject_logprob
            execution = {
                "request": request,
                "protocol_digest": request["protocol_digest"],
                "model": models[role],
                "configuration": configuration,
                "source": {"name": "lm-eval", "version": "0.4.12"},
                "generation_parameters": {
                    "until": [],
                    "max_gen_toks": 32,
                    "do_sample": False,
                },
                "tokenization": {
                    "context_token_ids": [1],
                    "continuation_token_ids": [2],
                    "joined_token_ids": [1, 2],
                    "decoded_context": case["input"],
                    "decoded_continuation": case["expected"],
                    "decoded_joined": case["input"] + case["expected"],
                },
                "generation_result": ["A"],
                "likelihood_result": [measured, False],
            }
            facts = {
                "basis": "reference_continuation",
                "logprob_sum": measured,
                "token_count": 1,
                "utf8_byte_count": 1,
                "input_digest": COMMON.digest(case["input"]),
                "reference_digest": COMMON.digest(case["expected"]),
                "artifact_digest": models[role]["artifact_digest"],
                "source": execution["source"],
                "configuration_digest": COMMON.digest(configuration),
                "tokenizer_digest": models[role]["tokenizer_digest"],
            }
            result = {
                "output": "A",
                "metadata": {
                    "invarlock_model_execution": execution,
                    "invarlock_likelihood": facts,
                },
            }
            stem = COMMON.digest(request).removeprefix("sha256:")
            COMMON.write(capture / "tasks" / (stem + ".request.json"), request)
            COMMON.write(
                capture / "tasks" / (stem + ".response.json"),
                {"request": request, "result": result},
            )
            metadata = {
                **case["metadata"],
                **LIVE.bindings.bind_result(
                    result, case, evaluator, versions[evaluator]
                )["metadata"],
            }
            if evaluator == "inspect-ai":
                samples.append(
                    {
                        "id": case["id"],
                        "input": case["input"],
                        "target": "A",
                        "output": {"choices": [{"message": {"content": "A"}}]},
                        "metadata": metadata,
                    }
                )
            elif evaluator == "langfuse":
                samples.append(
                    {
                        "item": {
                            "id": case["id"],
                            "input": case["input"],
                            "expected_output": "A",
                            "metadata": metadata,
                        },
                        "output": "A",
                        "evaluations": [],
                        "trace_id": "trace-" + case["id"],
                        "dataset_run_id": None,
                    }
                )
            else:
                samples.append(
                    {
                        "doc_id": case["id"],
                        "doc": copy.deepcopy(case),
                        "target": "A",
                        "arguments": [[case["input"], {}]],
                        "filtered_resps": ["A"],
                        "metadata": metadata,
                    }
                )
        native = (
            {"version": 2, "status": "success", "samples": samples}
            if evaluator == "inspect-ai"
            else {
                "name": "synthetic",
                "run_name": role,
                "experiment_id": role,
                "run_evaluations": [],
                "item_results": samples,
            }
            if evaluator == "langfuse"
            else samples
        )
        COMMON.write(capture / "native.json", native)
        manifest = {
            "format": "invarlock/live-evaluator-capture-v1",
            "status": "captured",
            "evaluator": evaluator,
            "version": versions[evaluator],
            "role": role,
            "protocol_digest": COMMON.digest(role_protocol),
            "model": models[role],
            "case_count": len(cases),
            "native_sha256": "sha256:"
            + hashlib.sha256((capture / "native.json").read_bytes()).hexdigest(),
        }
        COMMON.write(capture / "capture.json", manifest)
    return protocol, [
        protocol_path,
        COMMON.digest(protocol),
        tmp_path / "baseline",
        tmp_path / "subject",
        evaluator,
    ]


@pytest.mark.parametrize(
    "evaluator", ["inspect-ai", "langfuse", "lm-evaluation-harness"]
)
@pytest.mark.parametrize("route", ["envelope", "native-json"])
def test_actual_import_preserves_ledger_and_likelihood(tmp_path, evaluator, route):
    protocol, args = fixture(tmp_path, evaluator)
    output = tmp_path / "recipient"
    retained, runs = LIVE.prepare(*args, route, output)
    assert retained == protocol
    for role, run in zip(LIVE.ROLES, runs, strict=True):
        assert (output / "captures" / role / "native.json").read_bytes() == (
            tmp_path / role / "native.json"
        ).read_bytes()
        assert run["records"][0]["output"] == "A"
        assert run["records"][0]["input"] == protocol["cases"][0]["input"]
        assert run["records"][0]["likelihood"]["logprob_sum"] == -1.0
        if evaluator == "langfuse":
            assert run["run_id"] == role
    assert (output / "exact_match/signer.pem").stat().st_mode & 0o777 == 0o600
    assert (output / "exact_match/signer.pem").read_bytes() != (
        output / "exact_match/verifier.pem"
    ).read_bytes()
    assert not (output / "exact_match/evidence").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "manifest",
        "native",
        "request",
        "execution",
        "likelihood",
        "extra",
        "missing",
        "response",
    ],
)
def test_altered_capture_ledger_rejected_before_output(tmp_path, mutation):
    protocol, args = fixture(tmp_path)
    if mutation == "protocol":
        args[1] = "sha256:" + "0" * 64
    elif mutation == "manifest":
        path = tmp_path / "subject/capture.json"
        value = COMMON.read(path)
        value["role"] = "baseline"
        path.write_bytes(COMMON.encoded(value))
    elif mutation == "native":
        with (tmp_path / "subject/native.json").open("ab") as stream:
            stream.write(b" ")
    elif mutation == "extra":
        COMMON.write(tmp_path / "subject/tasks/extra.json", {})
    elif mutation == "missing":
        next((tmp_path / "subject/tasks").glob("*.response.json")).unlink()
    else:
        suffix = "request" if mutation == "request" else "response"
        path = next((tmp_path / "subject/tasks").glob(f"*.{suffix}.json"))
        value = COMMON.read(path)
        if mutation == "request":
            value["case_id"] = "other"
        elif mutation == "execution":
            value["result"]["metadata"]["invarlock_model_execution"][
                "configuration"
            ] = {}
        elif mutation == "likelihood":
            value["result"]["metadata"]["invarlock_model_execution"][
                "likelihood_result"
            ][0] = -9.0
        else:
            value["request"]["case_id"] = "other"
        path.write_bytes(COMMON.encoded(value))
    with pytest.raises((ValueError, OSError)):
        LIVE.prepare(*args, "native-json", tmp_path / "recipient")
    assert not (tmp_path / "recipient").exists()


@pytest.mark.parametrize("mutation", ["input", "metadata"])
def test_self_consistent_native_hash_cannot_erase_original_observations(
    tmp_path, mutation
):
    _, args = fixture(tmp_path)
    path = tmp_path / "subject/native.json"
    native = COMMON.read(path)
    sample = native["samples"][0]
    if mutation == "input":
        sample["input"] = "changed task text"
    else:
        sample["metadata"].pop("invarlock_model_execution")
    path.write_bytes(COMMON.encoded(native))
    manifest_path = tmp_path / "subject/capture.json"
    manifest = COMMON.read(manifest_path)
    manifest["native_sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    manifest_path.write_bytes(COMMON.encoded(manifest))
    with pytest.raises(ValueError):
        LIVE.prepare(*args, "native-json", tmp_path / "recipient")
    assert not (tmp_path / "recipient/exact_match").exists()


def test_real_cli_replay_accepts_an_authentic_policy_regression(tmp_path, monkeypatch):
    from invarlock.cli.app import app

    _, args = fixture(tmp_path, subject_logprob=-4.0)
    output = tmp_path / "recipient"
    LIVE.prepare(*args, "native-json", output)

    def local_command(directory, *arguments, allowed=(0,)):
        # Executes the real CLI in the source test environment; installed isolation
        # is independently required by recipient.main, never claimed by this test.
        with monkeypatch.context() as context:
            context.chdir(directory)
            result = CliRunner().invoke(app, list(arguments))
        assert result.exit_code in allowed, result.output
        return json.loads(result.stdout)

    monkeypatch.setattr(LIVE, "command", local_command)
    result = LIVE.journey(output)
    assert result["exact_match"]["decision"] == "regression"
    assert result["normalized_nll"]["decision"] == "regression"
    for scorer in LIVE.SCORERS:
        for name in result[scorer]["reports"]:
            assert (output / scorer / name).stat().st_size > 0
        assert (output / scorer / "verification.receipt.json").exists()


@pytest.mark.parametrize("fault", [None, "source", "sdk", "credentials", "isolation"])
def test_recipient_environment_requires_installed_offline_core(monkeypatch, fault):
    from types import SimpleNamespace

    import invarlock

    monkeypatch.setattr(
        invarlock,
        "__file__",
        "/isolated/lib/invarlock/__init__.py"
        if fault != "source"
        else "/source/invarlock/__init__.py",
    )
    monkeypatch.setattr(LIVE, "get_path", lambda _: "/isolated/lib")
    monkeypatch.setattr(
        LIVE,
        "sys",
        SimpleNamespace(flags=SimpleNamespace(isolated=fault != "isolation")),
    )
    monkeypatch.setattr(
        LIVE,
        "os",
        SimpleNamespace(
            environ={"OPENAI_API_KEY": "test-only"} if fault == "credentials" else {}
        ),
    )

    def find_spec(name):
        if name == "azure.ai.evaluation":
            raise ModuleNotFoundError("azure parent package absent")
        return object() if fault == "sdk" and name == "lm_eval" else None

    monkeypatch.setattr(LIVE.importlib.util, "find_spec", find_spec)
    if fault:
        with pytest.raises(ValueError):
            LIVE.installed_identity()
    else:
        assert LIVE.installed_identity()["sdk_modules_absent"] == list(LIVE.SDK_MODULES)


def test_judge_recipe_freezes_only_plan_with_actual_run_bindings(tmp_path):
    from invarlock.evaluation_records.io import run_digest

    _, args = fixture(tmp_path)
    recipe = COMMON.read(
        HERE.parents[2] / "src/invarlock/_data/examples/native-judge/judge-policy.json"
    )
    recipe["plan"]["sampling"]["case_units"] = [
        {"case_id": str(i), "unit_id": str(i)} for i in range(2)
    ]
    path = tmp_path / "recipe.json"
    COMMON.write(path, recipe)
    _, runs = LIVE.prepare(*args, "envelope", tmp_path / "recipient", path)
    plan = COMMON.read(tmp_path / "recipient/judge-plan.json")
    assert plan["baseline_run_sha256"] == run_digest(runs[0])
    assert plan["subject_run_sha256"] == run_digest(runs[1])
    assert (tmp_path / "recipient/judge-recipe.json").read_bytes() == path.read_bytes()
    assert not list((tmp_path / "recipient").rglob("*measurements*"))


@pytest.mark.parametrize("erase", [False, True])
def test_nonstring_planned_metadata_is_retained_in_native_context(tmp_path, erase):
    _, args = fixture(tmp_path, metadata={"cluster": {"index": 1}, "flag": False})
    if erase:
        path = tmp_path / "subject/native.json"
        native = COMMON.read(path)
        native["samples"][0]["metadata"].pop("cluster")
        path.write_bytes(COMMON.encoded(native))
        manifest_path = tmp_path / "subject/capture.json"
        manifest = COMMON.read(manifest_path)
        manifest["native_sha256"] = (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        manifest_path.write_bytes(COMMON.encoded(manifest))
        with pytest.raises(ValueError, match="frozen case metadata"):
            LIVE.prepare(*args, "native-json", tmp_path / "recipient")
    else:
        _, runs = LIVE.prepare(*args, "native-json", tmp_path / "recipient")
        assert runs[0]["records"][0]["context"]["upstream_record"]["metadata"][
            "cluster"
        ] == {"index": 1}


@pytest.mark.parametrize(
    "fault",
    ["directory", "taskdir", "symlink", "result", "output", "empty", "selection"],
)
def test_unsafe_or_incomplete_capture_refused(tmp_path, fault):
    protocol, args = fixture(tmp_path)
    directory = tmp_path / "subject"
    if fault == "directory":
        args[3] = tmp_path / "missing-directory"
    elif fault == "taskdir":
        (directory / "tasks").rename(directory / "missing-tasks")
    elif fault == "symlink":
        path = next((directory / "tasks").glob("*.request.json"))
        original = path.read_bytes()
        path.unlink()
        (directory / "original.json").write_bytes(original)
        path.symlink_to(directory / "original.json")
    elif fault == "selection":
        args[4] = "not-admitted"
    else:
        path = next((directory / "tasks").glob("*.response.json"))
        response = COMMON.read(path)
        if fault == "result":
            response["result"]["output"] = False
        elif fault == "output":
            response["result"]["output"] = "changed"
        else:
            response["result"]["output"] = None
            response["result"]["metadata"].pop("invarlock_likelihood")
        path.write_bytes(COMMON.encoded(response))
    with pytest.raises(ValueError):
        LIVE.prepare(*args, "native-json", tmp_path / "recipient")
    assert not (tmp_path / "recipient").exists()


@pytest.mark.parametrize("tamper_outcome", [False, True])
def test_ledger_generation_failure_is_preserved_without_invented_likelihood(
    tmp_path, tamper_outcome
):
    protocol, args = fixture(tmp_path)
    for role in LIVE.ROLES:
        directory = tmp_path / role
        native = COMMON.read(directory / "native.json")
        for path in (directory / "tasks").glob("*.response.json"):
            response = COMMON.read(path)
            response["result"]["output"] = None
            response["result"]["error"] = "synthetic generation failure"
            response["result"]["metadata"].pop("invarlock_likelihood")
            execution = response["result"]["metadata"]["invarlock_model_execution"]
            execution.pop("generation_result")
            execution.pop("likelihood_result")
            sample = next(
                row
                for row in native["samples"]
                if row["id"] == response["request"]["case_id"]
            )
            case = next(
                case for case in protocol["cases"] if case["id"] == sample["id"]
            )
            sample["metadata"] = {
                "family": "synthetic-test",
                **LIVE.bindings.bind_result(
                    response["result"], case, "inspect-ai", "0.3.254"
                )["metadata"],
            }
            sample["error"] = {
                "message": "SDK wrapper: synthetic generation failure",
                "type": "WrappedError",
            }
            if tamper_outcome and role == "subject":
                sample["metadata"]["invarlock_task_outcome"]["error"] = (
                    "different original error"
                )
            sample["output"] = {"choices": []}
            path.write_bytes(COMMON.encoded(response))
        native_path = directory / "native.json"
        native_path.write_bytes(COMMON.encoded(native))
        manifest_path = directory / "capture.json"
        manifest = COMMON.read(manifest_path)
        manifest["native_sha256"] = (
            "sha256:" + hashlib.sha256(native_path.read_bytes()).hexdigest()
        )
        manifest_path.write_bytes(COMMON.encoded(manifest))
    if tamper_outcome:
        with pytest.raises(ValueError, match="original execution metadata"):
            LIVE.prepare(*args, "native-json", tmp_path / "recipient")
        return
    _, runs = LIVE.prepare(*args, "native-json", tmp_path / "recipient")
    assert all(
        row["context"]["upstream_record"]["metadata"]["invarlock_task_outcome"]
        == {"output": None, "error": "synthetic generation failure"}
        for run in runs
        for row in run["records"]
    )
    assert all(row["error"] for run in runs for row in run["records"])
    assert all("likelihood" not in row for run in runs for row in run["records"])


@pytest.mark.parametrize("exit_code", [0, 2])
def test_subprocess_uses_isolated_core_and_rejects_failed_commands(
    monkeypatch, tmp_path, exit_code
):
    from types import SimpleNamespace

    seen = []

    def execute(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=exit_code, stdout='{"ok":true}\n', stderr="")

    monkeypatch.setenv("INVARLOCK_SIGNING_KEY", "must-not-be-inherited")
    monkeypatch.setenv("PYTHONPATH", "must-not-be-inherited")
    monkeypatch.setattr(LIVE.subprocess, "run", execute)
    if exit_code:
        with pytest.raises(RuntimeError, match="command failed"):
            LIVE.command(tmp_path, "verify", "evidence")
    else:
        assert LIVE.command(tmp_path, "verify", "evidence") == {"ok": True}
    assert seen[0][0][1:4] == ["-I", "-m", "invarlock"]
    assert "PYTHONPATH" not in seen[0][1]["env"]
    assert "INVARLOCK_SIGNING_KEY" not in seen[0][1]["env"]


@pytest.mark.parametrize("fault", ["preflight", "authentication", "replay"])
def test_journey_rejects_failed_preflight_or_verification(tmp_path, monkeypatch, fault):
    (tmp_path / "exact_match").mkdir()

    def command(_, operation, *args, **kwargs):
        if "--preflight" in args:
            return {"ok": fault != "preflight"}
        if operation == "evaluate":
            return {
                "authentication": "unsigned" if fault == "authentication" else "signed",
                "decision": "pass",
            }
        return {
            "integrity_ok": True,
            "replay_status": "incomplete" if fault == "replay" else "completed",
            "decision": "pass",
        }

    monkeypatch.setattr(LIVE, "command", command)
    with pytest.raises(ValueError):
        LIVE.journey(tmp_path)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("prepare_only", [False, True])
def test_cli_runs_recipient_preparation_and_optional_journey(
    tmp_path, monkeypatch, capsys, prepare_only
):
    _, args = fixture(tmp_path)
    output = tmp_path / "recipient"
    argv = [
        "recipient.py",
        "--protocol",
        str(args[0]),
        "--protocol-sha256",
        args[1],
        "--baseline-capture",
        str(args[2]),
        "--subject-capture",
        str(args[3]),
        "--evaluator",
        args[4],
        "--route",
        "native-json",
        "--output",
        str(output),
    ]
    if prepare_only:
        argv.append("--prepare-only")
    monkeypatch.setattr(LIVE.sys, "argv", argv)
    monkeypatch.setattr(
        LIVE, "installed_identity", lambda: {"test_only_identity": True}
    )
    called = []
    monkeypatch.setattr(
        LIVE, "journey", lambda path: called.append(path) or {"test_only_journey": True}
    )
    LIVE.main()
    assert COMMON.read(output / "recipient.json") == {"test_only_identity": True}
    assert bool(called) is not prepare_only
    assert bool(capsys.readouterr().out) is not prepare_only


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "field",
        "extra",
        "empty",
        "boolean",
        "negative",
        "huge",
        "fractional",
        "oversize",
        "joined",
        "decoded_context",
        "decoded_continuation",
        "decoded_joined",
        "token_count",
        "boolean_count",
        "generation_missing",
        "until",
        "max_gen_toks",
        "do_sample",
        "boolean_sampling",
        "generation_truncation",
    ],
)
def test_recipient_checks_retained_tokenization_and_generation_consistency(
    tmp_path, mutation
):
    protocol, _ = fixture(tmp_path)
    path = next((tmp_path / "baseline/tasks").glob("*.response.json"))
    response = COMMON.read(path)
    metadata = response["result"]["metadata"]
    execution = metadata["invarlock_model_execution"]
    tokens = execution["tokenization"]
    if mutation == "missing":
        execution.pop("tokenization")
    elif mutation == "field":
        tokens.pop("decoded_joined")
    elif mutation == "extra":
        tokens["unknown"] = True
    elif mutation in {"empty", "boolean", "negative", "huge", "fractional", "oversize"}:
        tokens["continuation_token_ids"] = {
            "empty": [],
            "boolean": [True],
            "negative": [-1],
            "huge": [2**31],
            "fractional": [2.0],
            "oversize": [2] * 514,
        }[mutation]
    elif mutation == "joined":
        tokens["joined_token_ids"] = [1, 3]
    elif mutation.startswith("decoded_"):
        tokens[mutation] = "changed original text"
    elif mutation in {"token_count", "boolean_count"}:
        metadata["invarlock_likelihood"]["token_count"] = (
            999 if mutation == "token_count" else True
        )
    elif mutation == "generation_missing":
        execution.pop("generation_parameters")
    elif mutation == "generation_truncation":
        tokens["context_token_ids"] = [1] * 481
        tokens["joined_token_ids"] = tokens["context_token_ids"] + [2]
    else:
        field = "do_sample" if mutation == "boolean_sampling" else mutation
        execution["generation_parameters"][field] = {
            "until": ["stop"],
            "max_gen_toks": 31,
            "do_sample": True,
            "boolean_sampling": 0,
        }[mutation]
    path.write_bytes(COMMON.encoded(response))
    with pytest.raises(ValueError, match="token|likelihood|generation"):
        LIVE.capture(tmp_path / "baseline", protocol, "baseline", "inspect-ai")


@pytest.mark.parametrize("mutation", ["missing", "output", "error"])
def test_original_task_outcome_is_required_even_when_other_native_fields_are_valid(
    tmp_path, mutation
):
    _, args = fixture(tmp_path)
    path = tmp_path / "subject/native.json"
    native = COMMON.read(path)
    metadata = native["samples"][0]["metadata"]
    if mutation == "missing":
        metadata.pop("invarlock_task_outcome")
    else:
        metadata["invarlock_task_outcome"][mutation] = "changed original task outcome"
    path.write_bytes(COMMON.encoded(native))
    manifest_path = tmp_path / "subject/capture.json"
    manifest = COMMON.read(manifest_path)
    manifest["native_sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    manifest_path.write_bytes(COMMON.encoded(manifest))
    with pytest.raises(ValueError, match="original execution metadata"):
        LIVE.prepare(*args, "native-json", tmp_path / "recipient")
    assert not (tmp_path / "recipient/exact_match").exists()
