"""Real framework orchestration with local fake tasks and no model requests."""

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "live_harness_test", ROOT / "examples/integrations/evaluator-live/harness.py"
)
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)
PINS = {
    "inspect-ai": ("inspect-ai", "0.3.254", "inspect_ai"),
    "lm-evaluation-harness": ("lm-eval", "0.4.12", "lm_eval"),
    "langfuse": ("langfuse", "4.14.1", "langfuse"),
    "lighteval": ("lighteval", "0.13.0", "lighteval"),
    "garak": ("garak", "0.15.1", "garak"),
    "openai-evals": ("evals", None, "evals"),
}


def _require(evaluator):
    selected = os.environ.get("INVARLOCK_LIVE_EVALUATOR")
    if selected and selected not in LIVE.EVALUATORS:
        pytest.fail("invalid live evaluator selection")
    if selected and not os.environ.get("INVARLOCK_EVALUATOR_PARITY_PYTHON"):
        pytest.fail("selected SDK requires separate installed recipient Python")
    if selected and selected != evaluator:
        pytest.skip("another evaluator is selected")
    try:
        if evaluator == "promptfoo":
            package = Path(os.environ["INVARLOCK_PROMPTFOO_PACKAGE"])
            assert (
                json.loads((package / "package.json").read_text())["version"]
                == "0.121.19"
            )
        else:
            package, pin, _ = PINS[evaluator]
            actual = importlib.metadata.version(package)
            assert pin is None or actual == pin
    except (KeyError, AssertionError, importlib.metadata.PackageNotFoundError) as exc:
        if selected:
            pytest.fail(f"required pinned evaluator unavailable: {exc}")
        pytest.skip("requires optional pinned evaluator")


@pytest.fixture
def cases():
    return [
        {
            "id": "case-a",
            "input": "Return alpha",
            "expected": "alpha",
            "metadata": {"split": "authored-test"},
        },
        {
            "id": "case-b",
            "input": "Return beta",
            "expected": "beta",
            "metadata": {"split": "authored-test"},
        },
    ]


@pytest.fixture(autouse=True)
def local_network_only(monkeypatch):
    connect = socket.socket.connect

    def guarded(sock, address):
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
            raise RuntimeError("SDK fake-task tests forbid external connections")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.mark.parametrize("evaluator", LIVE.EVALUATORS)
def test_actual_framework_drives_callback_and_writes_native_export(
    tmp_path, cases, evaluator, monkeypatch
):
    _require(evaluator)
    if evaluator == "lm-evaluation-harness":
        # Arrow expands the metadata struct across the complete dataset.
        cases[1]["metadata"]["reference_origin"] = "second-case-only"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    seen = []
    likelihoods = {}
    version = (
        "0.121.19"
        if evaluator == "promptfoo"
        else importlib.metadata.version(PINS[evaluator][0])
    )

    def digest(value):
        raw = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    def task(case):
        if evaluator != "promptfoo":
            prefix = PINS[evaluator][2] + "."
            assert any(
                frame.frame.f_globals.get("__name__", "").startswith(prefix)
                for frame in inspect.stack()
            )
        seen.append(case["id"])
        native_input = (
            case
            if evaluator == "lm-evaluation-harness"
            else {"prompt": case["input"], "case_id": case["id"]}
            if evaluator == "promptfoo"
            else case["input"]
        )
        # Authored likelihood values check lossless transport, never model quality.
        likelihoods[case["id"]] = {
            "basis": "reference_continuation",
            "logprob_sum": -2.25,
            "token_count": 2,
            "utf8_byte_count": len(case["expected"].encode()),
            "source": {"name": evaluator, "version": version},
            "input_digest": digest(native_input),
            "reference_digest": digest(case["expected"]),
            "artifact_digest": "sha256:" + "a" * 64,
            "tokenizer_digest": "sha256:" + "b" * 64,
            "configuration_digest": "sha256:" + "c" * 64,
        }
        return {
            "output": case["expected"],
            "metadata": {
                "runtime_fact": "actual-local-fake-task",
                "invarlock_likelihood": likelihoods[case["id"]],
            },
        }

    payload = LIVE.run(evaluator, cases, task, tmp_path)
    assert sorted(seen) == ["case-a", "case-b"]
    assert json.loads((tmp_path / "native.json").read_text()) == payload
    journal = [
        json.loads(row)
        for row in (tmp_path / "task-results.jsonl").read_text().splitlines()
    ]
    assert {row["id"]: row["output"] for row in journal} == {
        "case-a": "alpha",
        "case-b": "beta",
    }
    assert all(
        row["metadata"]["runtime_fact"] == "actual-local-fake-task" for row in journal
    )
    assert "alpha" in json.dumps(payload) and "beta" in json.dumps(payload)
    recipient = os.environ.get("INVARLOCK_EVALUATOR_PARITY_PYTHON")
    if recipient:
        version = (
            "0.121.19"
            if evaluator == "promptfoo"
            else importlib.metadata.version(PINS[evaluator][0])
        )
        pointer = {
            "lm-evaluation-harness": "/input/input",
            "promptfoo": "/input/prompt",
        }.get(evaluator)
        config = {
            "adapter": "evaluator-native-json",
            "source": {"name": evaluator, "version": version},
            "run_id": "local-live-capture",
            "artifact_digest": "sha256:" + "a" * 64,
            "input_projection": {"kind": "json-pointer", "pointer": pointer}
            if pointer
            else None,
        }
        imported = subprocess.run(
            [
                recipient,
                "-I",
                "-c",
                "import json,sys; from invarlock.evaluation_records.adapters import load_run; print(json.dumps(load_run(sys.argv[1], **json.loads(sys.argv[2]))))",
                str(tmp_path / "native.json"),
                json.dumps(config),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        records = {
            record["id"]: record for record in json.loads(imported.stdout)["records"]
        }
        assert set(records) == {case["id"] for case in cases}
        for case in cases:
            record = records[case["id"]]
            assert record["input"] == case["input"]
            assert record["expected"] == case["expected"]
            assert record["output"] == case["expected"]
            expected_likelihood = dict(likelihoods[case["id"]])
            if evaluator == "lm-evaluation-harness":
                native = next(row for row in payload if row["doc"]["id"] == case["id"])
                expected_likelihood["input_digest"] = digest(native["doc"])
                if case["id"] == "case-a":
                    assert native["doc"]["metadata"]["reference_origin"] is None
                    binding = native["metadata"]["invarlock_serialization_binding"]
                    assert binding["original_input"] == case
                    assert binding["original_likelihood"] == likelihoods[case["id"]]
                    assert binding["nullable_metadata_fields"] == ["reference_origin"]
            assert record["likelihood"] == expected_likelihood
            assert not record.get("error")
            assert record["metadata"]["split"] == "authored-test"


def test_callback_failure_retains_original_error_and_does_not_retry(tmp_path, cases):
    def task(case):
        raise RuntimeError("deliberate task execution failure")

    capture = LIVE.Capture(cases, task, tmp_path)
    with pytest.raises(RuntimeError, match="deliberate"):
        capture.call("case-a")
    with pytest.raises(ValueError, match="repeated"):
        capture.call("case-a")
    retained = json.loads((tmp_path / "task-results.jsonl").read_text())
    assert retained["output"] is None
    assert retained["error"] == "deliberate task execution failure"
    with pytest.raises(ValueError, match="every frozen"):
        capture.complete()


@pytest.mark.parametrize("evaluator", LIVE.EVALUATORS)
def test_actual_framework_prompt_mutation_is_refused_before_task(
    tmp_path, cases, evaluator, monkeypatch
):
    _require(evaluator)
    original = LIVE.Capture.check_prompt
    checked = []

    def altered(self, ident, prompt):
        checked.append(prompt)
        original(self, ident, prompt + " unexpected framework transform")

    monkeypatch.setattr(LIVE.Capture, "check_prompt", altered)
    called = []
    with pytest.raises(ValueError, match="frozen"):
        LIVE.run(evaluator, cases[:1], lambda case: called.append(case), tmp_path)
    assert checked == [cases[0]["input"]]
    assert called == []


@pytest.mark.parametrize("evaluator", LIVE.EVALUATORS)
def test_actual_framework_task_failure_is_retained(tmp_path, cases, evaluator):
    _require(evaluator)
    try:
        payload = LIVE.run(
            evaluator,
            cases[:1],
            lambda case: {"output": None, "error": "deliberate live task failure"},
            tmp_path,
        )
    except Exception:
        assert not (tmp_path / "native.json").exists()
    else:
        assert "deliberate live task failure" in json.dumps(payload)
    rows = [
        json.loads(line)
        for line in (tmp_path / "task-results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["output"] is None
    assert rows[0]["error"] == "deliberate live task failure"


@pytest.mark.parametrize(
    "result",
    [
        {"output": 1},
        {"output": "x", "metadata": []},
        {"output": "x", "metadata": {"split": "changed"}},
    ],
)
def test_callback_rejects_invalid_or_conflicting_capture_facts(tmp_path, cases, result):
    capture = LIVE.Capture(cases, lambda _: result, tmp_path)
    with pytest.raises(ValueError):
        capture.call("case-a")
    assert json.loads(capture.journal.read_text())["output"] is None


def test_no_fabricated_generation_from_likelihood_only_result(tmp_path, cases):
    capture = LIVE.Capture(
        cases,
        lambda _: {
            "output": None,
            "metadata": {"invarlock_likelihood": {"logprob_sum": -1}},
        },
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="no text"):
        capture.generation("case-a")
    assert json.loads(capture.journal.read_text())["output"] is None


def test_unknown_evaluator_and_duplicate_cases_fail_before_task(tmp_path, cases):
    with pytest.raises(ValueError, match="unsupported"):
        LIVE.run("unknown", cases, None, tmp_path)
    with pytest.raises(ValueError, match="unique"):
        LIVE.Capture(cases + cases, None, tmp_path)
