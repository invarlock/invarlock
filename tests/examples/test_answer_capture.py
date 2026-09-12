from __future__ import annotations

import asyncio
import copy
import json
import time

import pytest

from examples import answer_capture as capture
from invarlock.evaluation_records.cases import case_set_digest, validate_run_case_set


class Adapter:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0

    def count_input_tokens(self, request):
        assert "expected" not in request
        return 2

    async def generate(self, request):
        self.calls.append(copy.deepcopy(request))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        return {
            "output": f"answer:{request['side']}:{request['case_id']}",
            "input_tokens": 2,
            "output_tokens": 3,
        }


@pytest.fixture
def inputs():
    model = {
        "provider": "fixture",
        "model": "toy",
        "revision": "immutable-fixture-1",
        "tokenizer": "toy-1",
        "artifact_digest": "sha256:" + "a" * 64,
        "generation": {"temperature": 0},
    }
    limits = {
        "concurrency": 2,
        "deadline_unix_seconds": int(time.time()) + 300,
        "call_timeout_seconds": 2,
        "max_calls": 4,
        "max_input_tokens_per_call": 10,
        "max_output_tokens_per_call": 10,
        "max_output_bytes_per_call": 1000,
        "max_total_input_tokens": 40,
        "max_total_output_tokens": 40,
        "max_cost_microusd": 80,
        "input_microusd_per_million_tokens": 1_000_000,
        "output_microusd_per_million_tokens": 1_000_000,
    }
    cases = {
        "format": "invarlock/evaluation-case-set-v1",
        "cases": [
            {
                "id": name,
                "input": "question",
                "expected": "secret reference",
                "metadata": {},
            }
            for name in ("case-b", "case-a")
        ],
    }
    return {
        "config": {
            "baseline": model,
            "subject": {**model, "generation": {"temperature": 1}},
            "limits": limits,
        },
        "cases": cases,
        "adapter_sha256": "sha256:" + "b" * 64,
    }


def run(tmp_path, inputs, adapter=None):
    return asyncio.run(
        capture.capture(
            **inputs, adapter=adapter or Adapter(), directory=tmp_path / "capture"
        )
    )


def test_capture_freezes_membership_identity_and_resumes_without_calls(
    tmp_path, inputs
):
    adapter = Adapter()
    result = run(tmp_path, inputs, adapter)
    assert len(adapter.calls) == 4
    assert adapter.peak == 2
    assert result["case_set"] == case_set_digest(inputs["cases"])
    for side in ("baseline", "subject"):
        value = capture.read(tmp_path / "capture" / f"{side}_run.json")
        validate_run_case_set(value, result["case_set"])
        assert [row["id"] for row in value["records"]] == ["case-a", "case-b"]
        assert (
            value["records"][0]["context"]["model_identity"] == inputs["config"][side]
        )
        assert value["records"][0]["scores"] == {}
    assert run(tmp_path, inputs, adapter) == result
    assert len(adapter.calls) == 4


@pytest.mark.parametrize("mutation", ["case", "model", "adapter", "deadline", "result"])
def test_resume_rejects_changed_frozen_inputs_before_calls(tmp_path, inputs, mutation):
    run(tmp_path, inputs)
    adapter = Adapter()
    if mutation == "case":
        inputs["cases"]["cases"][0]["input"] = "changed"
    elif mutation == "model":
        inputs["config"]["subject"]["model"] = "changed"
    elif mutation == "adapter":
        inputs["adapter_sha256"] = "sha256:" + "c" * 64
    elif mutation == "deadline":
        inputs["config"]["limits"]["deadline_unix_seconds"] += 1
    else:
        result = tmp_path / "capture" / "000000.result.json"
        value = json.loads(result.read_text())
        value["job"] = 1
        result.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run(tmp_path, inputs, adapter)
    assert adapter.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "max_calls",
        "max_total_input_tokens",
        "max_total_output_tokens",
        "max_cost_microusd",
        "max_input_tokens_per_call",
    ],
)
def test_admission_caps_prevent_all_calls(tmp_path, inputs, field):
    inputs["config"]["limits"][field] = 1
    adapter = Adapter()
    with pytest.raises(ValueError):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_ambiguous_attempt_blocks_resume_without_retry(tmp_path, inputs):
    class Failed(Adapter):
        async def generate(self, request):
            self.calls.append(request)
            raise RuntimeError("connection lost after dispatch")

    failed = Failed()
    with pytest.raises(ExceptionGroup):
        run(tmp_path, inputs, failed)
    assert failed.calls
    adapter = Adapter()
    with pytest.raises(ValueError, match="ambiguous"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_partial_completed_results_resume_only_unattempted_jobs(tmp_path, inputs):
    directory = tmp_path / "capture"
    directory.mkdir()
    adapter = Adapter()
    manifest, jobs = capture.prepare(**inputs, adapter=adapter)
    capture.retain(directory / "manifest.json", manifest)
    binding = {"manifest_sha256": capture.digest(manifest), "job": 0}
    capture.retain(directory / "000000.attempt.json", binding)
    result = {"output": "bad answer is retained", "input_tokens": 2, "output_tokens": 3}
    capture.retain(directory / "000000.result.json", {**binding, "result": result})
    run(tmp_path, inputs, adapter)
    assert len(adapter.calls) == len(jobs) - 1
    baseline = capture.read(directory / "baseline_run.json")
    assert baseline["records"][0]["output"] == "bad answer is retained"


@pytest.mark.parametrize(
    "bad",
    [
        {"output": "a", "input_tokens": 3, "output_tokens": 3},
        {"output": "a", "input_tokens": 2, "output_tokens": 11},
        {"output": "a", "input_tokens": 2, "output_tokens": True},
        {"output": "a", "input_tokens": 2, "output_tokens": 3, "extra": 1},
    ],
)
def test_provider_usage_violation_leaves_no_retriable_result(tmp_path, inputs, bad):
    class Invalid(Adapter):
        async def generate(self, request):
            return bad

    with pytest.raises(ExceptionGroup):
        run(tmp_path, inputs, Invalid())
    assert not list((tmp_path / "capture").glob("*.result.json"))
    with pytest.raises(ValueError, match="ambiguous"):
        run(tmp_path, inputs)


def test_timeout_retains_ambiguous_dispatch(tmp_path, inputs):
    inputs["config"]["limits"]["call_timeout_seconds"] = 1

    class Slow(Adapter):
        async def generate(self, request):
            await asyncio.sleep(2)

    with pytest.raises(ExceptionGroup):
        run(tmp_path, inputs, Slow())
    with pytest.raises(ValueError, match="ambiguous"):
        run(tmp_path, inputs)


def test_expired_deadline_does_not_call(tmp_path, inputs):
    inputs["config"]["limits"]["deadline_unix_seconds"] = int(time.time()) - 1
    adapter = Adapter()
    with pytest.raises(ValueError, match="deadline"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_unsafe_directory_or_existing_unrelated_files_prevent_calls(tmp_path, inputs):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "capture").symlink_to(target, target_is_directory=True)
    adapter = Adapter()
    with pytest.raises(ValueError):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls
    (tmp_path / "capture").unlink()
    (tmp_path / "capture").mkdir()
    (tmp_path / "capture" / "unrelated").write_text("preserve me")
    with pytest.raises(ValueError, match="unexpected"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_existing_run_without_results_blocks_calls(tmp_path, inputs):
    directory = tmp_path / "capture"
    directory.mkdir()
    (directory / "baseline_run.json").write_text("{}")
    adapter = Adapter()
    with pytest.raises(ValueError, match="published run"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_completed_empty_answer_is_retained(tmp_path, inputs):
    class Empty(Adapter):
        async def generate(self, request):
            return {"output": "", "input_tokens": 2, "output_tokens": 0}

    run(tmp_path, inputs, Empty())
    assert (
        capture.read(tmp_path / "capture" / "baseline_run.json")["records"][0]["output"]
        == ""
    )


def test_concurrent_captures_do_not_duplicate_dispatch(tmp_path, inputs):
    async def together():
        adapter = Adapter()
        first = asyncio.create_task(
            capture.capture(**inputs, adapter=adapter, directory=tmp_path / "capture")
        )
        await asyncio.sleep(0)
        with pytest.raises(BlockingIOError):
            await capture.capture(
                **inputs, adapter=adapter, directory=tmp_path / "capture"
            )
        await first
        assert len(adapter.calls) == 4

    asyncio.run(together())


def test_output_byte_limit_violation_preserves_attempt(tmp_path, inputs):
    inputs["config"]["limits"]["max_output_bytes_per_call"] = 1
    with pytest.raises(ExceptionGroup):
        run(tmp_path, inputs)
    with pytest.raises(ValueError, match="ambiguous"):
        run(tmp_path, inputs)


def test_storage_envelope_is_admitted_before_calls(tmp_path, inputs):
    inputs["config"]["limits"]["max_output_bytes_per_call"] = 2 * 1024 * 1024
    adapter = Adapter()
    with pytest.raises(ValueError, match="byte cap"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_storage_reservation_counts_journal_and_run_output_copies(tmp_path, inputs):
    cases = inputs["cases"]["cases"]
    inputs["cases"]["cases"] = [
        {**copy.deepcopy(cases[index % len(cases)]), "id": f"case-{index:02}"}
        for index in range(10)
    ]
    limits = inputs["config"]["limits"]
    limits.update(
        max_calls=20,
        max_output_bytes_per_call=1024 * 1024,
        max_total_input_tokens=200,
        max_total_output_tokens=200,
        max_cost_microusd=400,
    )
    adapter = Adapter()
    with pytest.raises(ValueError, match="storage envelope"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls


def test_documented_cli_offline_journey_and_explicit_execution(
    tmp_path, monkeypatch, capsys
):
    config = capture.read(capture.Path("examples/answer-capture/config.json"))
    config["limits"]["deadline_unix_seconds"] = int(time.time()) + 300
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    command = [
        "answer_capture",
        "--config",
        str(config_path),
        "--cases",
        "examples/answer-capture/cases.json",
        "--adapter",
        "examples.answer_capture_offline",
        "--directory",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr("sys.argv", command)
    with pytest.raises(SystemExit) as exc:
        capture.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr("sys.argv", [*command, "--execute"])
    capture.main()
    assert "Frozen answer runs ready" in capsys.readouterr().out
    assert (tmp_path / "out" / "baseline_run.json").is_file()
    capture.main()


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_positive_limits_reject_nonpositive_and_noninteger_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        capture.positive(value, "limit")


def test_read_requires_an_object(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        capture.read(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", ""),
        ("model", None),
        ("revision", ""),
        ("tokenizer", ""),
        ("artifact_digest", "unpinned"),
        ("generation", []),
    ],
)
def test_capture_rejects_incomplete_model_identity_before_dispatch(
    tmp_path, inputs, field, value
):
    inputs["config"]["subject"][field] = value
    adapter = Adapter()
    with pytest.raises(ValueError):
        run(tmp_path, inputs, adapter)
    assert adapter.calls == []


def test_capture_requires_adapter_digest(tmp_path, inputs):
    inputs["adapter_sha256"] = "unpinned"
    with pytest.raises(ValueError, match="source digest"):
        run(tmp_path, inputs)


def test_retained_result_requires_dispatch_attempt(tmp_path, inputs):
    run(tmp_path, inputs)
    (tmp_path / "capture/000000.attempt.json").unlink()
    with pytest.raises(ValueError, match="missing its retained dispatch"):
        run(tmp_path, inputs)


def test_deadline_rechecked_before_dispatch(tmp_path, inputs, monkeypatch):
    deadline = inputs["config"]["limits"]["deadline_unix_seconds"]
    ticks = iter([deadline - 1, deadline + 1])
    monkeypatch.setattr(capture.time, "time", lambda: next(ticks, deadline + 1))
    adapter = Adapter()
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        run(tmp_path, inputs, adapter)
    assert not adapter.calls
    assert not list((tmp_path / "capture").glob("*.attempt.json"))
