"""Scheduled K2 captures keep frozen snapshot and preflight checks."""

import json
from pathlib import Path

import pytest

from examples.qualification import campaign_k2 as adapter
from tests.examples.test_campaign_scheduling import manifest as schedule_manifest
from tests.examples.test_k2_campaign import _ready_plan
from tests.examples.test_k2_producer import _transport


@pytest.fixture
def sample(monkeypatch):
    plan = _ready_plan()
    monkeypatch.setattr(
        adapter.campaign,
        "measure_snapshot",
        lambda path, files: plan["model"]["baseline"]["materialized"],
    )
    request, _ = _transport(plan)
    capture = adapter.capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    capture["resources"]["startup_seconds"] = 3
    capture["hardware"] = ["NVIDIA H100 80GB HBM3, 81559, 580.173.02, Disabled"]
    return plan, capture


def test_prepare_capture_and_postvalidation_preserve_identity(
    sample, tmp_path, monkeypatch
):
    plan, captured = sample
    prepared = adapter.prepare(plan, "baseline", tmp_path, tmp_path, "preflight")
    seen = []
    monkeypatch.setattr(
        adapter.capture_worker, "worker", lambda *args: seen.append(args)
    )
    adapter.capture(plan, "baseline", "preflight", prepared, tmp_path)
    assert seen[0][0] == plan
    report = adapter.validate(
        plan, "baseline", tmp_path, prepared, captured, tmp_path, "gpu-job", "work-v1"
    )
    assert report["semantic_ready"] is True
    assert report["capture_digest"] == adapter.campaign.digest(captured)
    sentinel = json.loads((tmp_path / "sentinel.json").read_text())
    assert sentinel["observed_job_id"] == "gpu-job" and sentinel["units"] == 24
    assert sentinel["fixed_seconds"] == 3
    assert "quality" in report["limit"]


def test_empty_final_answer_is_not_ready_but_raw_diagnostic_survives(sample, tmp_path):
    plan, captured = sample
    prepared = adapter.prepare(plan, "baseline", tmp_path, tmp_path, "preflight")
    captured["rows"][0]["response"]["choices"][0]["message"]["content"] = " \n "
    report = adapter.validate(
        plan, "baseline", tmp_path, prepared, captured, tmp_path, "gpu-job", "work-v1"
    )
    assert report["complete"] is True and report["semantic_ready"] is False
    captured["rows"][0]["error"] = "failed request"
    second = tmp_path / "second"
    second.mkdir()
    report = adapter.validate(
        plan, "baseline", tmp_path, prepared, captured, second, "gpu-job", "work-v1"
    )
    assert report["complete"] is False and report["nonempty_final_answers"] == 23


def test_materialization_and_prepared_receipt_changes_are_rejected(
    sample, tmp_path, monkeypatch
):
    plan, captured = sample
    prepared = adapter.prepare(plan, "baseline", tmp_path, tmp_path, "preflight")
    wrong = dict(prepared, role="candidate")
    with pytest.raises(ValueError, match="prepared"):
        adapter.capture(plan, "baseline", "preflight", wrong, tmp_path)
    with pytest.raises(ValueError, match="snapshot differs"):
        adapter.validate(
            plan, "baseline", tmp_path, wrong, captured, tmp_path, "gpu-job", "key"
        )
    monkeypatch.setattr(adapter.campaign, "measure_snapshot", lambda *args: {})
    with pytest.raises(ValueError, match="actual snapshot differs"):
        adapter.checked_snapshot(plan, "baseline", tmp_path)


def test_decision_requires_matching_complete_nonempty_preflight(sample, tmp_path):
    plan, captured = sample
    with pytest.raises(ValueError, match="requires"):
        adapter.prepare(plan, "baseline", tmp_path, tmp_path, "decision")
    adapter.prepare(plan, "baseline", tmp_path, tmp_path, "decision", captured)
    captured["rows"][0]["response"]["choices"][0]["message"]["content"] = ""
    with pytest.raises(ValueError, match="empty"):
        adapter.prepare(plan, "baseline", tmp_path, tmp_path, "decision", captured)


def test_wrong_roles_cannot_use_matching_capture_structure(
    sample, tmp_path, monkeypatch
):
    plan, captured = sample
    monkeypatch.setattr(
        adapter.campaign, "validate_capture", lambda *a, **kw: "candidate"
    )
    with pytest.raises(ValueError, match="role differs"):
        adapter.prepare(plan, "baseline", tmp_path, tmp_path, "decision", captured)
    prepared = adapter.prepare(plan, "baseline", tmp_path, tmp_path, "preflight")
    with pytest.raises(ValueError, match="role differs"):
        adapter.validate(
            plan, "baseline", tmp_path, prepared, captured, tmp_path, "gpu-job", "key"
        )


@pytest.mark.parametrize("exclusive,tp", [(False, 1), (True, 2)])
def test_manifest_restores_checks_and_binds_worker_source(
    sample, tmp_path, exclusive, tp
):
    plan = adapter.campaign.select_plan("mova-36b-a4b" if tp == 2 else "0.9b")
    ready = _ready_plan()
    for key in adapter.campaign._RUNTIME_BINDINGS:
        plan["runtime"][key] = ready["runtime"][key]
    plan["budget"] = ready["budget"]
    for role in adapter.campaign.ROLES:
        plan["model"][role]["materialized"] = ready["model"][role]["materialized"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    host = schedule_manifest()["host"]
    host.update(cpu_ids=list(range(72)), memory_mib=280000)
    result = adapter.make_manifest(
        plan_path,
        {"baseline": tmp_path / "base", "candidate": tmp_path / "candidate"},
        host,
        tmp_path / "results",
        Path(adapter.__file__),
        exclusive=exclusive,
    )
    assert len(result["jobs"]) == 6
    for role in ("baseline", "candidate"):
        prefix = plan["model"]["id"].replace(".", "-") + "-" + role
        jobs = {j["id"]: j for j in result["jobs"]}
        assert jobs[prefix + "-capture"]["depends_on"] == [prefix + "-prepare"]
        assert jobs[prefix + "-validate"]["depends_on"] == [prefix + "-capture"]
        assert jobs[prefix + "-capture"]["resources"]["gpus"] == tp
        assert jobs[prefix + "-capture"]["resources"]["exclusive"] is exclusive
        assert jobs[prefix + "-prepare"]["resources"]["gpus"] == 0
        assert all(
            m["sha256"].startswith("sha256:")
            for m in jobs[prefix + "-prepare"]["container"]["mounts"][:2]
        )


def test_cli_exercises_all_stages_and_missing_inputs(sample, tmp_path, monkeypatch):
    plan, captured = sample
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    capture_path = tmp_path / "capture-input.json"
    capture_path.write_text(json.dumps(captured))
    base = [
        "--plan",
        str(plan_path),
        "--role",
        "baseline",
        "--phase",
        "preflight",
        "--snapshot",
        str(tmp_path),
        "--output",
        str(tmp_path),
    ]
    assert adapter.main(["prepare", *base]) == 0
    monkeypatch.setattr(adapter.capture_worker, "worker", lambda *a: None)
    assert (
        adapter.main(["capture", *base, "--prepared", str(tmp_path / "prepared.json")])
        == 0
    )
    assert (
        adapter.main(
            [
                "validate",
                *base,
                "--prepared",
                str(tmp_path / "prepared.json"),
                "--capture",
                str(capture_path),
                "--observed-job-id",
                "gpu-job",
                "--workload-key",
                "key",
            ]
        )
        == 0
    )
    for command in ("capture", "validate"):
        with pytest.raises(SystemExit):
            adapter.main([command, *base])
    decision = base.copy()
    decision[decision.index("preflight")] = "decision"
    decision_output = tmp_path / "decision"
    decision_output.mkdir()
    decision[decision.index("--output") + 1] = str(decision_output)
    assert adapter.main(["prepare", *decision, "--preflight", str(capture_path)]) == 0
