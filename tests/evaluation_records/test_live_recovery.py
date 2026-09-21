"""Explicit partial transport repair uses synthetic observations and no model calls."""

from __future__ import annotations

import copy
import sys

import pytest

from tests.evaluation_records.test_live_campaign_helpers import exchange as exchange
from tests.evaluation_records.test_live_campaign_helpers import modules as modules
from tests.evaluation_records.test_live_campaign_helpers import planned


def partial(modules, tmp_path):
    common = modules.common
    recovery = common.module("recovery")
    rows = [{**planned()[0], "id": f"case-{i}"} for i in range(8)]
    protocol = {"cases": rows, "evaluators": ["azure-ai-evaluation"]}
    source, worker = tmp_path / "failed-capture", tmp_path / "worker"
    (source / "tasks").mkdir(parents=True)
    (source / "sdk").mkdir()
    worker.mkdir()
    common.write(source / "protocol.json", {**protocol, "role": "subject"})
    common.write(
        source / "failure.json",
        {"status": "capture_failed", "completed_case_ids": [r["id"] for r in rows[:3]]},
    )
    common.write(
        source / "sdk/native.json",
        {"status": "synthetic failed SDK artifact retained unchanged"},
    )
    requests = {}
    for i, row in enumerate(rows):
        request = {
            "evaluator": "azure-ai-evaluation",
            "case_id": row["id"],
            "protocol_digest": common.digest({**protocol, "role": "subject"}),
        }
        stem = common.digest(request).removeprefix("sha256:")
        requests[row["id"]] = (request, stem)
        common.write(source / "tasks" / (stem + ".request.json"), request)
        if i < 3:
            response = {
                "request": request,
                "result": {
                    "output": "Original answer " + row["id"],
                    "metadata": {"synthetic": True},
                },
            }
            common.write(source / "tasks" / (stem + ".response.json"), response)
            common.write(worker / (stem + ".request.json"), request)
            common.write(worker / (stem + ".response.json"), response)
    args = (source, worker, protocol, "subject", "azure-ai-evaluation")
    proposal, *_ = recovery.inspect(*args)
    return recovery, args, proposal, requests


def test_recovery_replays_three_original_results_without_transport(
    modules, tmp_path, monkeypatch
):
    recovery, args, proposal, requests = partial(modules, tmp_path)
    common = modules.common
    approved = recovery.Recovery(*args, common.digest(proposal))
    output = tmp_path / "new-capture"
    output.mkdir()
    approved.stage(output)
    client = recovery.TaskClient(
        "unused",
        "azure-ai-evaluation",
        proposal["protocol_digest"],
        args[2]["cases"],
        output / "tasks",
        recovery=approved,
    )
    monkeypatch.setattr(
        common.socket,
        "socket",
        lambda *a: pytest.fail("retained response must not call model transport"),
    )
    for case in args[2]["cases"][:3]:
        raw = approved.responses[case["id"]]
        result = client(case)
        replay = result["metadata"].pop("invarlock_transport_replay")
        assert result == common.decode(raw)["result"]
        assert replay["original_response_bytes"].encode() == raw
        assert replay["original_response_sha256"] == recovery.physical(raw)
        assert replay["recovery_sha256"] == common.digest(proposal)
        with pytest.raises(ValueError, match="repeats"):
            client(case)
    assert len(client.results) == 3
    assert proposal["new_case_ids"] == [f"case-{i}" for i in range(3, 8)]
    for name, raw in approved.files.items():
        assert (output / "recovery-original" / name).read_bytes() == raw
        assert (args[0] / name).read_bytes() == raw
    for name, raw in approved.worker_files.items():
        assert (output / "recovery-worker" / name).read_bytes() == raw


def test_recovery_forwards_only_proven_unstarted_request(modules, tmp_path, exchange):
    recovery, args, proposal, _ = partial(modules, tmp_path)
    common = modules.common
    approved = recovery.Recovery(*args, common.digest(proposal))
    with exchange(
        lambda request: common.encoded(
            {
                "request": request,
                "result": {"output": "Fresh synthetic test answer", "metadata": {}},
            }
        )
    ) as (address, received):
        client = recovery.TaskClient(
            address,
            "azure-ai-evaluation",
            proposal["protocol_digest"],
            args[2]["cases"],
            tmp_path / "new-tasks",
            recovery=approved,
        )
        assert client(args[2]["cases"][3])["output"] == "Fresh synthetic test answer"
    assert [request["case_id"] for request in received] == ["case-3"]


@pytest.mark.parametrize(
    "fault",
    [
        "pin",
        "protocol",
        "complete",
        "failure",
        "admitted",
        "worker-changed",
        "extra",
        "nested",
        "source-link",
        "file-link",
        "oversized",
        "changed-case",
    ],
)
def test_recovery_cannot_repeat_uncertain_admission_or_change_originals(
    modules, tmp_path, fault
):
    recovery, args, proposal, requests = partial(modules, tmp_path)
    common = modules.common
    source, worker, protocol, *_ = args
    pin = common.digest(proposal)
    if fault == "pin":
        pin = "wrong-independent-pin"
    elif fault == "protocol":
        protocol["unexpected"] = True
    elif fault == "complete":
        common.write(source / "capture.json", {})
    elif fault == "failure":
        (source / "failure.json").write_bytes(common.encoded({"status": "success"}))
    elif fault == "admitted":
        request, stem = requests["case-3"]
        common.write(worker / (stem + ".request.json"), request)
    elif fault == "worker-changed":
        _, stem = requests["case-0"]
        path = worker / (stem + ".response.json")
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "extra":
        common.write(source / "tasks/extra.json", {})
    elif fault == "nested":
        _, stem = requests["case-0"]
        path = source / "tasks" / (stem + ".response.json")
        response = common.read(path)
        response["result"]["metadata"]["invarlock_transport_replay"] = {}
        path.write_bytes(common.encoded(response))
        (worker / path.name).write_bytes(path.read_bytes())
    elif fault == "source-link":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        args = (alias, *args[1:])
    elif fault == "file-link":
        (source / "sdk/link.json").symlink_to(source / "protocol.json")
    elif fault == "oversized":
        (source / "sdk/large.json").write_bytes(b"x" * (common.MAX_MESSAGE + 1))
    elif fault == "changed-case":
        approved = recovery.Recovery(*args, pin)
        client = recovery.TaskClient(
            "unused",
            "azure-ai-evaluation",
            proposal["protocol_digest"],
            protocol["cases"],
            tmp_path / "new-tasks",
            recovery=approved,
        )
        case = copy.deepcopy(protocol["cases"][0])
        case["input"] += " changed"
        with pytest.raises(ValueError, match="admitted case"):
            client(case)
        return
    with pytest.raises(ValueError):
        recovery.Recovery(*args, pin)


def test_recovery_proposal_cli_is_read_only(modules, tmp_path, monkeypatch, capsys):
    recovery, args, proposal, _ = partial(modules, tmp_path)
    source, worker, protocol, role, evaluator = args
    path = tmp_path / "protocol.json"
    modules.common.write(path, protocol)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recovery.py",
            "--recover-from",
            str(source),
            "--worker-ledger",
            str(worker),
            "--protocol",
            str(path),
            "--role",
            role,
            "--evaluator",
            evaluator,
        ],
    )
    recovery.main()
    result = modules.common.decode(capsys.readouterr().out.encode())
    assert result == {
        "proposal": proposal,
        "recovery_sha256": modules.common.digest(proposal),
    }


@pytest.mark.parametrize(
    "fault", ["missing-worker", "too-many-files", "request", "response"]
)
def test_recovery_inventory_and_response_identity_are_bounded(modules, tmp_path, fault):
    recovery, args, proposal, requests = partial(modules, tmp_path)
    source, worker, *_ = args
    _, stem = requests["case-0"]
    if fault == "missing-worker":
        (worker / (stem + ".response.json")).unlink()
    elif fault == "too-many-files":
        for i in range(1001):
            (source / "sdk" / f"extra-{i}").write_bytes(b"")
    elif fault == "request":
        (source / "tasks" / (stem + ".request.json")).write_bytes(b"{}")
    else:
        for directory in (source / "tasks", worker):
            path = directory / (stem + ".response.json")
            response = modules.common.read(path)
            response["request"]["case_id"] = "different-case"
            path.write_bytes(modules.common.encoded(response))
    with pytest.raises(ValueError):
        recovery.Recovery(*args, modules.common.digest(proposal))


def test_capture_recovery_orchestration_retains_old_sdk_failures_and_calls_only_five(
    modules, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    recovery, args, _, _ = partial(modules, tmp_path)
    source, worker, protocol, role, evaluator = args
    common = modules.common
    protocol.update(
        versions={evaluator: "1.18.1"}, models={role: {"synthetic-model": True}}
    )
    # Recreate the original protocol/request binding after adding manifest fields.
    old_digest = common.digest(common.read(source / "protocol.json"))
    common_protocol = {**protocol, "role": role}
    for directory in (source / "tasks", worker):
        for path in list(directory.iterdir()):
            value = common.read(path)
            request = value.get("request", value)
            assert request["protocol_digest"] == old_digest
            request["protocol_digest"] = common.digest(common_protocol)
            name = common.digest(request).removeprefix("sha256:") + (
                ".response.json" if "result" in value else ".request.json"
            )
            path.unlink()
            common.write(directory / name, value)
    (source / "protocol.json").write_bytes(common.encoded(common_protocol))
    proposal, *_ = recovery.inspect(*args)
    pin = common.digest(proposal)
    actual_module = common.module
    new_calls = []

    def fresh(self, case):
        new_calls.append(case["id"])
        request = {
            "evaluator": self.evaluator,
            "case_id": case["id"],
            "protocol_digest": self.protocol_digest,
        }
        result = {"output": "New synthetic answer", "metadata": {}}
        stem = common.digest(request).removeprefix("sha256:")
        common.write(self.output / (stem + ".request.json"), request)
        common.write(
            self.output / (stem + ".response.json"),
            {"request": request, "result": result},
        )
        self.results[case["id"]] = result
        return copy.deepcopy(result)

    def driver(evaluator, cases, task, workdir):
        return {"rows": [task(case) for case in cases]}

    monkeypatch.setattr(common.TaskClient, "_call", fresh)
    monkeypatch.setattr(
        common,
        "module",
        lambda name: (
            SimpleNamespace(run=driver) if name == "batch" else actual_module(name)
        ),
    )
    monkeypatch.setattr(modules.capture, "local_environment", lambda _: None)
    monkeypatch.setattr(
        modules.capture.importlib.metadata, "version", lambda _: "1.18.1"
    )
    output = tmp_path / "capture"
    with pytest.raises(ValueError, match="requires original capture"):
        modules.capture.capture(
            protocol, role, evaluator, "unused", output, recover_from=source
        )
    manifest = modules.capture.capture(
        protocol,
        role,
        evaluator,
        "unused",
        output,
        recover_from=source,
        worker_ledger=worker,
        recovery_sha256=pin,
    )
    assert new_calls == [f"case-{i}" for i in range(3, 8)]
    assert manifest["case_count"] == 8
    assert manifest["transport_recovery"] == {"admission_sha256": pin, **proposal}
    assert manifest["driver_files"]["recovery"] == recovery.physical(
        (common.HERE / "recovery.py").read_bytes()
    )
    assert (output / "recovery-original/sdk/native.json").read_bytes() == (
        source / "sdk/native.json"
    ).read_bytes()
    assert len(list((output / "tasks").glob("*.response.json"))) == 8


def recipient_fixture(tmp_path):
    import shutil

    from tests.evaluation_records.test_live_recipient import LIVE, fixture

    protocol, args = fixture(tmp_path)
    args.extend(["envelope", tmp_path / "recipient"])
    common = LIVE.common
    recovery = common.module("recovery")
    source, worker = tmp_path / "original-failure", tmp_path / "original-worker"
    shutil.copytree(tmp_path / "baseline", source)
    (source / "capture.json").unlink()
    worker.mkdir()
    for path in list((source / "tasks").glob("*.response.json")):
        response = common.read(path)
        if response["request"]["case_id"] == "0":
            shutil.copy2(path, worker / path.name)
            request_path = path.with_name(path.name.replace(".response.", ".request."))
            shutil.copy2(request_path, worker / request_path.name)
        else:
            path.unlink()
    common.write(
        source / "failure.json",
        {"status": "capture_failed", "completed_case_ids": ["0"]},
    )
    proposal, *_ = recovery.inspect(source, worker, protocol, "baseline", "inspect-ai")
    approved = recovery.Recovery(
        source, worker, protocol, "baseline", "inspect-ai", common.digest(proposal)
    )
    capture = tmp_path / "baseline"
    approved.stage(capture)
    replay = recovery.TaskClient(
        "unused",
        "inspect-ai",
        proposal["protocol_digest"],
        protocol["cases"],
        tmp_path / "replay-journal",
        recovery=approved,
    )
    result = replay(protocol["cases"][0])
    for path in replay.output.iterdir():
        (capture / "tasks" / path.name).write_bytes(path.read_bytes())
    native = common.read(capture / "native.json")
    native["samples"][0]["metadata"] = {
        **protocol["cases"][0]["metadata"],
        **LIVE.bindings.bind_result(
            result, protocol["cases"][0], "inspect-ai", "0.3.254"
        )["metadata"],
    }
    (capture / "native.json").write_bytes(common.encoded(native))
    manifest = common.read(capture / "capture.json")
    manifest["native_sha256"] = recovery.physical(
        (capture / "native.json").read_bytes()
    )
    manifest["transport_recovery"] = {"admission_sha256": approved.digest, **proposal}
    (capture / "capture.json").write_bytes(common.encoded(manifest))
    return LIVE, args, capture, proposal


@pytest.mark.parametrize("route", ["envelope", "native-json"])
def test_independent_recipient_preserves_and_authenticates_recovery(tmp_path, route):
    recipient, args, capture, proposal = recipient_fixture(tmp_path)
    args[5] = route
    _, runs = recipient.prepare(*args)
    assert len(runs[0]["records"]) == 2
    copied = args[6] / "captures/baseline"
    assert (copied / "recovery.json").read_bytes() == (
        capture / "recovery.json"
    ).read_bytes()
    for name in proposal["original_capture_files"]:
        assert (copied / "recovery-original" / name).read_bytes() == (
            capture / "recovery-original" / name
        ).read_bytes()
    replayed = runs[0]["records"][0]
    assert recipient.contains(
        replayed,
        "invarlock_transport_replay",
        recipient.common.read(
            capture
            / "tasks"
            / next(
                name
                for name in proposal["worker_files"]
                if name.endswith(".response.json")
            )
        )["result"]["metadata"]["invarlock_transport_replay"],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "proposal",
        "worker-extra",
        "raw-sha",
        "raw-bytes",
        "restored-output",
        "missing-replay",
        "false-replay",
        "native-replay",
    ],
)
def test_independent_recipient_rejects_changed_recovery_provenance(tmp_path, fault):
    recipient, args, capture, proposal = recipient_fixture(tmp_path)
    common = recipient.common
    if fault == "manifest":
        path = capture / "capture.json"
        value = common.read(path)
        value.pop("transport_recovery")
        path.write_bytes(common.encoded(value))
    elif fault == "proposal":
        path = capture / "recovery.json"
        value = common.read(path)
        value["new_case_ids"] = []
        path.write_bytes(common.encoded(value))
    elif fault == "worker-extra":
        common.write(capture / "recovery-worker/extra.json", {})
    elif fault == "native-replay":
        path = capture / "native.json"
        value = common.read(path)
        value["samples"][0]["metadata"].pop("invarlock_transport_replay")
        path.write_bytes(common.encoded(value))
        manifest = common.read(capture / "capture.json")
        manifest["native_sha256"] = (
            "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        )
        (capture / "capture.json").write_bytes(common.encoded(manifest))
    else:
        path = next(
            path
            for path in (capture / "tasks").glob("*.response.json")
            if common.read(path)["request"]["case_id"]
            == ("1" if fault == "false-replay" else "0")
        )
        value = common.read(path)
        metadata = value["result"]["metadata"]
        if fault == "false-replay":
            metadata["invarlock_transport_replay"] = {}
        elif fault == "missing-replay":
            metadata.pop("invarlock_transport_replay")
        elif fault == "restored-output":
            metadata["changed-observation"] = "injected"
        elif fault == "raw-sha":
            metadata["invarlock_transport_replay"]["original_response_sha256"] = (
                "changed"
            )
        else:
            metadata["invarlock_transport_replay"]["original_response_bytes"] += " "
        path.write_bytes(common.encoded(value))
    with pytest.raises(ValueError, match="recovery|replay|execution metadata"):
        recipient.prepare(*args)
