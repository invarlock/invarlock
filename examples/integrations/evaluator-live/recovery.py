"""Explicitly recover unstarted transport tasks while retaining completed calls."""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path

import common


def physical(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def file_bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("recovery requires ordinary retained files")
    with path.open("rb") as stream:
        raw = stream.read(common.MAX_MESSAGE + 1)
    if len(raw) > common.MAX_MESSAGE:
        raise ValueError("recovery file exceeds its byte limit")
    return raw


def inspect(source, worker_ledger, protocol, role, evaluator):
    source, worker_ledger = Path(source), Path(worker_ledger)
    if any(path.is_symlink() or not path.is_dir() for path in (source, worker_ledger)):
        raise ValueError("recovery requires original capture and worker directories")
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("recovery cannot traverse symbolic links")
        if path.is_file():
            files[str(path.relative_to(source))] = file_bytes(path)
        if len(files) > 1000 or sum(map(len, files.values())) > 32 * 1024 * 1024:
            raise ValueError("original capture exceeds recovery inventory limits")
    expected_protocol = {**protocol, "role": role}
    if (
        common.encoded(common.decode(files["protocol.json"]))
        != common.encoded(expected_protocol)
        or evaluator not in protocol["evaluators"]
        or role not in {"baseline", "subject"}
        or "capture.json" in files
    ):
        raise ValueError("recovery requires the exact incomplete original capture")
    failure = common.decode(files["failure.json"])
    if failure.get("status") != "capture_failed":
        raise ValueError("recovery requires a retained capture failure")
    responses, worker_files, fresh, expected_tasks = {}, {}, [], set()
    for case in common.cases(protocol["cases"]):
        request = {
            "evaluator": evaluator,
            "case_id": case["id"],
            "protocol_digest": common.digest(expected_protocol),
        }
        stem = common.digest(request).removeprefix("sha256:")
        request_name, response_name = stem + ".request.json", stem + ".response.json"
        expected_tasks.add("tasks/" + request_name)
        if common.encoded(
            common.decode(files["tasks/" + request_name])
        ) != common.encoded(request):
            raise ValueError("original task admission differs from frozen request")
        original = files.get("tasks/" + response_name)
        if original is None:
            if any(
                (worker_ledger / name).exists() or (worker_ledger / name).is_symlink()
                for name in (request_name, response_name)
            ):
                raise ValueError(
                    "unfinished model task was already admitted; recovery cannot rerun it"
                )
            fresh.append(case["id"])
        else:
            expected_tasks.add("tasks/" + response_name)
            for name in (request_name, response_name):
                worker_files[name] = file_bytes(worker_ledger / name)
                if worker_files[name] != files["tasks/" + name]:
                    raise ValueError(
                        "retained task differs from durable worker evidence"
                    )
            value = common.decode(original)
            if set(value) != {"request", "result"} or common.encoded(
                value["request"]
            ) != common.encoded(request):
                raise ValueError("retained response differs from its admitted request")
            if "invarlock_transport_replay" in value["result"]["metadata"]:
                raise ValueError(
                    "nested recovery is outside this bounded transport repair"
                )
            responses[case["id"]] = original
    if (
        {name for name in files if name.startswith("tasks/")} != expected_tasks
        or set(failure["completed_case_ids"]) != set(responses)
        or not responses
        or not fresh
    ):
        raise ValueError("recovery requires an exact partial task inventory")
    proposal = {
        "format": "invarlock/live-transport-recovery-v1",
        "evaluator": evaluator,
        "role": role,
        "protocol_digest": common.digest(expected_protocol),
        "replayed_case_ids": sorted(responses),
        "new_case_ids": sorted(fresh),
        "original_capture_files": {name: physical(raw) for name, raw in files.items()},
        "worker_files": {name: physical(raw) for name, raw in worker_files.items()},
        "unstarted_worker_requests_absent": sorted(fresh),
    }
    return proposal, files, worker_files, responses


class Recovery:
    def __init__(self, source, worker_ledger, protocol, role, evaluator, expected):
        self.proposal, self.files, self.worker_files, self.responses = inspect(
            source, worker_ledger, protocol, role, evaluator
        )
        self.digest = common.digest(self.proposal)
        if self.digest != expected:
            raise ValueError("recovery differs from its independently admitted digest")

    def stage(self, output):
        common.write(output / "recovery.json", self.proposal)
        for prefix, files in (
            ("recovery-original", self.files),
            ("recovery-worker", self.worker_files),
        ):
            for name, raw in files.items():
                path = output / prefix / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(raw)


class TaskClient(common.TaskClient):
    def __init__(self, *args, recovery):
        super().__init__(*args)
        self.recovery = recovery

    def _call(self, case):
        ident = case.get("id")
        if ident not in self.recovery.responses:
            return super()._call(case)
        if ident in self.results or common.encoded(case) != common.encoded(
            self.planned[ident]
        ):
            raise ValueError("SDK replay differs from the admitted case or repeats it")
        original = self.recovery.responses[ident]
        value = common.decode(original)
        value["result"]["metadata"]["invarlock_transport_replay"] = {
            "format": "invarlock/live-retained-task-replay-v1",
            "recovery_sha256": self.recovery.digest,
            "original_response_sha256": physical(original),
            "original_response_bytes": original.decode("utf-8"),
            "execution": "Retained model response replayed into a new SDK capture; no new model call",
        }
        stem = common.digest(value["request"]).removeprefix("sha256:")
        common.write(self.output / (stem + ".request.json"), value["request"])
        common.write(self.output / (stem + ".response.json"), value)
        self.results[ident] = deepcopy(value["result"])
        return deepcopy(value["result"])


def verify_capture(directory, manifest, protocol, role, evaluator, results, originals):
    """Authenticate replay provenance and retain the entire failed SDK attempt."""
    directory = Path(directory)
    admitted = manifest.get("transport_recovery")
    if not isinstance(admitted, dict) or "admission_sha256" not in admitted:
        raise ValueError("replayed tasks require their independently admitted recovery")
    proposal, files, worker_files, responses = inspect(
        directory / "recovery-original",
        directory / "recovery-worker",
        protocol,
        role,
        evaluator,
    )
    raw = file_bytes(directory / "recovery.json")
    if (
        common.encoded(common.decode(raw)) != common.encoded(proposal)
        or common.encoded(admitted)
        != common.encoded({"admission_sha256": common.digest(proposal), **proposal})
        or {
            str(path.relative_to(directory / "recovery-worker"))
            for path in (directory / "recovery-worker").rglob("*")
            if path.is_file()
        }
        != set(worker_files)
    ):
        raise ValueError("retained recovery proposal or worker inventory changed")
    for ident, result in results.items():
        metadata = result["metadata"]
        if ident not in responses:
            if "invarlock_transport_replay" in metadata:
                raise ValueError("new model task falsely claims a retained replay")
            continue
        original = responses[ident]
        replay = metadata.get("invarlock_transport_replay")
        expected = {
            "format": "invarlock/live-retained-task-replay-v1",
            "recovery_sha256": common.digest(proposal),
            "original_response_sha256": physical(original),
            "original_response_bytes": original.decode("utf-8"),
            "execution": "Retained model response replayed into a new SDK capture; no new model call",
        }
        restored = deepcopy(result)
        restored["metadata"].pop("invarlock_transport_replay", None)
        if common.encoded(replay) != common.encoded(expected) or common.encoded(
            restored
        ) != common.encoded(common.decode(original)["result"]):
            raise ValueError(
                "replayed model result differs from unchanged original bytes"
            )
    originals["recovery.json"] = raw
    for prefix, retained in (
        ("recovery-original", files),
        ("recovery-worker", worker_files),
    ):
        originals.update({prefix + "/" + name: raw for name, raw in retained.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recover-from", "worker-ledger", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    parser.add_argument("--evaluator", required=True)
    args = parser.parse_args()
    proposal, *_ = inspect(
        args.recover_from,
        args.worker_ledger,
        common.read(args.protocol),
        args.role,
        args.evaluator,
    )
    print(
        common.encoded(
            {"proposal": proposal, "recovery_sha256": common.digest(proposal)}
        ).decode()
    )


if __name__ == "__main__":
    main()
