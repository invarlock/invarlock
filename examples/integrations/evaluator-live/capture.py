"""Run one actual evaluator workflow against an admitted local model worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import time
from pathlib import Path

import bindings
import common

GROUPS = {
    "scalar": (
        "deepeval",
        "ragas",
        "hugging-face-evaluate",
        "autoevals",
        "openevals",
        "arize-phoenix-evals",
        "opik",
    ),
    "batch": (
        "pydantic-evals",
        "azure-ai-evaluation",
        "evidently",
        "mlflow",
        "trulens",
    ),
    "harness": (
        "lm-evaluation-harness",
        "inspect-ai",
        "promptfoo",
        "lighteval",
        "garak",
        "openai-evals",
        "langfuse",
    ),
}


def local_environment(evaluator):
    common.module("network").configure(evaluator)


def capture(
    protocol,
    role,
    evaluator,
    socket_path,
    output,
    *,
    recover_from=None,
    worker_ledger=None,
    recovery_sha256=None,
):
    if evaluator not in protocol["evaluators"] or role not in {"baseline", "subject"}:
        raise ValueError("evaluator or role is outside the admitted campaign")
    if set(protocol["versions"]) != set(protocol["evaluators"]):
        raise ValueError("every admitted evaluator requires a version pin")
    expected = common.versions()[evaluator]
    if protocol["versions"][evaluator] != expected:
        raise ValueError("campaign SDK pin differs from the maintained profile")
    package = {
        "lm-evaluation-harness": "lm-eval",
        "hugging-face-evaluate": "evaluate",
        "openai-evals": "evals",
    }.get(evaluator, evaluator)
    if evaluator == "promptfoo":
        actual = common.read(
            Path(os.environ["INVARLOCK_PROMPTFOO_PACKAGE"]) / "package.json"
        )["version"]
    else:
        actual = importlib.metadata.version(package)
    if actual != expected:
        raise ValueError("installed SDK version differs from the campaign")
    recovery = None
    recovery_module = None
    if any(
        value is not None for value in (recover_from, worker_ledger, recovery_sha256)
    ):
        if any(
            value is None for value in (recover_from, worker_ledger, recovery_sha256)
        ):
            raise ValueError(
                "recovery requires original capture, worker ledger, and independent digest"
            )
        recovery_module = common.module("recovery")
        recovery = recovery_module.Recovery(
            recover_from, worker_ledger, protocol, role, evaluator, recovery_sha256
        )
    local_environment(evaluator)
    group = next(name for name, entries in GROUPS.items() if evaluator in entries)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    worker_protocol = {**protocol, "role": role}
    client = recovery_module.TaskClient if recovery else common.TaskClient
    task = client(
        socket_path,
        evaluator,
        common.digest(worker_protocol),
        protocol["cases"],
        output / "tasks",
        **({"recovery": recovery} if recovery else {}),
    )
    if recovery:
        recovery.stage(output)
    workdir = output / "sdk"
    workdir.mkdir()
    started = time.time()
    common.write(output / "protocol.json", worker_protocol)
    driver = common.module(group)
    try:

        def model_task(case):
            return bindings.bind_result(task(case), case, evaluator, actual)

        payload = driver.run(evaluator, protocol["cases"], model_task, workdir)
        results = task.complete()
        common.write(output / "native.json", payload)
    except Exception as exc:
        common.write(
            output / "failure.json",
            {
                "status": "capture_failed",
                "exception_type": type(exc).__name__,
                "completed_case_ids": list(task.results),
                "started_unix": started,
                "finished_unix": time.time(),
            },
        )
        raise
    paths = {
        name: common.HERE / (name + ".py")
        for name in ("capture", "common", "bindings", "network", group)
    }
    if recovery:
        paths["recovery"] = common.HERE / "recovery.py"
    manifest = {
        "format": "invarlock/live-evaluator-capture-v1",
        "status": "captured",
        "evaluator": evaluator,
        "version": actual,
        "role": role,
        "protocol_digest": common.digest(worker_protocol),
        "native_sha256": "sha256:"
        + hashlib.sha256((output / "native.json").read_bytes()).hexdigest(),
        "case_count": len(results),
        "started_unix": started,
        "finished_unix": time.time(),
        "model": protocol["models"][role],
        "driver_files": {
            name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
        "qualification": "pending independent recipient and campaign audit",
    }
    if recovery:
        manifest["transport_recovery"] = {
            "admission_sha256": recovery.digest,
            **recovery.proposal,
        }
    common.write(output / "capture.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    parser.add_argument("--evaluator", choices=common.versions(), required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--worker-ledger", type=Path)
    parser.add_argument("--recovery-sha256")
    args = parser.parse_args()
    protocol = common.read(args.protocol)
    if common.digest(protocol) != args.protocol_sha256:
        raise ValueError("protocol differs from its independently approved digest")
    print(
        common.encoded(
            capture(
                protocol,
                args.role,
                args.evaluator,
                args.socket,
                args.output,
                **(
                    {
                        "recover_from": args.recover_from,
                        "worker_ledger": args.worker_ledger,
                        "recovery_sha256": args.recovery_sha256,
                    }
                    if any(
                        value is not None
                        for value in (
                            args.recover_from,
                            args.worker_ledger,
                            args.recovery_sha256,
                        )
                    )
                    else {}
                ),
            )
        ).decode()
    )


if __name__ == "__main__":
    main()
