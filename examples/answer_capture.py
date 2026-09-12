"""Bounded answer capture at a user-owned pipeline boundary (POSIX example)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Protocol

from invarlock.captured_contracts import read_file, secure_directory
from invarlock.evaluation_record_contracts.contracts import validate
from invarlock.evaluation_records.cases import canonical_case_set, case_set_digest
from invarlock.evaluation_records.io import physical_file_digest, run_digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace


class Adapter(Protocol):
    """A trusted integration; requests never include reference answers."""

    def count_input_tokens(self, request: dict[str, Any]) -> int: ...

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]: ...


def read(path: Path) -> dict[str, Any]:
    value = parse_json_bytes(read_file(path, 16 * 1024 * 1024), label=str(path))
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    return value


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def exact(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")


def prepare(
    config: dict[str, Any], cases: dict[str, Any], adapter_sha256: str, adapter: Adapter
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze membership and reserve the entire workload before generating."""
    exact(config, {"baseline", "subject", "limits"}, "configuration")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", adapter_sha256) is None:
        raise ValueError("adapter source digest is required")
    limits = config["limits"]
    exact(
        limits,
        {
            "concurrency",
            "deadline_unix_seconds",
            "call_timeout_seconds",
            "max_calls",
            "max_input_tokens_per_call",
            "max_output_tokens_per_call",
            "max_output_bytes_per_call",
            "max_total_input_tokens",
            "max_total_output_tokens",
            "max_cost_microusd",
            "input_microusd_per_million_tokens",
            "output_microusd_per_million_tokens",
        },
        "limits",
    )
    for key, value in limits.items():
        positive(value, key)
    if limits["max_output_bytes_per_call"] > 1024 * 1024:
        raise ValueError(
            "per-call output byte cap exceeds this example's 1 MiB ceiling"
        )
    cases = canonical_case_set(cases)
    jobs = []
    for side in ("baseline", "subject"):
        model = config[side]
        exact(
            model,
            {
                "provider",
                "model",
                "revision",
                "tokenizer",
                "artifact_digest",
                "generation",
            },
            side,
        )
        if not all(
            isinstance(model[key], str) and model[key]
            for key in ("provider", "model", "revision", "tokenizer")
        ):
            raise ValueError(
                "provider, model, revision and tokenizer identities are required"
            )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(model["artifact_digest"])) is None:
            raise ValueError("deployment artifact_digest must be a sha256 digest")
        if not isinstance(model["generation"], dict):
            raise ValueError("generation must be an explicit configuration object")
        for case in cases["cases"]:
            request = {
                "side": side,
                "case_id": case["id"],
                "input": case["input"],
                "model": model,
                "max_output_tokens": limits["max_output_tokens_per_call"],
            }
            tokens = positive(
                adapter.count_input_tokens(copy.deepcopy(request)), "input token count"
            )
            if tokens > limits["max_input_tokens_per_call"]:
                raise ValueError("rendered request exceeds per-call input token cap")
            jobs.append({"request": request, "input_tokens": tokens})
    count = len(jobs)
    if not count or count > limits["max_calls"]:
        raise ValueError("case membership exceeds call cap or is empty")
    input_reserve = count * limits["max_input_tokens_per_call"]
    output_reserve = count * limits["max_output_tokens_per_call"]
    numerator = (
        input_reserve * limits["input_microusd_per_million_tokens"]
        + output_reserve * limits["output_microusd_per_million_tokens"]
    )
    cost = (numerator + 999_999) // 1_000_000
    if (
        input_reserve > limits["max_total_input_tokens"]
        or output_reserve > limits["max_total_output_tokens"]
        or cost > limits["max_cost_microusd"]
    ):
        raise ValueError("full-plan token or cost reservation exceeds budget")
    manifest = {
        "format": "answer-capture-example-v1",
        "config": config,
        "case_set": cases,
        "adapter_sha256": adapter_sha256,
        "jobs": jobs,
        "reservation": {
            "calls": count,
            "input_tokens": input_reserve,
            "output_tokens": output_reserve,
            "cost_microusd": cost,
        },
    }
    if hasattr(adapter, "identity"):
        manifest["adapter_identity"] = copy.deepcopy(adapter.identity)
    manifest_bytes = len(canonical_json_bytes(manifest))
    # Six bytes per output byte covers worst-case JSON escaping. Each answer is
    # retained twice: once in its result journal and once in an evaluation run.
    # The remaining reservation covers duplicated identities, case fields and
    # journal wrappers.
    storage_reserve = 2 * manifest_bytes + count * (
        12 * limits["max_output_bytes_per_call"] + 4096
    )
    if manifest_bytes > 16 * 1024 * 1024 or storage_reserve > 128 * 1024 * 1024:
        raise ValueError(
            "workload exceeds this example's bounded JSON storage envelope"
        )
    return copy.deepcopy(manifest), copy.deepcopy(jobs)


def check_result(value: Any, job: dict[str, Any], limits: dict[str, int]) -> None:
    exact(value, {"output", "input_tokens", "output_tokens"}, "adapter result")
    if (
        not isinstance(value["output"], str)
        or len(value["output"].encode()) > limits["max_output_bytes_per_call"]
    ):
        raise ValueError("output must be a string within the admitted byte cap")
    if (
        value["input_tokens"] != job["input_tokens"]
        or type(value["input_tokens"]) is not int
    ):
        raise ValueError("provider input usage differs from prepared tokenizer count")
    if type(value["output_tokens"]) is not int or value["output_tokens"] < 0:
        raise ValueError("output token usage must be a nonnegative integer")
    if value["output_tokens"] > limits["max_output_tokens_per_call"]:
        raise ValueError("provider exceeded output token cap")


def retain(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        if read_file(path, 128 * 1024 * 1024) != payload:
            raise ValueError(f"retained {path.name} differs from frozen workload")
    else:
        write_file_no_replace(path, payload, create_parents=False)


async def capture(
    *,
    config: dict[str, Any],
    cases: dict[str, Any],
    adapter_sha256: str,
    adapter: Adapter,
    directory: Path,
) -> dict[str, str]:
    """Resume completed results only; an unresolved dispatched attempt blocks all calls."""
    manifest, jobs = prepare(config, cases, adapter_sha256, adapter)
    limits = manifest["config"]["limits"]
    with secure_directory(directory, create=True) as parent:
        lock = os.open(
            ".lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
            dir_fd=parent,
        )
        try:
            if not stat.S_ISREG(os.fstat(lock).st_mode):
                raise ValueError("checkpoint lock must be a regular file")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return await _capture_locked(manifest, jobs, limits, adapter, directory)
        finally:
            os.close(lock)


async def _capture_locked(
    manifest: dict[str, Any],
    jobs: list[dict[str, Any]],
    limits: dict[str, int],
    adapter: Adapter,
    directory: Path,
) -> dict[str, str]:
    allowed = {".lock", "manifest.json", "baseline_run.json", "subject_run.json"}
    allowed |= {
        f"{index:06}.{suffix}.json"
        for index in range(len(jobs))
        for suffix in ("attempt", "result")
    }
    if set(os.listdir(directory)) - allowed:
        raise ValueError("checkpoint contains unexpected files")
    retain(directory / "manifest.json", manifest)
    manifest_digest = digest(manifest)
    completed = {}
    for index, job in enumerate(jobs):
        attempt = directory / f"{index:06}.attempt.json"
        result = directory / f"{index:06}.result.json"
        if result.exists() or result.is_symlink():
            if not attempt.exists():
                raise ValueError("result is missing its retained dispatch attempt")
            retain(attempt, {"manifest_sha256": manifest_digest, "job": index})
            retained = read(result)
            exact(retained, {"manifest_sha256", "job", "result"}, "retained result")
            if (
                retained["manifest_sha256"] != manifest_digest
                or type(retained["job"]) is not int
                or retained["job"] != index
            ):
                raise ValueError("result identity differs from dispatch")
            check_result(retained["result"], job, limits)
            completed[index] = retained["result"]
        elif attempt.exists() or attempt.is_symlink():
            raise ValueError(
                "ambiguous in-flight attempt: automatic retry is prohibited"
            )
    if len(completed) != len(jobs) and any(
        (directory / name).exists() or (directory / name).is_symlink()
        for name in ("baseline_run.json", "subject_run.json")
    ):
        raise ValueError("published run exists without complete retained results")
    if len(completed) != len(jobs) and time.time() >= limits["deadline_unix_seconds"]:
        raise ValueError("fixed campaign deadline has elapsed")
    semaphore = asyncio.Semaphore(limits["concurrency"])

    async def run(index, job):
        if index in completed:
            return
        async with semaphore:
            remaining = limits["deadline_unix_seconds"] - time.time()
            if remaining <= 0:
                raise ValueError("fixed campaign deadline has elapsed")
            write_file_no_replace(
                directory / f"{index:06}.attempt.json",
                canonical_json_bytes(
                    {"manifest_sha256": manifest_digest, "job": index}
                ),
                create_parents=False,
            )
            async with asyncio.timeout(min(remaining, limits["call_timeout_seconds"])):
                result = await adapter.generate(copy.deepcopy(job["request"]))
            check_result(result, job, limits)
            write_file_no_replace(
                directory / f"{index:06}.result.json",
                canonical_json_bytes(
                    {"manifest_sha256": manifest_digest, "job": index, "result": result}
                ),
                create_parents=False,
            )
            completed[index] = result

    async with asyncio.TaskGroup() as group:
        for index, job in enumerate(jobs):
            group.create_task(run(index, job))
    runs = {}
    for side in ("baseline", "subject"):
        records = []
        for index, job in enumerate(jobs):
            if job["request"]["side"] != side:
                continue
            case = manifest["case_set"]["cases"][
                index % len(manifest["case_set"]["cases"])
            ]
            records.append(
                {
                    **case,
                    "output": completed[index]["output"],
                    "error": None,
                    "scores": {},
                    "context": {
                        "capture_manifest_sha256": manifest_digest,
                        "model_identity": manifest["config"][side],
                        "usage": {
                            key: completed[index][key]
                            for key in ("input_tokens", "output_tokens")
                        },
                    },
                }
            )
        value = {
            "format": "invarlock/evaluation-run-v1",
            "run_id": side,
            "artifact_digest": manifest["config"][side]["artifact_digest"],
            "source": {"name": "answer-capture-example", "version": "1"},
            "source_digest": manifest_digest,
            "score_provenance": {},
            "records": records,
        }
        validate(value, "run")
        retain(directory / f"{side}_run.json", value)
        runs[side] = run_digest(value)
    return {**runs, "case_set": case_set_digest(manifest["case_set"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    adapter_options = parser.add_mutually_exclusive_group(required=True)
    adapter_options.add_argument(
        "--adapter",
        help="trusted importable module exposing count_input_tokens and generate",
    )
    adapter_options.add_argument(
        "--transport", type=Path, help="local JSON process transport configuration"
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="authorize this invocation's bounded adapter calls",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("review the frozen inputs and budget, then supply --execute")
    try:
        config = read(args.config)
        if args.transport:
            from examples.answer_capture_process import ProcessAdapter

            adapter = ProcessAdapter(args.transport, config["limits"])
            adapter_sha256 = adapter.sha256
        else:
            adapter = importlib.import_module(args.adapter)
            if not adapter.__file__:
                parser.error("adapter must have a source file")
            adapter_sha256 = physical_file_digest(adapter.__file__)
        result = asyncio.run(
            capture(
                config=config,
                cases=read(args.cases),
                adapter_sha256=adapter_sha256,
                adapter=adapter,
                directory=args.directory,
            )
        )
    except (OSError, ValueError, ExceptionGroup) as exc:
        parser.exit(
            2,
            f"Answer capture stopped: {exc}\nInspect retained attempts before any new capture.\n",
        )
    print("Frozen answer runs ready: " + str(result))


if __name__ == "__main__":
    main()
