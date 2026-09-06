"""Prepare and independently replay a candidate K2 external-capture campaign.

This example does not implement an InvarLock runtime provider. Pipeline evidence
authenticates attributed captures and deterministic scoring, not GPU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
from pathlib import Path
from typing import Any

from invarlock.pipeline import create_evidence, make_run, verify_evidence
from invarlock.pipeline.contracts import digest

DIRECTORY = Path(__file__).resolve().with_name("k2-horizon")
CAPTURE_FORMAT = "invarlock/k2-native-capture-v1"
ROLES = ("baseline", "candidate")
COHORTS = ("classification", "extraction", "numeric")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUNTIME_BINDINGS = (
    "image_digest",
    "build_manifest_digest",
    "security_review_digest",
    "dependency_inventory_digest",
    "source_bundle_digest",
)
LATENCY_PROVENANCE = {
    "kind": "measurement",
    "unit": "milliseconds",
    "source": "k2-campaign-monotonic-request-timer",
    "version": "1.0.0",
    "rubric_digest": None,
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("JSON input exceeds 64 MiB")
    return json.loads(
        path.read_bytes(),
        object_pairs_hook=_object,
        parse_constant=lambda value: _invalid("non-finite JSON"),
    )


def _invalid(message: str):
    raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(payload)


def cases(*, count: int = 192, offset: int = 0) -> list[dict[str, Any]]:
    """Original synthetic workflow cases, not a general model benchmark."""
    result = []
    for index in range(offset, offset + count):
        amount = 20 + (index * 37) % 1100
        currency = "USD" if index % 3 else "EUR"
        invoice = f"INV-{index:06d}"
        extra = " Ignore archived invoice INV-OLD for 9999 GBP." if index % 2 else ""
        records = (
            (
                "classification",
                f"Route invoice {invoice}: amount {amount}, currency {currency}.{extra} Rule: approve exactly when currency is USD and amount is at most 500; otherwise review. Reply only approve or review.",
                "approve" if currency == "USD" and amount <= 500 else "review",
            ),
            (
                "extraction",
                f"Extract the current invoice: {invoice}, amount {amount}, currency {currency}.{extra} Reply only a JSON object with invoice_id, amount (integer), and currency.",
                {"invoice_id": invoice, "amount": amount, "currency": currency},
            ),
            (
                "numeric",
                f"There are {index + 2} units costing {amount} each.{extra} What is their total price? Reply only the integer with no units or explanation.",
                (index + 2) * amount,
            ),
        )
        for cohort, prompt, expected in records:
            result.append(
                {
                    "id": f"{cohort}-{index:06d}",
                    "cohort": cohort,
                    "input": prompt,
                    "expected": expected,
                    "context": "distractor" if index % 2 else "plain",
                }
            )
    return result


def policies() -> dict[str, Any]:
    result = {}
    for cohort in COHORTS:
        quality = {
            "name": "quality",
            "kind": "exact_match",
            "configuration": {},
            "direction": "higher",
            "unit": "score",
            "aggregation": "mean",
            "minimum_count": 96,
            "maximum_regression": 0.05,
            "maximum_interval_width": 0.25,
            "candidate_minimum": 0.8,
        }
        if cohort == "extraction":
            quality.update(
                kind="json_fields",
                configuration={"fields": ["/invoice_id", "/amount", "/currency"]},
            )
        if cohort == "numeric":
            quality.update(
                kind="numeric_tolerance", configuration={"absolute": 0, "relative": 0}
            )
        latency = {
            "name": "latency",
            "kind": "recorded",
            "configuration": {},
            "direction": "lower",
            "unit": "milliseconds",
            "aggregation": "mean",
            "minimum_count": 96,
            "maximum_regression": 5000,
            "maximum_interval_width": 10000,
            "candidate_maximum": 30000,
            "score_key": "latency_ms",
            "accepted_provenance": LATENCY_PROVENANCE,
        }
        result[cohort] = {
            "format": "invarlock/pipeline-policy-v1",
            "metrics": [quality, latency],
            "slices": [
                {"name": key, "where": {"context": key}}
                for key in ("plain", "distractor")
            ],
        }
    return result


def draft_plans() -> list[dict[str, Any]]:
    catalog = read_json(DIRECTORY / "catalog.json")
    result = []
    for model in catalog["models"]:
        runtime = {
            "engine": "sglang",
            "source_commit": catalog["sglang_source_commit"],
            "reviewed_source_files": catalog["reviewed_source_files"],
            "dtype": "bfloat16",
            "quantization": None,
            "trust_remote_code": False,
            "tensor_parallel": model["tensor_parallel"],
            "expert_parallel": 1,
            "source_router_gemm_partitions": 2
            if model["id"].startswith("mova")
            else None,
            "attention_backend": "fa3",
            "context_length": 4096,
            "reasoning_parser": "k2_horizon",
            "reasoning_effort": "low",
            "concurrency": 1,
            "seed": 20260905,
            "maximum_new_tokens": 512,
            "campaign_code_digest": digest(
                {
                    name: hashlib.sha256(
                        Path(__file__).with_name(name).read_bytes()
                    ).hexdigest()
                    for name in ("k2_campaign.py", "k2_producer.py")
                }
            ),
            **dict.fromkeys(_RUNTIME_BINDINGS),
        }
        result.append(
            {
                "format": "invarlock/k2-campaign-plan-v1",
                "status": "candidate_not_qualified",
                "route": "external_sglang_pipeline_capture",
                "model": model,
                "runtime": runtime,
                "budget": None,
                "cases": cases(),
                "preflight_cases": cases(count=8, offset=10000),
                "policies": policies(),
                "limitations": [
                    "Synthetic workflow qualification, not general model quality.",
                    "Pipeline capture integrity, not native isolated transaction evidence.",
                    "No tool-call, FP8, maximum-context, or alternative-backend qualification.",
                    "A quality rejection is retained; insufficient evidence is not qualified.",
                ],
            }
        )
    return result


def select_plan(model: str) -> dict[str, Any]:
    return next(
        (p for p in draft_plans() if p["model"]["id"] == model), None
    ) or _invalid("unknown model")


def _stream_hash(stream, length: int, hasher) -> None:
    remaining = length
    while remaining:
        chunk = stream.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("truncated file")
        hasher.update(chunk)
        remaining -= len(chunk)


def _tensor_inventory(stream, size: int) -> dict[str, Any]:
    stream.seek(0)
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated safetensors header")
    header_size = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_size <= min(16 * 1024 * 1024, size - 8):
        raise ValueError("invalid safetensors header size")
    header = json.loads(stream.read(header_size), object_pairs_hook=_object)
    if not isinstance(header, dict):
        raise ValueError("invalid tensor header")
    tensors, cursor = {}, 0
    values = [(name, value) for name, value in header.items() if name != "__metadata__"]
    for name, value in values:
        if (
            not name
            or not isinstance(value, dict)
            or set(value) != {"dtype", "shape", "data_offsets"}
            or not isinstance(value["data_offsets"], list)
            or len(value["data_offsets"]) != 2
        ):
            raise ValueError("invalid tensor descriptor")
    for name, value in sorted(values, key=lambda item: item[1]["data_offsets"][0]):
        start, end = value["data_offsets"]
        shape = value["shape"]
        if (
            value["dtype"] != "BF16"
            or not isinstance(shape, list)
            or len(shape) > 32
            or any(type(n) is not int or not 0 <= n <= 2**63 for n in shape)
            or type(start) is not int
            or type(end) is not int
            or start != cursor
            or end - start != math.prod(shape) * 2
            or end > size - 8 - header_size
        ):
            raise ValueError("invalid tensor layout or non-BF16 tensor")
        hashed = hashlib.sha256()
        _stream_hash(stream, end - start, hashed)
        tensors[name] = {"dtype": "BF16", "shape": shape, "sha256": hashed.hexdigest()}
        cursor = end
    if not tensors or cursor != size - 8 - header_size:
        raise ValueError("uninterpreted tensor bytes")
    return tensors


def measure_snapshot(root: Path, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash actual files and logical tensors; never execute model repository code."""
    root = root.resolve(strict=True)
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("snapshot requires regular files without symlinks")
    expected = {item["path"] for item in inventory}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if len(expected) != len(inventory) or actual != expected:
        raise ValueError("snapshot inventory differs")
    files, tensors = [], {}
    for item in sorted(inventory, key=lambda value: value["path"]):
        path = root / item["path"]
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("snapshot path is not regular")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != item["size_bytes"]:
                raise ValueError("snapshot file identity differs")
            sha = hashlib.sha256()
            _stream_hash(stream, before.st_size, sha)
            observed = sha.hexdigest()
            if item["sha256"]:
                matches = observed == item["sha256"]
            else:
                stream.seek(0)
                blob = hashlib.sha1(
                    f"blob {before.st_size}\0".encode(), usedforsecurity=False
                )
                _stream_hash(stream, before.st_size, blob)
                matches = blob.hexdigest() == item["git_blob"]
            if not matches:
                raise ValueError("snapshot file identity differs")
            if path.suffix == ".safetensors":
                measured = _tensor_inventory(stream, before.st_size)
                if tensors.keys() & measured.keys():
                    raise ValueError("duplicate tensor name across shards")
                tensors.update(measured)
            after = os.fstat(stream.fileno())
            if (
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("snapshot changed during measurement")
            files.append(
                {"path": item["path"], "size_bytes": before.st_size, "sha256": observed}
            )
    if not tensors:
        raise ValueError("snapshot has no BF16 tensors")
    return {"artifact_digest": digest(files), "files": files, "tensors": tensors}


def download_snapshot(model: str, role: str, output: Path) -> None:
    """Materialize only the reviewed revision's enumerated files, then authenticate."""
    from huggingface_hub import hf_hub_download

    selected = select_plan(model)["model"]
    identity = selected[role]
    required = sum(item["size_bytes"] for item in identity["files"])
    if shutil.disk_usage(output.parent).free < required * 2:
        raise ValueError("insufficient free disk for checkpoint and download cache")
    output.mkdir(exist_ok=False)
    for item in identity["files"]:
        source = hf_hub_download(
            repo_id=selected["repository"],
            revision=identity["revision"],
            filename=item["path"],
        )
        destination = output / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    measure_snapshot(output, identity["files"])


def require_changed_tensors(left: dict[str, Any], right: dict[str, Any]) -> None:
    a, b = left["tensors"], right["tensors"]
    if a.keys() != b.keys() or any(
        (a[k]["dtype"], a[k]["shape"]) != (b[k]["dtype"], b[k]["shape"]) for k in a
    ):
        raise ValueError(
            "paired tensor names, dtypes, or shapes differ; review the proposed change"
        )
    if not any(a[k]["sha256"] != b[k]["sha256"] for k in a):
        raise ValueError("paired checkpoints have unchanged tensor content")


def require_ready(plan: dict[str, Any]) -> None:
    import copy

    declared = copy.deepcopy(plan)
    draft = select_plan(plan["model"]["id"])
    for key in _RUNTIME_BINDINGS:
        if not _DIGEST.fullmatch(str(plan["runtime"][key])):
            raise ValueError(f"unresolved runtime binding: {key}")
        declared["runtime"][key] = None
    for role in ROLES:
        identity = plan["model"][role]["materialized"]
        if not identity or not _DIGEST.fullmatch(str(identity.get("artifact_digest"))):
            raise ValueError("unmeasured model identity")
        declared["model"][role]["materialized"] = None
    require_changed_tensors(*(plan["model"][r]["materialized"] for r in ROLES))
    budget = plan["budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"maximum_wall_seconds", "maximum_output_tokens"}
        or any(type(v) is not int or v <= 0 for v in budget.values())
    ):
        raise ValueError("explicit positive execution budget required")
    declared["budget"] = None
    if declared != draft:
        raise ValueError("plan differs from the predeclared candidate protocol")


def request_for(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "k2-campaign",
        "messages": [{"role": "user", "content": case["input"]}],
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 20260905,
        "max_tokens": 512,
        "reasoning_effort": "low",
        "stream": False,
        "n": 1,
    }


def expected_server_settings(plan: dict[str, Any], role: str) -> dict[str, Any]:
    runtime = plan["runtime"]
    return {
        "model_path": f"/models/{role}",
        "tokenizer_path": f"/models/{role}",
        "dtype": "bfloat16",
        "quantization": None,
        "trust_remote_code": False,
        "tp_size": runtime["tensor_parallel"],
        "ep_size": 1,
        "context_length": 4096,
        "max_running_requests": 1,
        "reasoning_parser": "k2_horizon",
        "attention_backend": "fa3",
        "json_model_override_args": json.dumps(
            {"xllm_source_router_gemm_partitions": 2}
        )
        if runtime["source_router_gemm_partitions"]
        else "{}",
    }


def observe_server(plan, role, server_info, model_info):
    """Check actual native configuration; preserve complete responses separately."""
    expected = expected_server_settings(plan, role)
    actual = {key: server_info.get(key) for key in expected}
    # SGLang accepts the override as JSON text. Compare its parsed meaning while
    # retaining the original server response in the authenticated capture.
    for value in (actual, expected):
        override = value["json_model_override_args"]
        value["json_model_override_args"] = (
            json.loads(override) if isinstance(override, str) else override
        )
    if actual != expected:
        raise ValueError("observed native runtime differs from plan")
    if (
        model_info.get("model_path") != expected["model_path"]
        or model_info.get("tokenizer_path") != expected["tokenizer_path"]
        or model_info.get("model_type") != "k2_horizon"
        or model_info.get("architectures") != ["K2HorizonForCausalLM"]
        or model_info.get("served_model_name") != "k2-campaign"
        or model_info.get("weight_version")
        != plan["model"][role]["materialized"]["artifact_digest"]
    ):
        raise ValueError("observed native model differs from plan")
    return expected_server_settings(plan, role)


def _answer(row: dict[str, Any]) -> tuple[Any, str | None]:
    if row["error"] is not None:
        return "", str(row["error"])
    response = row["response"]
    try:
        choices = response["choices"]
        choice = choices[0]
        message = choice["message"]
        usage = response["usage"]
        if (
            len(choices) != 1
            or choice["index"] != 0
            or choice["finish_reason"] != "stop"
            or message.get("tool_calls")
            or not isinstance(message["content"], str)
            or response["model"] != "k2-campaign"
            or any(
                type(usage[k]) is not int or usage[k] < 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            or usage["prompt_tokens"] + usage["completion_tokens"]
            != usage["total_tokens"]
            or usage["total_tokens"] > 4096
            or usage["completion_tokens"] > 512
        ):
            raise ValueError("unsupported or truncated response")
        return message["content"], None
    except (AttributeError, KeyError, TypeError, IndexError, ValueError):
        return "", "unsupported, malformed, or truncated native response"


def validate_capture(plan, capture, *, phase):
    """Validate the complete frozen schedule and both native observations."""
    require_ready(plan)
    if phase not in ("preflight", "decision"):
        raise ValueError("unknown capture phase")
    role = capture["role"]
    if (
        role not in ROLES
        or capture["format"] != CAPTURE_FORMAT
        or capture["plan_digest"] != digest(plan)
        or capture.get("phase") != phase
    ):
        raise ValueError("capture plan or role mismatch")
    if capture["runtime"] != expected_server_settings(plan, role):
        raise ValueError("observed native runtime differs from plan")
    observe_server(
        plan, role, capture["native_server_info"], capture["native_model_info"]
    )
    if not {"final_native_server_info", "final_native_model_info"} <= capture.keys():
        raise ValueError("capture requires both post-run native observations")
    observe_server(
        plan,
        role,
        capture["final_native_server_info"],
        capture["final_native_model_info"],
    )
    schedule = plan["preflight_cases"] if phase == "preflight" else plan["cases"]
    if [r["id"] for r in capture["rows"]] != [c["id"] for c in schedule]:
        raise ValueError("capture schedule differs")
    for case, row in zip(schedule, capture["rows"], strict=True):
        if row["request"] != request_for(case):
            raise ValueError("captured native request differs from frozen request")
        latency = row["latency_ms"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("invalid measured latency")
    return role


def project_capture(plan: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    role = validate_capture(plan, capture, phase="decision")
    records = {key: [] for key in COHORTS}
    for case, row in zip(plan["cases"], capture["rows"], strict=True):
        latency = row["latency_ms"]
        output, error = _answer(row)
        records[case["cohort"]].append(
            {
                "id": case["id"],
                "input": case["input"],
                "expected": case["expected"],
                "output": output,
                "error": error,
                "scores": {"latency_ms": latency},
                "metadata": {"context": case["context"]},
            }
        )
    return {
        cohort: make_run(
            rows,
            source={
                "name": "sglang-k2-native-capture",
                "version": plan["runtime"]["source_commit"],
            },
            run_id=f"{plan['model']['id']}-{role}-{cohort}",
            artifact_digest=plan["model"][role]["materialized"]["artifact_digest"],
            source_digest=digest(capture),
            score_provenance={"latency_ms": LATENCY_PROVENANCE},
        )
        for cohort, rows in records.items()
    }


def publish(plan, baseline, candidate, key):
    if (baseline["role"], candidate["role"]) != ROLES:
        raise ValueError("capture roles are reversed")
    left, right = project_capture(plan, baseline), project_capture(plan, candidate)
    return {
        cohort: create_evidence(
            left[cohort], right[cohort], plan["policies"][cohort], key
        )
        for cohort in COHORTS
    }


def verify(
    plan,
    baseline,
    candidate,
    evidence,
    public_key,
    *,
    expected_plan,
    expected_baseline_capture,
    expected_candidate_capture,
):
    if digest(plan) != expected_plan:
        raise ValueError("plan differs from independent expected plan")
    if (
        digest(baseline) != expected_baseline_capture
        or digest(candidate) != expected_candidate_capture
    ):
        raise ValueError("capture differs from independent expected capture")
    if (baseline["role"], candidate["role"]) != ROLES or set(evidence) != set(COHORTS):
        raise ValueError("campaign roles or cohorts differ")
    left, right = project_capture(plan, baseline), project_capture(plan, candidate)
    return {
        cohort: verify_evidence(
            evidence[cohort],
            public_key=public_key,
            expected_baseline=digest(left[cohort]),
            expected_candidate=digest(right[cohort]),
            policy=plan["policies"][cohort],
        )["decision"]
        for cohort in COHORTS
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("plan")
    draft.add_argument("--model", required=True)
    draft.add_argument("--output", type=Path, required=True)
    measure = commands.add_parser("measure")
    measure.add_argument("--model", required=True)
    measure.add_argument("--role", choices=ROLES, required=True)
    measure.add_argument("--snapshot", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    download = commands.add_parser("download")
    download.add_argument("--model", required=True)
    download.add_argument("--role", choices=ROLES, required=True)
    download.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--model", required=True)
    for name in (
        "runtime-build",
        "baseline-measurement",
        "candidate-measurement",
        "output",
    ):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    freeze.add_argument("--maximum-wall-seconds", type=int, required=True)
    freeze.add_argument("--maximum-output-tokens", type=int, required=True)
    report = commands.add_parser("report")
    report.add_argument("--evidence", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    for name in ("publish", "verify"):
        command = commands.add_parser(name)
        for item in ("plan", "baseline", "candidate", "key", "output"):
            command.add_argument(f"--{item}", type=Path, required=True)
        if name == "verify":
            command.add_argument("--evidence", type=Path, required=True)
            for item in ("plan", "baseline-capture", "candidate-capture"):
                command.add_argument(f"--expected-{item}", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = select_plan(args.model)
        elif args.command == "measure":
            result = measure_snapshot(
                args.snapshot, select_plan(args.model)["model"][args.role]["files"]
            )
        elif args.command == "download":
            download_snapshot(args.model, args.role, args.output)
            return 0
        elif args.command == "report":
            from invarlock.pipeline.report import (
                render_html,
                render_junit,
                render_markdown,
            )

            evidence = read_json(args.evidence)
            if set(evidence) != set(COHORTS):
                raise ValueError("report requires all campaign cohorts")
            rendered = {
                cohort: {
                    "html": render_html(evidence[cohort]["comparison"]).encode(),
                    "md": render_markdown(evidence[cohort]["comparison"]).encode(),
                    "xml": render_junit(evidence[cohort]["comparison"]),
                }
                for cohort in COHORTS
            }
            args.output.mkdir(exist_ok=False)
            for cohort, formats in rendered.items():
                for suffix, payload in formats.items():
                    with (args.output / f"{cohort}.{suffix}").open("xb") as stream:
                        stream.write(payload)
            return 0
        elif args.command == "freeze":
            result = select_plan(args.model)
            build = read_json(args.runtime_build)
            if (
                build.get("format") != "invarlock/k2-runtime-build-v1"
                or build.get("status") != "ready"
                or build.get("source_commit") != result["runtime"]["source_commit"]
                or build.get("reviewed_source_files")
                != result["runtime"]["reviewed_source_files"]
            ):
                raise ValueError("runtime build is not reviewed and ready")
            for key in _RUNTIME_BINDINGS:
                result["runtime"][key] = (
                    digest(build) if key == "build_manifest_digest" else build[key]
                )
            for role in ROLES:
                result["model"][role]["materialized"] = read_json(
                    getattr(args, f"{role}_measurement")
                )
            result["budget"] = {
                "maximum_wall_seconds": args.maximum_wall_seconds,
                "maximum_output_tokens": args.maximum_output_tokens,
            }
            require_ready(result)
        else:
            from cryptography.hazmat.primitives import serialization

            plan, baseline, candidate = (
                read_json(getattr(args, name))
                for name in ("plan", "baseline", "candidate")
            )
            if args.command == "publish":
                key = serialization.load_pem_private_key(
                    args.key.read_bytes(), password=None
                )
                result = publish(plan, baseline, candidate, key)
            else:
                key = serialization.load_pem_public_key(args.key.read_bytes())
                result = verify(
                    plan,
                    baseline,
                    candidate,
                    read_json(args.evidence),
                    key,
                    expected_plan=args.expected_plan,
                    expected_baseline_capture=args.expected_baseline_capture,
                    expected_candidate_capture=args.expected_candidate_capture,
                )
        write_json(args.output, result)
        if args.command == "publish":
            decisions = [v["comparison"]["decision"] for v in result.values()]
            return (
                3
                if "insufficient_evidence" in decisions
                else 1
                if "regression" in decisions
                else 0
            )
        return 0
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"k2 campaign: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
