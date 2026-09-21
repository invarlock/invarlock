"""Freeze, explicitly admit, and independently verify fresh Luna judge collections."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RECIPIENT = common.module("recipient")
PREPARE = common.module("prepare")
MAX_CALLS = 1008
MAX_COST = 32_000_000
REPRESENTATIVES = {"inspect-ai", "lm-evaluation-harness", "promptfoo"}
PROFILES = {
    "primary": (1, "per_case"),
    "reference-free": (1, "none"),
    "repeat-control": (3, "per_case"),
}
SIGNER = "live-luna-judge"


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read(path):
    return RECIPIENT.read(path, 128 * 1024 * 1024)


def closed_recipe(recipe, plan):
    """Limit collection to the approved model and conservative per-call reserve."""
    judge, schedule, options = plan["judge"], plan["schedule"], recipe["collection"]
    if (
        judge["provider"] != "openai"
        or judge["requested_model"] != "openai/gpt-5.6-luna"
        or judge["approved_resolved_models"] != ["gpt-5.6-luna"]
        or judge["config"]["reasoning_effort"] != "xhigh"
        or judge["config"]["max_output_tokens"] != 25000
        or judge["tools"] is not False
        or schedule["max_attempts"] != 1
        or options["grader"] != "openai/gpt-5.6-luna"
        or options["sdk_max_retries"] != 0
        or options["epochs"] != 1
        or options["cost_microusd_per_call"] != 31200
    ):
        raise ValueError("collection differs from the approved Luna profile")
    calls = schedule["expected_trials"]
    if any(
        type(options[key]) is not int or options[key] != expected
        for key, expected in {
            "max_calls": calls,
            "input_tokens_per_call": 4096,
            "max_input_tokens": calls * 4096,
            "max_output_tokens": calls * 25000,
            "max_cost_microusd": calls * 31200,
        }.items()
    ):
        raise ValueError("collection reserves do not match every planned call")
    return calls, calls * 31200


def freeze(
    protocol_path, protocol_sha256, capture_index, output, *, supplementary=True
):
    """Write a proposal only. Its printed digest must be independently admitted."""
    from invarlock.evaluation_records.io import run_digest
    from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge
    from invarlock.judge_measurements.evidence import object_sha256

    protocol, protocol_raw = read(protocol_path)
    if common.digest(protocol) != protocol_sha256:
        raise ValueError("protocol differs from its independent admission")
    index, _ = read(capture_index)
    if set(index) != set(protocol["evaluators"]) or any(
        set(value) != {"baseline", "subject"} for value in index.values()
    ):
        raise ValueError(
            "capture index must contain exactly both roles for every evaluator"
        )
    output = Path(output).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_raw)
    entries = []
    for profile, (repetitions, reference_mode) in PROFILES.items():
        if profile != "primary" and not supplementary:
            continue
        for evaluator in protocol["evaluators"]:
            if profile != "primary" and evaluator not in REPRESENTATIVES:
                continue
            for route in ("envelope", "native-json"):
                ident = f"{profile}-{evaluator}-{route}"
                directory = output / ident
                directory.mkdir(mode=0o700)
                _, runs = RECIPIENT.prepare(
                    protocol_path,
                    protocol_sha256,
                    index[evaluator]["baseline"],
                    index[evaluator]["subject"],
                    evaluator,
                    route,
                    directory / "capture",
                )
                recipe = PREPARE.judge_recipe(
                    protocol, repetitions=repetitions, reference_mode=reference_mode
                )
                plan, policy = prepare_evaluator_judge(recipe, *runs)
                calls, cost = closed_recipe(recipe, plan)
                fingerprint = RECIPIENT.key(directory / "signer.pem")
                request = {
                    "format_version": "invarlock/evaluation-request-v2",
                    "execution": {"mode": "captured"},
                    "comparison": {
                        "baseline": {
                            "path": "baseline_run.json",
                            "adapter": "invarlock",
                            "expected_run_digest": run_digest(runs[0]),
                        },
                        "subject": {
                            "path": "subject_run.json",
                            "adapter": "invarlock",
                            "expected_run_digest": run_digest(runs[1]),
                        },
                        "policy": "recipe.json",
                        "metric": "judge",
                        "judge": {
                            "workspace": "collection-work",
                            "signer_identity": SIGNER,
                        },
                    },
                    "output": {"evidence": "evidence"},
                }
                values = {
                    "baseline_run": runs[0],
                    "subject_run": runs[1],
                    "recipe": recipe,
                    "plan": plan,
                    "analysis_policy": policy,
                    "request": request,
                }
                for name, value in values.items():
                    common.write(directory / (name + ".json"), value)
                    (directory / (name + ".json")).chmod(0o444)
                entries.append(
                    {
                        "id": ident,
                        "profile": profile,
                        "evaluator": evaluator,
                        "route": route,
                        "calls": calls,
                        "cost_microusd": cost,
                        "plan_sha256": object_sha256(plan),
                        "signer_fingerprint": fingerprint,
                        "files": {
                            name + ".json": sha(common.encoded(value))
                            for name, value in values.items()
                        },
                    }
                )
    ledger = {
        "format": "invarlock/live-judge-admission-v1",
        "campaign_directory": str(output),
        "protocol_sha256": protocol_sha256,
        "maximum_calls": MAX_CALLS,
        "maximum_cost_microusd": MAX_COST,
        "reserved_calls": sum(entry["calls"] for entry in entries),
        "reserved_cost_microusd": sum(entry["cost_microusd"] for entry in entries),
        "entries": entries,
    }
    validate_ledger(ledger)
    common.write(output / "admission.json", ledger)
    return ledger


def validate_ledger(ledger):
    if (
        ledger.get("format") != "invarlock/live-judge-admission-v1"
        or ledger.get("maximum_calls") != MAX_CALLS
        or ledger.get("maximum_cost_microusd") != MAX_COST
    ):
        raise ValueError("unsupported judge admission or campaign ceiling")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > 51:
        raise ValueError("admission requires a bounded explicit entry inventory")
    seen = set()
    for entry in entries:
        ident = entry.get("id")
        if (
            not isinstance(ident, str)
            or not re.fullmatch(r"[a-z0-9-]{1,160}", ident)
            or ident in seen
            or entry.get("profile") not in PROFILES
            or entry.get("route") not in ("envelope", "native-json")
        ):
            raise ValueError("duplicate or unsupported judge admission entry")
        seen.add(ident)
        if any(
            type(entry.get(key)) is not int or entry[key] <= 0
            for key in ("calls", "cost_microusd")
        ):
            raise ValueError("admission reserves require positive integer amounts")
    calls, cost = (
        sum(entry[key] for entry in entries) for key in ("calls", "cost_microusd")
    )
    if (
        calls > MAX_CALLS
        or cost > MAX_COST
        or (calls, cost)
        != (ledger.get("reserved_calls"), ledger.get("reserved_cost_microusd"))
    ):
        raise ValueError("admission exceeds or misstates the approved global budget")


def entry_inputs(root, admission_sha256, ident):
    from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge
    from invarlock.judge_measurements.evidence import object_sha256

    root = Path(root)
    ledger, _ = read(root / "admission.json")
    if common.digest(ledger) != admission_sha256:
        raise ValueError("ledger differs from its independently admitted digest")
    validate_ledger(ledger)
    matches = [entry for entry in ledger["entries"] if entry["id"] == ident]
    if len(matches) != 1:
        raise ValueError("entry is outside the admitted campaign")
    entry = matches[0]
    if set(entry["files"]) != {
        name + ".json"
        for name in (
            "baseline_run",
            "subject_run",
            "recipe",
            "plan",
            "analysis_policy",
            "request",
        )
    }:
        raise ValueError("entry control-file inventory differs")
    values = {}
    for filename, expected in entry["files"].items():
        value, raw = read(root / ident / filename)
        if sha(raw) != expected:
            raise ValueError("frozen judge control file changed")
        values[filename.removesuffix(".json")] = value
    protocol, _ = read(root / "protocol.json")
    if common.digest(protocol) != ledger["protocol_sha256"]:
        raise ValueError("frozen campaign protocol changed")
    plan, policy = prepare_evaluator_judge(
        values["recipe"], values["baseline_run"], values["subject_run"]
    )
    if (
        plan != values["plan"]
        or policy != values["analysis_policy"]
        or object_sha256(plan) != entry["plan_sha256"]
        or closed_recipe(values["recipe"], plan)
        != (entry["calls"], entry["cost_microusd"])
    ):
        raise ValueError(
            "frozen plan or collection reserves differ from admitted inputs"
        )
    return ledger, entry, values, protocol


OFFLINE_LAUNCHER = """
import runpy, sys
runpy.run_path(sys.argv[1])["configure"]("judge-recipient")
sys.argv = sys.argv[2:]
if sys.argv[0] == "invarlock":
    runpy.run_module("invarlock", run_name="__main__")
else:
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


def environment(offline=False, *, home, collection=False):
    """Only explicit credentials enter collection; verification gets none."""
    directory = Path(home)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    value = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TMPDIR",
            "TEMP",
            "TMP",
            "SYSTEMROOT",
            "WINDIR",
        )
        if key in os.environ
    }
    value.update(
        HOME=str(directory),
        XDG_CONFIG_HOME=str(directory / "config"),
        XDG_CACHE_HOME=str(directory / "cache"),
        XDG_DATA_HOME=str(directory / "data"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    if not offline and "OPENAI_API_KEY" in os.environ:
        value["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if collection and not offline:
        value["INVARLOCK_ALLOW_JUDGE_NETWORK"] = "1"
    return value


def offline_command(python, target, *args):
    return [
        str(python),
        "-I",
        "-c",
        OFFLINE_LAUNCHER,
        str(Path(__file__).with_name("network.py").resolve()),
        str(target),
        *args,
    ]


def cli(directory, *args, allowed=(0,), offline_home=None, collection=False):
    offline = args[0] in {"verify", "report"}
    command = (
        offline_command(sys.executable, "invarlock", *args)
        if offline
        else [sys.executable, "-I", "-m", "invarlock", *args]
    )
    home = offline_home or Path(tempfile.mkdtemp(prefix="command-home-", dir=directory))
    result = subprocess.run(
        command,
        cwd=directory,
        env=environment(offline=offline, home=home, collection=collection),
        capture_output=True,
        text=True,
        timeout=22000,
    )
    if result.returncode not in allowed:
        raise RuntimeError(
            f"judge command stopped ({result.returncode}): {result.stdout}{result.stderr}"
        )
    return common.decode(result.stdout.encode())


def collect(
    root, admission_sha256, ident, recipient_python, *, execute=False, resume=False
):
    """Reserve an entire entry before any provider call; resume only its checkpoint."""
    if not execute:
        raise ValueError("collection requires explicit execute authorization")
    ledger, entry, values, protocol = entry_inputs(root, admission_sha256, ident)
    root = Path(root).absolute()
    if (
        str(root) != ledger["campaign_directory"]
        or protocol["configuration"].get("device") != "cuda"
    ):
        raise ValueError(
            "paid collection requires original campaign location and admitted fresh GPU captures"
        )
    directory = root / ident
    from cryptography.hazmat.primitives import serialization

    from invarlock.captured_contracts import read_file
    from invarlock.evidence_pack_integrity import public_key_fingerprint

    signing_key = serialization.load_pem_private_key(
        read_file(directory / "signer.pem", 65536), password=None
    )
    if public_key_fingerprint(signing_key.public_key()) != entry["signer_fingerprint"]:
        raise ValueError("signer differs from the independently admitted key")
    if (directory / "verification.json").exists():
        raise ValueError("entry already verified; new calls are not authorized")
    admission = {
        "admission_sha256": admission_sha256,
        "entry": ident,
        "calls": entry["calls"],
        "cost_microusd": entry["cost_microusd"],
    }
    started = directory / "collection-admission.json"
    if started.exists():
        if not resume or read(started)[0] != admission:
            raise ValueError(
                "entry already admitted; explicit unchanged-checkpoint resume is required"
            )
    else:
        if resume:
            raise ValueError("cannot resume an entry without its prior admission")
        if any(
            (directory / name).exists()
            for name in ("evidence", "collection-work", "collection-result.json")
        ):
            raise ValueError("preexisting collection cannot enter a fresh admission")
        common.write(started, admission)
    if not (directory / "evidence").exists():
        preflight = cli(
            directory,
            "evaluate",
            "request.json",
            "--preflight",
            "--signing-key",
            "signer.pem",
            "--json",
        )
        if (
            preflight.get("network_calls") != 0
            or preflight.get("planned_trials") != entry["calls"]
            or not preflight.get("collection_available")
        ):
            raise ValueError("judge preflight differs from exact admitted collection")
        result = cli(
            directory,
            "evaluate",
            "request.json",
            "--signing-key",
            "signer.pem",
            "--fail-on-policy",
            "--json",
            collection=True,
            allowed=(0, 7),
        )
        common.write(directory / "collection-result.json", result)
    recipient_home = Path(tempfile.mkdtemp(prefix="verification-home-", dir=directory))
    process = subprocess.run(
        offline_command(
            recipient_python,
            Path(__file__).resolve(),
            "verify",
            "--root",
            str(root),
            "--admission-sha256",
            admission_sha256,
            "--entry",
            ident,
        ),
        cwd=root,
        env=environment(offline=True, home=recipient_home),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"offline recipient failed: {process.stdout}{process.stderr}"
        )
    return common.decode(process.stdout.encode())


def verify(root, admission_sha256, ident, *, input_loader=None):
    """Replay newly collected evidence against pre-call pins in the offline recipient."""
    from invarlock.evaluation_records.identity import evaluated_subject_digest
    from invarlock.evaluation_records.io import run_digest
    from invarlock.judge_measurements.analysis import (
        analyze_measurements,
        decode_analysis_policy,
    )
    from invarlock.judge_measurements.evidence import DECISION_SCOPE, object_sha256

    _, entry, values, _ = (input_loader or entry_inputs)(root, admission_sha256, ident)
    directory = Path(root) / ident
    measurements, _ = read(directory / "evidence/measurements.json")
    analysis = analyze_measurements(
        values["plan"],
        measurements,
        decode_analysis_policy(values["analysis_policy"], plan=values["plan"]),
        baseline_run=values["baseline_run"],
        subject_run=values["subject_run"],
    ).to_dict()
    trust = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": DECISION_SCOPE,
        "intended_subject": evaluated_subject_digest(values["subject_run"]),
        "required_metric_name": values["analysis_policy"]["metric_name"],
        "trusted_signer": {
            "identity": SIGNER,
            "public_key_sha256": entry["signer_fingerprint"],
        },
        "bindings": {
            "baseline_run_sha256": run_digest(values["baseline_run"]),
            "subject_run_sha256": run_digest(values["subject_run"]),
            "case_set_sha256": values["plan"]["case_set_sha256"],
            "plan_sha256": object_sha256(values["plan"]),
            "measurements_sha256": object_sha256(measurements),
            "analysis_policy_sha256": object_sha256(values["analysis_policy"]),
            "analysis_result_sha256": object_sha256(analysis),
        },
        "required_decision": "pass",
    }
    # Preserve failed offline attempts without repeating any provider collection.
    attempt = Path(tempfile.mkdtemp(prefix="recipient-", dir=directory)).resolve()
    common.write(attempt / "trust.json", trust)
    RECIPIENT.key(attempt / "verifier.pem")
    verified = cli(
        directory,
        "verify",
        "evidence",
        "--trust-profile",
        str(attempt / "trust.json"),
        "--receipt",
        str(attempt / "verification.receipt.json"),
        "--verifier-signing-key",
        str(attempt / "verifier.pem"),
        "--verifier-identity",
        "live-luna-recipient",
        "--json",
        allowed=(0, 7),
        offline_home=attempt / "home",
    )
    if (
        not all(
            verified.get(field) for field in ("authenticated", "verified", "replayed")
        )
        or verified["decision"] != analysis["decision"]
    ):
        raise ValueError(
            "offline judge recipient did not authenticate and replay the fresh collection"
        )
    rendered = cli(
        directory,
        "report",
        "evidence",
        "--html",
        str(attempt / "report.html"),
        "--markdown",
        str(attempt / "report.md"),
        "--junit",
        str(attempt / "report.xml"),
        "--explain",
        "--json",
        offline_home=attempt / "home",
    )
    common.write(attempt / "report.json", rendered)
    verified["recipient_artifacts"] = attempt.name
    common.write(directory / "verification.json", verified)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proposal = commands.add_parser("freeze")
    for field in ("protocol", "captures", "output"):
        proposal.add_argument("--" + field, type=Path, required=True)
    proposal.add_argument("--protocol-sha256", required=True)
    proposal.add_argument("--primary-only", action="store_true")
    for operation in ("collect", "verify"):
        command = commands.add_parser(operation)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--admission-sha256", required=True)
        command.add_argument("--entry", required=True)
        if operation == "collect":
            command.add_argument("--recipient-python", type=Path, required=True)
            command.add_argument("--execute-collection", action="store_true")
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        common.module("network").configure("judge-recipient")
    if args.command != "collect":
        RECIPIENT.installed_identity()
    if args.command == "freeze":
        result = freeze(
            args.protocol,
            args.protocol_sha256,
            args.captures,
            args.output,
            supplementary=not args.primary_only,
        )
        print(common.digest(result))
    else:
        result = (
            collect(
                args.root,
                args.admission_sha256,
                args.entry,
                args.recipient_python,
                execute=args.execute_collection,
                resume=args.resume,
            )
            if args.command == "collect"
            else verify(args.root, args.admission_sha256, args.entry)
        )
        print(common.encoded(result).decode())


if __name__ == "__main__":
    main()
