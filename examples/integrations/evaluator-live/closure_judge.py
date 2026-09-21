"""Propose and execute separately admitted judge campaigns with graceful resume."""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BASE = common.module("judge")
RECIPIENT = BASE.RECIPIENT
FORMAT = "invarlock/live-judge-campaign-v2"
MAX_PROPOSAL_CALLS = 1288
CALL_RESERVE_MICROUSD = 31200
MAX_PROPOSAL_COST = MAX_PROPOSAL_CALLS * CALL_RESERVE_MICROUSD
EVALUATORS = {"inspect-ai", "lm-evaluation-harness", "promptfoo", "langfuse"}
PROFILES = {
    "primary": (1, "per_case"),
    "reference-free": (1, "none"),
    "repeat-control": (3, "per_case"),
    "budget-control": (3, "none"),
}
FILES = {"protocol", "baseline_run", "subject_run", "recipe", "plan", "analysis_policy"}


def require(value, message):
    if not value:
        raise ValueError(message)


def positive(value, maximum):
    return type(value) is int and 0 < value <= maximum


def validate_spec(spec):
    """Explicit new limits never change the legacy campaign's admission format."""
    require(
        set(spec)
        == {
            "format",
            "maximum_calls",
            "maximum_cost_microusd",
            "cost_microusd_per_call",
            "groups",
        },
        "unsupported campaign specification fields",
    )
    require(spec["format"] == FORMAT, "new campaign format required")
    require(
        positive(spec["maximum_calls"], MAX_PROPOSAL_CALLS)
        and positive(spec["maximum_cost_microusd"], MAX_PROPOSAL_COST)
        and type(spec["cost_microusd_per_call"]) is int
        and spec["cost_microusd_per_call"] == CALL_RESERVE_MICROUSD,
        "invalid explicit campaign limits",
    )
    groups = spec["groups"]
    require(
        isinstance(groups, list) and 0 < len(groups) <= 16,
        "bounded campaign groups required",
    )
    seen, calls = set(), 0
    for group in groups:
        require(
            set(group)
            == {
                "id",
                "protocol",
                "protocol_sha256",
                "captures",
                "evaluators",
                "routes",
                "case_count",
                "profile",
                "admitted_calls",
            },
            "unsupported campaign group fields",
        )
        ident = group["id"]
        require(
            isinstance(ident, str)
            and re.fullmatch(r"[a-z0-9-]{1,48}", ident)
            and ident not in seen,
            "duplicate or invalid group identity",
        )
        seen.add(ident)
        require(
            group["profile"] in PROFILES and positive(group["case_count"], 4096),
            "unsupported profile or case count",
        )
        for field, allowed in (
            ("evaluators", EVALUATORS),
            ("routes", {"envelope", "native-json"}),
        ):
            values = group[field]
            require(
                isinstance(values, list)
                and values
                and all(isinstance(value, str) for value in values)
                and len(set(values)) == len(values)
                and set(values) <= allowed,
                "unsupported or duplicate campaign selection",
            )
        require(
            isinstance(group["protocol_sha256"], str)
            and re.fullmatch(r"sha256:[a-f0-9]{64}", group["protocol_sha256"]),
            "independent protocol digest required",
        )
        expected = 2 * group["case_count"] * PROFILES[group["profile"]][0]
        limit = group["admitted_calls"]
        if group["profile"] == "budget-control":
            require(
                positive(limit, expected - 1),
                "budget control must reserve fewer calls than its separate plan",
            )
        else:
            require(limit is None, "only budget controls may truncate a plan's reserve")
        calls += (
            (expected if limit is None else limit)
            * len(group["evaluators"])
            * len(group["routes"])
        )
    require(
        calls <= spec["maximum_calls"]
        and calls * spec["cost_microusd_per_call"] <= spec["maximum_cost_microusd"],
        "proposal exceeds its explicit aggregate ceiling",
    )
    return calls


def recipe_for(protocol, group, unit_cost):
    repetitions, reference_mode = PROFILES[group["profile"]]
    recipe = BASE.PREPARE.judge_recipe(
        protocol, repetitions=repetitions, reference_mode=reference_mode
    )
    calls = group["admitted_calls"] or 2 * group["case_count"] * repetitions
    recipe["collection"].update(
        max_calls=calls,
        max_input_tokens=calls * 4096,
        max_output_tokens=calls * 25000,
        max_cost_microusd=calls * unit_cost,
        cost_microusd_per_call=unit_cost,
    )
    return recipe


def freeze(specification, output):
    """Freeze reviewed captures and plans; this function never authorizes calls."""
    from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge

    spec, _ = BASE.read(specification)
    reserved = validate_spec(spec)
    output = Path(output).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    entries = []
    for group in spec["groups"]:
        protocol, _ = BASE.read(group["protocol"])
        require(
            common.digest(protocol) == group["protocol_sha256"]
            and len(common.cases(protocol["cases"])) == group["case_count"],
            "group protocol or case count differs",
        )
        index, _ = BASE.read(group["captures"])
        require(
            set(group["evaluators"]) <= set(index)
            and set(group["evaluators"]) <= set(protocol["evaluators"]),
            "selected evaluator missing from protocol or captures",
        )
        for evaluator in group["evaluators"]:
            require(
                set(index[evaluator]) == {"baseline", "subject"},
                "both capture roles required",
            )
            for route in group["routes"]:
                ident = f"{group['id']}-{evaluator}-{route}"
                directory = output / ident
                directory.mkdir(mode=0o700)
                _, runs = RECIPIENT.prepare(
                    Path(group["protocol"]),
                    group["protocol_sha256"],
                    index[evaluator]["baseline"],
                    index[evaluator]["subject"],
                    evaluator,
                    route,
                    directory / "capture",
                )
                recipe = recipe_for(protocol, group, spec["cost_microusd_per_call"])
                plan, policy = prepare_evaluator_judge(recipe, *runs)
                values = {
                    "protocol": protocol,
                    "baseline_run": runs[0],
                    "subject_run": runs[1],
                    "recipe": recipe,
                    "plan": plan,
                    "analysis_policy": policy,
                }
                for name, value in values.items():
                    common.write(directory / (name + ".json"), value)
                    (directory / (name + ".json")).chmod(0o444)
                entries.append(
                    {
                        "id": ident,
                        "group": group["id"],
                        "evaluator": evaluator,
                        "route": route,
                        "calls": recipe["collection"]["max_calls"],
                        "cost_microusd": recipe["collection"]["max_cost_microusd"],
                        "signer_fingerprint": RECIPIENT.key(directory / "signer.pem"),
                        "files": {
                            name + ".json": BASE.sha(common.encoded(value))
                            for name, value in values.items()
                        },
                    }
                )
    ledger = {
        "format": FORMAT,
        "campaign_directory": str(output),
        "specification": spec,
        "reserved_calls": reserved,
        "reserved_cost_microusd": reserved * spec["cost_microusd_per_call"],
        "entries": entries,
    }
    validate_ledger(ledger)
    common.write(output / "admission.json", ledger)
    return ledger


def validate_ledger(ledger):
    require(
        set(ledger)
        == {
            "format",
            "campaign_directory",
            "specification",
            "reserved_calls",
            "reserved_cost_microusd",
            "entries",
        }
        and ledger["format"] == FORMAT,
        "unsupported campaign ledger",
    )
    spec = ledger["specification"]
    reserved = validate_spec(spec)
    expected = {
        f"{group['id']}-{evaluator}-{route}": (group, evaluator, route)
        for group in spec["groups"]
        for evaluator in group["evaluators"]
        for route in group["routes"]
    }
    require(
        isinstance(ledger["entries"], list) and len(ledger["entries"]) == len(expected),
        "ledger entry inventory differs",
    )
    for entry in ledger["entries"]:
        require(
            set(entry)
            == {
                "id",
                "group",
                "evaluator",
                "route",
                "calls",
                "cost_microusd",
                "signer_fingerprint",
                "files",
            }
            and entry["id"] in expected,
            "unsupported or duplicate ledger entry",
        )
        group, evaluator, route = expected.pop(entry["id"])
        calls = (
            group["admitted_calls"]
            or 2 * group["case_count"] * PROFILES[group["profile"]][0]
        )
        require(
            (entry["group"], entry["evaluator"], entry["route"])
            == (group["id"], evaluator, route)
            and type(entry["calls"]) is int
            and entry["calls"] == calls
            and type(entry["cost_microusd"]) is int
            and entry["cost_microusd"] == calls * spec["cost_microusd_per_call"],
            "entry differs from explicit group reserves",
        )
    require(
        type(ledger["reserved_calls"]) is int
        and ledger["reserved_calls"] == reserved
        and type(ledger["reserved_cost_microusd"]) is int
        and ledger["reserved_cost_microusd"]
        == reserved * spec["cost_microusd_per_call"],
        "ledger aggregate reserve differs",
    )


def entry_inputs(root, admission_sha256, ident):
    from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge

    root = Path(root).resolve(strict=True)
    ledger, _ = BASE.read(root / "admission.json")
    require(
        common.digest(ledger) == admission_sha256,
        "ledger differs from its independent admission",
    )
    validate_ledger(ledger)
    require(str(root) == ledger["campaign_directory"], "campaign location differs")
    matches = [entry for entry in ledger["entries"] if entry["id"] == ident]
    require(len(matches) == 1, "entry outside admitted campaign")
    entry = matches[0]
    require(
        set(entry["files"]) == {name + ".json" for name in FILES},
        "entry file inventory differs",
    )
    values = {}
    for name, pin in entry["files"].items():
        value, raw = BASE.read(root / ident / name)
        require(BASE.sha(raw) == pin, "frozen entry file differs")
        values[name.removesuffix(".json")] = value
    group = next(
        group
        for group in ledger["specification"]["groups"]
        if group["id"] == entry["group"]
    )
    protocol = values["protocol"]
    require(
        common.digest(protocol) == group["protocol_sha256"]
        and len(common.cases(protocol["cases"])) == group["case_count"],
        "frozen protocol differs",
    )
    recipe = recipe_for(
        protocol, group, ledger["specification"]["cost_microusd_per_call"]
    )
    require(recipe == values["recipe"], "recipe differs from admitted profile")
    plan, policy = prepare_evaluator_judge(
        recipe, values["baseline_run"], values["subject_run"]
    )
    require(
        plan == values["plan"] and policy == values["analysis_policy"],
        "frozen plan or policy differs",
    )
    return ledger, entry, values, protocol


def execute_entry(
    root, pin, ident, *, execute=False, resume=False, stop_after_batches=None
):
    """Use public collection APIs; the same durable checkpoint owns all resumes."""
    from cryptography.hazmat.primitives import serialization

    from invarlock.captured_contracts import read_file
    from invarlock.evidence_pack_integrity import public_key_fingerprint
    from invarlock.judge_measurements import (
        CollectionOptions,
        RunnerOptions,
        collect_configured,
    )
    from invarlock.judge_measurements.evidence import publish_judge_evidence
    from invarlock.judge_measurements.native_workflow import locked_workspace

    require(execute, "explicit execute authorization required")
    _, entry, values, protocol = entry_inputs(root, pin, ident)
    directory = Path(root).resolve(strict=True) / ident
    require(
        protocol["configuration"].get("device") == "cuda",
        "execution requires independently admitted model captures",
    )
    runner = RunnerOptions(
        directory / "collection-work/collection",
        **values["recipe"]["runner"],
        stop_after_batches=stop_after_batches,
    )
    runner.validate()
    key = serialization.load_pem_private_key(
        read_file(directory / "signer.pem", 65536), password=None
    )
    require(
        public_key_fingerprint(key.public_key()) == entry["signer_fingerprint"],
        "admitted signer differs",
    )
    require(
        not (directory / "evidence").exists(),
        "entry already published; no further calls authorized",
    )
    admission = {
        "admission_sha256": pin,
        "entry": ident,
        "calls": entry["calls"],
        "cost_microusd": entry["cost_microusd"],
    }
    with locked_workspace(directory / "collection-work"):
        started = directory / "collection-admission.json"
        if started.exists():
            require(
                resume and BASE.read(started)[0] == admission,
                "explicit unchanged-checkpoint resume required",
            )
        else:
            require(
                not resume and not runner.checkpoint_directory.exists(),
                "fresh admission cannot reuse prior collection",
            )
            common.write(started, admission)
        stops = []
        measured = asyncio.run(
            collect_configured(
                values["plan"],
                CollectionOptions(**values["recipe"]["collection"]),
                runner,
                values["baseline_run"],
                values["subject_run"],
                on_stop=stops.append,
            )
        )
        require(
            len(stops) == 1
            and stops[0] in {"complete", "requested", "deadline", "capacity_exhausted"},
            "collector did not retain a supported stop reason",
        )
        result = {
            "entry": ident,
            "stop_reason": stops[0],
            "resumable": stops[0] in {"requested", "deadline"},
            "completeness": measured["completeness"],
        }
        if result["resumable"]:
            attempt = Path(tempfile.mkdtemp(prefix="stopped-", dir=directory))
            common.write(attempt / "measurements.json", measured)
            common.write(attempt / "result.json", result)
            return result
        publication = publish_judge_evidence(
            directory / "evidence",
            plan=values["plan"],
            measurements=measured,
            baseline_run=values["baseline_run"],
            subject_run=values["subject_run"],
            analysis_policy=values["analysis_policy"],
            signing_key=key,
            signer_identity=BASE.SIGNER,
        )
        result["decision"] = publication.analysis_result.to_dict()["decision"]
        common.write(directory / "collection-result.json", result)
        return result


def verify(root, pin, ident):
    return BASE.verify(root, pin, ident, input_loader=entry_inputs)


def collect(
    root,
    pin,
    ident,
    recipient_python,
    *,
    execute=False,
    resume=False,
    stop_after_batches=None,
):
    """Collect in the credentialed SDK process, then verify in a separate recipient."""
    require(execute, "explicit execute authorization required")
    _, entry, _, _ = entry_inputs(root, pin, ident)
    directory = Path(root).resolve(strict=True) / ident
    require(
        not (directory / "verification.json").exists(),
        "entry already independently verified",
    )
    if (directory / "evidence").exists():
        expected = {
            "admission_sha256": pin,
            "entry": ident,
            "calls": entry["calls"],
            "cost_microusd": entry["cost_microusd"],
        }
        require(
            resume
            and BASE.read(directory / "collection-admission.json")[0] == expected,
            "published evidence requires its original admission and explicit resume",
        )
    else:
        args = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "execute",
            "--root",
            str(Path(root).resolve()),
            "--admission-sha256",
            pin,
            "--entry",
            ident,
            "--execute-collection",
        ]
        if resume:
            args.append("--resume")
        if stop_after_batches is not None:
            args.extend(["--stop-after-batches", str(stop_after_batches)])
        home = Path(tempfile.mkdtemp(prefix="collection-home-", dir=directory))
        process = subprocess.run(
            args,
            cwd=directory,
            env=BASE.environment(home=home, collection=True),
            capture_output=True,
            text=True,
            timeout=22000,
        )
        require(
            process.returncode == 0,
            "collection child stopped: " + process.stdout + process.stderr,
        )
        result = common.decode(process.stdout.encode())
        if result["resumable"]:
            return result
    home = Path(tempfile.mkdtemp(prefix="verification-home-", dir=directory))
    process = subprocess.run(
        BASE.offline_command(
            recipient_python,
            Path(__file__).resolve(),
            "verify",
            "--root",
            str(Path(root).resolve()),
            "--admission-sha256",
            pin,
            "--entry",
            ident,
        ),
        cwd=directory,
        env=BASE.environment(offline=True, home=home),
        capture_output=True,
        text=True,
        timeout=300,
    )
    require(
        process.returncode == 0,
        "offline recipient failed: " + process.stdout + process.stderr,
    )
    return common.decode(process.stdout.encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--specification", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    for name in ("collect", "execute", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--admission-sha256", required=True)
        command.add_argument("--entry", required=True)
        if name != "verify":
            command.add_argument("--execute-collection", action="store_true")
            command.add_argument("--resume", action="store_true")
            command.add_argument("--stop-after-batches", type=int)
        if name == "collect":
            command.add_argument("--recipient-python", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"freeze", "verify"}:
        common.module("network").configure("judge-recipient")
        RECIPIENT.installed_identity()
    if args.command == "freeze":
        print(common.digest(freeze(args.specification, args.output)))
    elif args.command == "verify":
        print(
            common.encoded(
                verify(args.root, args.admission_sha256, args.entry)
            ).decode()
        )
    else:
        options = {
            "execute": args.execute_collection,
            "resume": args.resume,
            "stop_after_batches": args.stop_after_batches,
        }
        result = (
            collect(
                args.root,
                args.admission_sha256,
                args.entry,
                args.recipient_python,
                **options,
            )
            if args.command == "collect"
            else execute_entry(args.root, args.admission_sha256, args.entry, **options)
        )
        print(common.encoded(result).decode())


if __name__ == "__main__":
    main()
