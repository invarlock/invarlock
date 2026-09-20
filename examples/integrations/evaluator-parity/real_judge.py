#!/usr/bin/env python3
"""Replay complete real Luna grounded-QA campaigns in an installed recipient."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REFERENCES = {
    "heldout": {
        "sha256": "19574e2f68f00e06f13432369d99818e625b58d10bf53156bebf86b538409242",
        "expanded_bytes": 216183325,
        "cases": 422,
        "trials": 2532,
        "decision": "pass",
    },
    "pilot": {
        "sha256": "4d8f50e1cba0056d2118695a4dab73cce4a5ab10829320e2f8ea4b0c48d0e766",
        "expanded_bytes": 64 * 1024 * 1024,
        "cases": 40,
        "trials": 240,
        "decision": "insufficient_evidence",
    },
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PARITY = module("real_judge_parity_helpers", HERE / "run.py")
ARCHIVE = module(
    "real_judge_archive", ROOT / "examples/judge_measurements_pilot_reference.py"
)


def retained(reference):
    selected = REFERENCES[reference]
    archive = (
        ROOT
        / f"examples/judge-measurements/references/k2-32b-luna-xhigh-{reference}/reference.zip"
    )
    files = ARCHIVE.read_archive(
        archive, selected["sha256"], max_expanded_bytes=selected["expanded_bytes"]
    )
    prefix = f"{reference}/grounded_qa/"
    raw = {
        name: files[f"{prefix}evidence/{name}.json"]
        for name in (
            "baseline_run",
            "subject_run",
            "plan",
            "analysis_policy",
            "measurements",
            "analysis_result",
            "case_set",
            "envelope",
        )
    }
    raw["collection"] = files[prefix + "collection.json"]
    raw["receipt"] = files[prefix + "receipt.json"]
    raw["recipient"] = files[prefix + "recipient.json"]
    documents = {name: json.loads(value) for name, value in raw.items()}
    if (
        len(documents["baseline_run"]["records"]) != selected["cases"]
        or len(documents["measurements"]["trials"]) != selected["trials"]
        or documents["analysis_result"]["decision"] != selected["decision"]
    ):
        raise ValueError(
            "retained judge campaign differs from its independent inventory"
        )
    return raw, documents


def recipe(documents):
    from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge

    plan = copy.deepcopy(documents["plan"])
    for name in (
        "answer_bindings",
        "baseline_run_sha256",
        "subject_run_sha256",
        "case_set_sha256",
    ):
        plan.pop(name)
    plan["rubric"].pop("sha256")
    plan["schedule"].pop("expected_trials")
    analysis = copy.deepcopy(documents["analysis_policy"])
    analysis.pop("plan_sha256")
    value = {
        "format": "invarlock/native-judge-policy-v1",
        "plan": plan,
        "analysis": analysis,
        "collection": documents["collection"],
        "runner": {
            "scorer_id": "retained-luna-judge",
            "invocation_timeout_seconds": 600,
        },
    }
    reconstructed, policy = prepare_evaluator_judge(
        value, documents["baseline_run"], documents["subject_run"]
    )
    if reconstructed != documents["plan"] or policy != documents["analysis_policy"]:
        raise ValueError(
            "retained judge plan or policy changed during recipe reconstruction"
        )
    return value


def prepare(output, reference):
    from invarlock.judge_measurements.acceptance import (
        replay_signed_judge_verification_receipt,
    )
    from invarlock.judge_measurements.evidence import DECISION_SCOPE, object_sha256

    raw, documents = retained(reference)
    policy = recipe(documents)
    destination = output / "retained"
    (destination / "evidence").mkdir(parents=True)
    for name, content in raw.items():
        directory = (
            destination
            if name in {"collection", "receipt", "recipient"}
            else destination / "evidence"
        )
        (directory / f"{name}.json").write_bytes(content)
    # Original receipt authority comes from the independently pinned archive;
    # authenticate it before issuing any new replay publication or trust inputs.
    verifier = documents["receipt"]["statement"]["verifier"]
    original = replay_signed_judge_verification_receipt(
        destination / "receipt.json",
        evidence_path=destination / "evidence",
        recipient_policy_path=destination / "recipient.json",
        expected_verifier_identity=verifier["identity"],
        expected_verifier_fingerprint=verifier["signing_key_fingerprint"],
    )
    if not (original.authenticated and original.verified and original.replayed):
        raise ValueError(
            "original retained judge receipt failed authentication or replay"
        )
    if original.decision != REFERENCES[reference]["decision"] or original.accepted != (
        reference == "heldout"
    ):
        raise ValueError("original retained judge decision differs")
    fingerprint = PARITY._key(output / "signer.pem")
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {
                "path": "retained/evidence/baseline_run.json",
                "adapter": "invarlock",
            },
            "subject": {
                "path": "retained/evidence/subject_run.json",
                "adapter": "invarlock",
            },
            "policy": "policy.json",
            "metric": "judge",
            "judge": {
                "workspace": "unused-judge-work",
                "signer_identity": "retained-judge-replay",
                "measurements": "retained/evidence/measurements.json",
            },
        },
        "output": {"evidence": "evidence"},
    }
    plan = documents["plan"]
    trust = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": DECISION_SCOPE,
        "intended_subject": documents["subject_run"]["artifact_digest"],
        "required_metric_name": documents["analysis_policy"]["metric_name"],
        "trusted_signer": {
            "identity": "retained-judge-replay",
            "public_key_sha256": fingerprint,
        },
        "bindings": {
            **{
                name: plan[name]
                for name in (
                    "baseline_run_sha256",
                    "subject_run_sha256",
                    "case_set_sha256",
                )
            },
            **{
                f"{name}_sha256": object_sha256(documents[name])
                for name in (
                    "plan",
                    "measurements",
                    "analysis_policy",
                    "analysis_result",
                )
            },
        },
        "required_decision": "pass",
    }
    origin = {
        "archive_sha256": REFERENCES[reference]["sha256"],
        "reference": reference,
        "workflow": "grounded_qa",
        "files": {
            name: {"sha256": ARCHIVE.sha(content), "size_bytes": len(content)}
            for name, content in raw.items()
        },
        "scope": "Complete retained real Luna grounded-QA campaign; no new model or judge calls.",
        "original_receipt_authenticated": True,
    }
    for name, value in (
        ("request", request),
        ("policy", policy),
        ("trust", trust),
        ("origin", origin),
    ):
        PARITY.write(output / f"{name}.json", value)
    return documents, origin


def journey(output, reference, python):
    documents, origin = prepare(output, reference)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("INVARLOCK_SIGNING_KEY", None)
    environment.pop("INVARLOCK_ALLOW_NETWORK", None)
    environment.pop("INVARLOCK_ALLOW_JUDGE_NETWORK", None)
    commands = []

    def command(*args, allowed=(0,)):
        # Block network at the process executing the installed CLI, including
        # accidental collection paths. The measurements are already complete.
        bootstrap = (
            "import runpy,socket,sys\n"
            "def blocked(*args,**kwargs): raise RuntimeError('retained replay forbids network calls')\n"
            "socket.socket.connect=blocked; socket.create_connection=blocked\n"
            "sys.argv=['invarlock',*sys.argv[1:]]\n"
            "runpy.run_module('invarlock',run_name='__main__')"
        )
        result = subprocess.run(
            [str(python), "-I", "-c", bootstrap, *args],
            cwd=output,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        commands.append({"command": list(args), "exit_code": result.returncode})
        if result.returncode not in allowed:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)

    preflight = command(
        "evaluate",
        "request.json",
        "--preflight",
        "--signing-key",
        "signer.pem",
        "--json",
    )
    assert preflight["network_calls"] == 0 and not preflight["collection_available"]
    assert preflight["planned_trials"] == REFERENCES[reference]["trials"]
    assert not (output / "evidence").exists()
    PARITY.write(output / "preflight.json", preflight)
    status = 0 if reference == "heldout" else 7
    evaluated = command(
        "evaluate",
        "request.json",
        "--signing-key",
        "signer.pem",
        "--fail-on-policy",
        "--json",
        allowed=(status,),
    )
    assert evaluated["authentication"] == "signed"
    assert evaluated["decision"] == REFERENCES[reference]["decision"]
    verified = command(
        "verify",
        "evidence",
        "--trust-profile",
        "trust.json",
        "--json",
        allowed=(status,),
    )
    assert verified["authenticated"] and verified["replayed"] and verified["verified"]
    assert verified["accepted"] == (reference == "heldout")
    assert verified["decision"] == REFERENCES[reference]["decision"]
    PARITY.write(output / "verification.json", verified)
    command(
        "report",
        "evidence",
        "--html",
        "report.html",
        "--markdown",
        "report.md",
        "--json",
    )
    assert "gpt-5.6-luna" in (output / "report.html").read_text()
    for name in (
        "baseline_run",
        "subject_run",
        "plan",
        "analysis_policy",
        "measurements",
        "analysis_result",
    ):
        assert PARITY.read(output / "evidence" / f"{name}.json") == documents[name]
    path = output / "evidence/measurements.json"
    raw, mode = path.read_bytes(), path.stat().st_mode & 0o777
    altered = json.loads(raw)
    altered["trials"][0]["attempts"][0]["response"]["text"] = '{"rating":"incorrect"}'
    assert altered != json.loads(raw)
    path.chmod(mode | 0o200)
    try:
        PARITY.write(path, altered)
        rejected = command(
            "verify",
            "evidence",
            "--trust-profile",
            "trust.json",
            "--json",
            allowed=(4,),
        )
    finally:
        path.write_bytes(raw)
        path.chmod(mode)
    assert not rejected["accepted"] and not rejected["verified"]
    PARITY.write(output / "altered-measurement-refusal.json", rejected)
    assert not (output / "unused-judge-work").exists()
    for name, pin in origin["files"].items():
        directory = output / "retained"
        if name not in {"collection", "receipt", "recipient"}:
            directory /= "evidence"
        assert ARCHIVE.sha((directory / f"{name}.json").read_bytes()) == pin["sha256"]
    result = {
        "reference": reference,
        "workflow": "grounded_qa",
        "cases": REFERENCES[reference]["cases"],
        "trials": REFERENCES[reference]["trials"],
        "decision": evaluated["decision"],
        "accepted": verified["accepted"],
        "authenticated": True,
        "replayed": True,
        "altered_measurement_rejected": True,
        "new_calls": 0,
        "scope": origin["scope"],
        "commands": commands,
    }
    PARITY.write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipient-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", choices=("heldout", "pilot", "both"), default="both"
    )
    args = parser.parse_args()
    recipient = PARITY.installed_identity(args.recipient_python)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for reference in REFERENCES if args.reference == "both" else [args.reference]:
        current = output / reference
        current.mkdir()
        results.append(journey(current, reference, args.recipient_python.absolute()))
    print(json.dumps({"recipient": recipient, "references": results}, indent=2))


if __name__ == "__main__":
    main()
