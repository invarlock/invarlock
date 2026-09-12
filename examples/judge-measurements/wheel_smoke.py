"""Rehearse the core-only offline judge recipient using the bounded example.

The one-case fixture must remain insufficient evidence. No provider is called,
no policy is weakened, and recipient pins are computed before publication from
separately supplied inputs, never copied from the submitted envelope.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from invarlock.judge_measurements.analysis import (
    analyze_measurements,
    decode_analysis_policy,
)
from invarlock.judge_measurements.contracts import measurement_plan_digest
from invarlock.judge_measurements.evidence import object_sha256

FIXTURE_FILES = (
    "request.yaml",
    "plan.json",
    "measurements.json",
    "baseline_run.json",
    "subject_run.json",
    "analysis_policy.json",
)


def require_core_only() -> None:
    for module in ("inspect_ai", "openai", "invarlock_addins.inspect_judge"):
        try:
            available = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            available = False
        if available or module in sys.modules:
            raise RuntimeError(f"core-only rehearsal found optional module: {module}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="invarlock")
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    require_core_only()
    executable = shutil.which(args.cli)
    if executable is None:
        raise SystemExit(
            "Install the candidate wheel and supply its invarlock executable"
        )
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "INVARLOCK_SIGNING_KEY", "OPENAI_API_KEY"):
        environment.pop(key, None)
    environment.update(PYTHONSAFEPATH="1", PYTHONNOUSERSITE="1")
    with tempfile.TemporaryDirectory(prefix="invarlock-judge-smoke-") as directory:
        root = Path(directory).resolve()
        for name in FIXTURE_FILES:
            shutil.copyfile(args.fixture / name, root / name)

        def run(*arguments: str, expected: int = 0) -> dict:
            result = subprocess.run(
                [executable, *arguments, "--json"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != expected:
                raise RuntimeError(
                    f"{arguments[0]} returned {result.returncode}, expected {expected}: "
                    f"{result.stdout}{result.stderr}"
                )
            return json.loads(result.stdout)

        def load(name: str) -> dict:
            return json.loads((root / f"{name}.json").read_bytes())

        plan, measurements, policy = (
            load("plan"),
            load("measurements"),
            load("analysis_policy"),
        )
        analysis = analyze_measurements(
            plan,
            measurements,
            decode_analysis_policy(policy, plan=plan),
            baseline_run=load("baseline_run"),
            subject_run=load("subject_run"),
        ).to_dict()
        signer = run("evaluate", "--keygen", "signer")["details"]
        recipient = {
            "format": "invarlock/judge-measurement-recipient-policy-v1",
            "decision_scope": "bounded-judge-fixed-benchmark-v1",
            "intended_subject": load("subject_run")["artifact_digest"],
            "required_metric_name": policy["metric_name"],
            "trusted_signer": {
                "identity": "example-signer",
                "public_key_sha256": signer["public_key_fingerprint"],
            },
            "bindings": {
                "baseline_run_sha256": plan["baseline_run_sha256"],
                "subject_run_sha256": plan["subject_run_sha256"],
                "case_set_sha256": plan["case_set_sha256"],
                "plan_sha256": measurement_plan_digest(plan),
                "measurements_sha256": object_sha256(measurements),
                "analysis_policy_sha256": object_sha256(policy),
                "analysis_result_sha256": object_sha256(analysis),
            },
            "required_decision": "pass",
        }
        (root / "recipient.json").write_text(json.dumps(recipient))
        created = run(
            "evaluate", "request.yaml", "--signing-key", signer["private_key"]
        )
        assert created["kind"] == "judge" and created["authentication"] == "signed"
        assert created["decision"] == "insufficient_evidence"
        evidence = root / "evidence"
        original = {p.name: p.read_bytes() for p in evidence.iterdir()}
        verified = run(
            "verify",
            "evidence",
            "--trust-profile",
            "recipient.json",
            "--receipt",
            "receipt.json",
            expected=7,
        )
        assert (
            verified["verified"] and verified["authenticated"] and verified["replayed"]
        )
        assert (
            not verified["accepted"] and verified["decision"] == "insufficient_evidence"
        )
        assert load("receipt")["bindings"] == recipient["bindings"]
        report = run(
            "report",
            "evidence",
            "--html",
            "report.html",
            "--markdown",
            "report.md",
            "--junit",
            "report.xml",
        )
        assert report["kind"] == "judge" and report["ok"]
        for extension in ("html", "md", "xml"):
            assert (root / f"report.{extension}").read_bytes()
        for name in ("signer", "plan", "subject"):
            wrong = copy.deepcopy(recipient)
            if name == "signer":
                wrong["trusted_signer"]["public_key_sha256"] = "sha256:" + "0" * 64
            elif name == "plan":
                wrong["bindings"]["plan_sha256"] = "0" * 64
            else:
                wrong["intended_subject"] = "sha256:" + "0" * 64
            path = f"wrong-{name}.json"
            (root / path).write_text(json.dumps(wrong))
            rejected = run("verify", "evidence", "--trust-profile", path, expected=4)
            assert not rejected["verified"] and not rejected["accepted"]
        run("evaluate", "request.yaml", "--unsigned", "--output", "unsigned")
        unsigned = run(
            "verify", "unsigned", "--trust-profile", "recipient.json", expected=4
        )
        assert not unsigned["authenticated"] and not unsigned["accepted"]
        assert original == {p.name: p.read_bytes() for p in evidence.iterdir()}
        require_core_only()
        print(
            "judge: core-only signed replay, insufficient-evidence gate, reports and independent-pin rejections pass"
        )


if __name__ == "__main__":
    main()
