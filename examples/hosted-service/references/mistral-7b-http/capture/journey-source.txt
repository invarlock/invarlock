"""Pinned hosted captures through separate installed operator and recipient CLIs.

Copies only imported evidence and recipient-owned inputs across the handoff.
Never contacts a service or discovers trust anchors inside an evidence bundle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def capture_module():
    spec = importlib.util.spec_from_file_location(
        "hosted_capture", Path(__file__).with_name("capture.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    return {
        **{
            name: value
            for name, value in os.environ.items()
            if name in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
        },
        "PYTHONSAFEPATH": "1",
    }


def command(cli, cwd, *arguments, expected=(0,)):
    result = subprocess.run(
        [str(cli), *map(str, arguments)],
        cwd=cwd,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode not in expected:
        # Do not include subprocess logs: imported text may be sensitive.
        raise RuntimeError(
            f"{arguments[0]} returned unexpected status {result.returncode}"
        )
    return json.loads(result.stdout)


def require_wheel(cli):
    cli = Path(cli).absolute()
    python = cli.with_name("python")
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib.metadata,json,pathlib,sys,invarlock; "
                "d=importlib.metadata.distribution('invarlock'); "
                "print(json.dumps({'module':str(pathlib.Path(invarlock.__file__).resolve()),"
                "'prefix':sys.prefix,'direct_url':d.read_text('direct_url.json')}))"
            ),
        ],
        cwd=cli.parent,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    value = json.loads(probe.stdout)
    module_path = Path(value["module"])
    direct_url = json.loads(value["direct_url"] or "{}")
    if (
        not module_path.is_relative_to(Path(value["prefix"]))
        or "site-packages" not in module_path.parts
        or direct_url.get("dir_info", {}).get("editable", False)
    ):
        raise ValueError("operator and recipient require installed wheels")
    return cli


def prepare(
    protocol,
    protocol_sha256,
    baseline,
    baseline_sha256,
    subject,
    subject_sha256,
    output,
):
    from invarlock.engine import (
        captured_request_digest,
        normalize_captured_request,
        run_digest,
    )

    helper = capture_module()
    declaration = helper.load_protocol(protocol, protocol_sha256)
    helper.checked(
        helper.digest(helper.read(Path(__file__)))
        == declaration["journey_source_digest"],
        "journey source differs from protocol",
    )
    helper.checked(
        helper.digest(helper.read(Path(__file__).with_name("capture.py")))
        == declaration["collector_source_digest"],
        "collector source differs from protocol",
    )
    runs = {
        role: helper.export_run(protocol, protocol_sha256, path, pin, role)
        for role, path, pin in (
            ("baseline", baseline, baseline_sha256),
            ("subject", subject, subject_sha256),
        )
    }

    def parse_time(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    if parse_time(
        runs["baseline"]["service_identity"]["observation_window"]["ended_at"]
    ) > parse_time(
        runs["subject"]["service_identity"]["observation_window"]["started_at"]
    ):
        raise ValueError("baseline must end before subject starts")
    output = helper.create_directory(output)
    for role, run in runs.items():
        helper.write(output / f"{role}.json", run)
    policy = declaration["policy"]
    helper.write(output / "policy.json", policy)
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            **{
                role: {
                    "path": f"{role}.json",
                    "adapter": "invarlock",
                    "expected_run_digest": run_digest(run),
                }
                for role, run in runs.items()
            },
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    normalized = normalize_captured_request(
        request, baseline=runs["baseline"], subject=runs["subject"], policy=policy
    )
    helper.write(output / "request.json", request)
    anchors = {
        **{f"{role}_run_digest": run_digest(run) for role, run in runs.items()},
        "request_digest": captured_request_digest(normalized),
    }
    helper.write(output / "anchors.json", anchors)
    return anchors


def journey(
    *,
    operator_cli,
    recipient_cli,
    protocol,
    protocol_sha256,
    baseline,
    baseline_sha256,
    subject,
    subject_sha256,
    output,
):
    helper = capture_module()
    operator_cli, recipient_cli = (
        require_wheel(operator_cli),
        require_wheel(recipient_cli),
    )
    helper.checked(
        operator_cli.parent.resolve() != recipient_cli.parent.resolve(),
        "operator and recipient must use separate wheel environments",
    )
    output = helper.create_directory(output)
    operator, recipient = output / "operator", output / "recipient"
    inputs = {
        "protocol": str(Path(protocol).absolute()),
        "protocol_sha256": protocol_sha256,
        "baseline": str(Path(baseline).absolute()),
        "baseline_sha256": baseline_sha256,
        "subject": str(Path(subject).absolute()),
        "subject_sha256": subject_sha256,
    }
    expectations = []
    for cli, destination in ((operator_cli, operator), (recipient_cli, recipient)):
        prepared = subprocess.run(
            [
                str(cli.with_name("python")),
                "-I",
                str(Path(__file__).resolve()),
                "_prepare",
            ],
            input=helper.encoded({**inputs, "output": str(destination)}),
            capture_output=True,
            env=environment(),
            cwd=output,
            timeout=300,
            check=True,
        )
        expectations.append(json.loads(prepared.stdout))
    helper.checked(
        expectations[0] == expectations[1], "independent import expectations differ"
    )
    anchors = expectations[1]
    signer = command(
        operator_cli, operator, "evaluate", "--keygen", "signer", "--json"
    )["details"]
    verifier = command(
        recipient_cli, recipient, "evaluate", "--keygen", "verifier", "--json"
    )["details"]
    created = command(
        operator_cli,
        operator,
        "evaluate",
        "request.json",
        "--signing-key",
        signer["private_key"],
        "--json",
    )
    helper.checked(
        created["ok"] is True and created["authentication"] == "signed",
        "signed publication failed",
    )
    shutil.copytree(operator / "evidence", recipient / "evidence")
    profile = {
        "format": "invarlock/trust-inputs-v2",
        "kind": "captured",
        "policy": {"path": "policy.json"},
        "anchors": {
            **anchors,
            "evidence_signer_fingerprint": signer["public_key_fingerprint"],
        },
        "verifier": {
            "identity": "hosted-service-recipient",
            "signing_key_path": "verifier/private.pem",
        },
    }
    helper.write(recipient / "trust.json", profile)
    verified = command(
        recipient_cli,
        recipient,
        "verify",
        "evidence",
        "--trust-profile",
        "trust.json",
        "--receipt",
        "verification.receipt.json",
        "--json",
        expected=(0, 7),
    )
    helper.checked(
        verified["integrity_ok"] is True and verified["replay_status"] == "completed",
        "independent replay failed",
    )
    report = command(
        recipient_cli,
        recipient,
        "report",
        "evidence",
        "--html",
        "report.html",
        "--markdown",
        "report.md",
        "--junit",
        "report.xml",
        "--json",
    )
    result = {
        "source_assurance": "captured_inputs",
        "scope": "temporal-service-observations",
        "anchors": anchors,
        "signer_fingerprint": signer["public_key_fingerprint"],
        "verifier_fingerprint": verifier["public_key_fingerprint"],
        "evaluation": created,
        "verification": verified,
        "report": report,
    }
    helper.write(output / "result.json", result)
    return result


def main():
    if sys.argv[1:] == ["_prepare"]:
        values = json.loads(sys.stdin.buffer.read(65536))
        print(json.dumps(prepare(**values)))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "operator-cli",
        "recipient-cli",
        "protocol",
        "baseline",
        "subject",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha256", "baseline-sha256", "subject-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = journey(**vars(args))
    print(
        json.dumps(
            {
                "decision": result["evaluation"]["decision"],
                "integrity_ok": result["verification"]["integrity_ok"],
                "source_assurance": result["source_assurance"],
            }
        )
    )


if __name__ == "__main__":
    main()
