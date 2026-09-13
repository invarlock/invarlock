"""Run the literal hosted guide through public APIs and the signed CLI path."""

import copy
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from invarlock.engine import (
    capture_evaluator_run,
    captured_request_digest,
    digest,
    evaluated_subject_digest,
    load_evaluation_request,
    normalize_captured_request,
    run_digest,
    validate_service_identity,
)
from tests.evaluation_records.test_hosted_service_identity import identity

ROOT = Path(__file__).resolve().parents[2]


def test_hosted_guide_public_api_and_signed_commands(tmp_path):
    guide = (ROOT / "docs/user-guide/hosted-service-requalification.md").read_text()
    reference = (ROOT / "docs/reference/evaluation-records.md").read_text()
    request = yaml.safe_load(re.search(r"```yaml\n(.*?)\n```", guide, re.S)[1])
    rows = [
        {
            "id": f"case-{i}",
            "input": f"question-{i}",
            "expected": "yes",
            "output": "yes",
        }
        for i in range(100)
    ]
    descriptor = identity()
    validate_service_identity(descriptor)
    scope = {"identity": copy.deepcopy(descriptor), "rows": rows}
    exec(re.search(r"```python\n(.*?)\n```", reference, re.S)[1], scope)
    subject = scope["run"]
    assert evaluated_subject_digest(subject) == digest(descriptor)
    baseline = capture_evaluator_run(
        rows,
        source={"name": "service-capture", "version": "1"},
        run_id="baseline-campaign",
        artifact_digest=None,
        service_identity=descriptor,
    )
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "slices": [],
        "metrics": [
            {
                "name": "quality",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 100,
                "maximum_regression": 0.1,
                "maximum_interval_width": 1,
                "subject_minimum": 0.5,
            }
        ],
    }
    campaign, trust = tmp_path / "campaign", tmp_path / "trust"
    campaign.mkdir()
    trust.mkdir()

    def save(path, value):
        path.write_text(json.dumps(value) + "\n")

    save(campaign / "baseline.json", baseline)
    save(campaign / "subject.json", subject)
    save(campaign / "policy.json", policy)
    (campaign / "request.yaml").write_text(yaml.safe_dump(request))
    loaded = load_evaluation_request(campaign / "request.yaml")
    assert loaded.evidence == campaign / "evidence"
    normalized = normalize_captured_request(
        request,
        baseline=baseline,
        subject=subject,
        policy=policy,
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("INVARLOCK_")
        and name not in ("OPENAI_API_KEY", "PYTHONPATH")
    }

    def command(*arguments):
        completed = subprocess.run(
            [sys.executable, "-m", "invarlock", *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(completed.stdout)

    signer = command("evaluate", "--keygen", "signer", "--json")["details"]
    command("evaluate", "--keygen", "trust/verifier", "--json")
    (tmp_path / "evidence-signer.pem").write_bytes(
        (tmp_path / "signer/private.pem").read_bytes()
    )
    save(trust / "policy.json", policy)
    save(
        trust / "trust-inputs.json",
        {
            "format": "invarlock/trust-inputs-v2",
            "kind": "captured",
            "policy": {"path": "policy.json"},
            "anchors": {
                "baseline_run_digest": run_digest(baseline),
                "subject_run_digest": run_digest(subject),
                "request_digest": captured_request_digest(normalized),
                "evidence_signer_fingerprint": signer["public_key_fingerprint"],
            },
            "verifier": {
                "identity": "recipient-verifier",
                "signing_key_path": "verifier/private.pem",
            },
        },
    )
    commands = [
        shlex.split(line)
        for block in re.findall(r"```bash\n(.*?)\n```", guide, re.S)
        for line in block.replace("\\\n", " ").splitlines()
        if line.startswith("invarlock ")
    ]
    assert len(commands) == 4
    results = [command(*arguments[1:]) for arguments in commands]
    preflight, evaluation, verification, report = results
    assert preflight["ok"] and preflight["requested_authentication"] == "signed"
    assert evaluation["decision"] == "pass"
    assert verification["integrity_ok"] and verification["ok"]
    assert (campaign / "verification.receipt.json").is_file()
    assert report["written_outputs"] == {
        "html": "campaign/report.html",
        "markdown": "campaign/report.md",
    }
    assert (campaign / "report.html").is_file() and (campaign / "report.md").is_file()
