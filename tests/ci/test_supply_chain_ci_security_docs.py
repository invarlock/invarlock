from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml


def _load(path: str) -> dict[str, Any]:
    workflow = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _step(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_secret_history_workflow_runs_a_scheduled_full_history_scan() -> None:
    workflow = _load(".github/workflows/secret-history.yml")
    assert workflow["on"]["schedule"] == [{"cron": "17 9 * * 1"}]
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["gitleaks-history"]
    checkout = _step(job["steps"], "Checkout repository")
    assert checkout["with"]["fetch-depth"] == 0

    scan = _step(job["steps"], "Run gitleaks full history scan")
    assert "gitleaks git ." in scan["run"]
    assert "--config .gitleaks.toml" in scan["run"]
    assert "--log-opts" not in scan["run"]


def test_gitleaks_allowlists_require_the_complete_detected_field_match() -> None:
    config = tomllib.loads(Path(".gitleaks.toml").read_text(encoding="utf-8"))
    allowlists = config["allowlists"]

    digest_rule = next(
        item for item in allowlists if "tokenizer digest" in item["description"]
    )
    assert digest_rule["condition"] == "AND"
    assert digest_rule["regexTarget"] == "match"
    assert any(
        re.search(pattern, 'tokenizer_metadata_sha256":"' + "a" * 32 + '"')
        for pattern in digest_rule["regexes"]
    )
    assert any(
        re.fullmatch(
            pattern,
            "public_evidence/evidence/example/evidence/request.json",
        )
        for pattern in digest_rule["paths"]
    )
    for path in (
        "examples/evaluator-qualification/signed-transactions/qwen35-inspect-ai/"
        "evidence/request.json",
        "tests/fixtures/compatibility/v0.13.0/package/evidence/request.json",
    ):
        assert any(re.fullmatch(pattern, path) for pattern in digest_rule["paths"])
    assert not any(
        re.search(pattern, 'api_key":"' + "a" * 32 + '"')
        for pattern in digest_rule["regexes"]
    )

    type_rule = next(
        item for item in allowlists if "type annotations" in item["description"]
    )
    assert type_rule["condition"] == "AND"
    assert type_rule["regexTarget"] == "match"
    type_annotation = (
        "_".join(("signing", "key")) + ": " + ".".join(("ed25519", "Ed25519PrivateKey"))
    )
    assert any(re.search(pattern, type_annotation) for pattern in type_rule["regexes"])


def test_end_of_file_hook_preserves_canonical_signed_evidence_bytes() -> None:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    end_of_file = next(hook for hook in hooks if hook["id"] == "end-of-file-fixer")
    excluded = re.compile(end_of_file["exclude"])

    for path in (
        "public_evidence/evidence/example/manifest.json",
        "examples/evaluator-qualification/signed-transactions/qwen35-inspect-ai/"
        "build-attestation.json",
        "examples/integrations/spdx-ai-observation/source/model-aibom.spdx3.json",
        "examples/integrations/spdx-ai-observation/source/model-artifact.identity.json",
        "tests/fixtures/compatibility/v0.13.0/package/verification.receipt.json",
    ):
        assert excluded.search(path)
    assert not excluded.search(
        "examples/evaluator-qualification/signed-transactions/README.md"
    )
    assert not excluded.search(
        "examples/ci/standalone-consumer/review/"
        "inspect-ai-deployment-approval-inputs.json"
    )
    assert not excluded.search(
        "examples/integrations/spdx-ai-observation/observation-payload.json"
    )


def test_pipeline_annotation_allowance_requires_exact_path_and_empty_default() -> None:
    config = tomllib.loads(Path(".gitleaks.toml").read_text(encoding="utf-8"))
    allowance = next(
        item
        for item in config["allowlists"]
        if "pipeline signing-key" in item["description"]
    )
    assert allowance["condition"] == "AND"
    assert allowance["regexTarget"] == "line"
    assert allowance["targetRules"] == ["generic-api-key"]
    path = "src/invarlock/pipeline/evidence.py"
    name = "_".join(("signing", "key"))
    annotation = "Ed25519" + "PrivateKey"
    line = f"    {name}: {annotation} | None = None,"
    assert any(re.search(pattern, path) for pattern in allowance["paths"])
    for detected_line in (line, "\n" + line):
        assert any(
            re.search(pattern, detected_line) for pattern in allowance["regexes"]
        )
    for other_path in (
        f"prefix/{path}",
        f"{path}.bak",
        path.replace("evidence", "other"),
    ):
        assert not any(re.search(pattern, other_path) for pattern in allowance["paths"])
    for other_line in (
        line.replace("= None", '= "credential"'),
        line + ' api_key = "credential"',
        line.replace(name, "private_key"),
        f'{name} = "credential"',
    ):
        assert not any(
            re.search(pattern, other_line) for pattern in allowance["regexes"]
        )


def test_gitleaks_still_detects_credentials_on_pipeline_annotation_path(
    tmp_path: Path,
) -> None:
    executable = shutil.which("gitleaks")
    if executable is None:
        pytest.skip("the pinned Gitleaks executable is required for the scanner probe")
    config = Path(".gitleaks.toml").resolve()
    relative = Path("src/invarlock/pipeline/evidence.py")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    name = "_".join(("signing", "key"))
    annotation = "Ed25519" + "PrivateKey"
    canary = hashlib.sha256(b"invarlock synthetic scanner boundary probe").hexdigest()
    source.write_text(
        "# Synthetic scanner boundary probe.\n"
        f"    {name}: {annotation} | None = None,\n"
        f'{name} = "{canary}"\n'
        f'    {name}: {annotation} | None = "{canary}",\n',
        encoding="utf-8",
    )
    report = tmp_path / "findings.json"
    result = subprocess.run(
        [
            executable,
            "dir",
            ".",
            "--config",
            str(config),
            "--redact",
            "--no-banner",
            "--report-format",
            "json",
            "--report-path",
            str(report),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    findings = json.loads(report.read_text(encoding="utf-8"))
    assert {(item["File"], item["StartLine"]) for item in findings} == {
        (relative.as_posix(), 3),
        (relative.as_posix(), 4),
    }


def test_full_ci_pins_make_to_setup_python() -> None:
    workflow = _load(".github/workflows/ci.yml")
    full = workflow["jobs"]["verify-full"]
    assert full["env"]["PYTHON"] == "python"
    assert _step(full["steps"], "Run complete repository gates")["run"] == (
        "make verify"
    )
