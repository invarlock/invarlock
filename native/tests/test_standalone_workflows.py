"""Public offline commands run without a Python executable in their environment."""

import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from invarlock.client_review import build_client_review
from invarlock.core.evaluation_request import _load_yaml, load_evaluation_request
from invarlock.evaluation_transaction import (
    evaluate_request_file,
    preflight_evaluation_request,
)
from invarlock.evidence_reporting import render_evidence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/rust/python"))
full_fixtures = importlib.import_module("invarlock_rust_experiment.full_fixtures")


BIN = Path(os.environ["CARGO_TARGET_DIR"]) / "debug/invarlock-native-verify"


def run(*args):
    return subprocess.run(
        [str(BIN), *map(str, args)],
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent", "HOME": os.environ["HOME"]},
    )


def files(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture(params=["captured", "native", "judge", "evidence_set"])
def pack(request, tmp_path):
    root = tmp_path.resolve() / "fixture"
    cfg = (
        full_fixtures.captured(root, 12, "pass")
        if request.param == "captured"
        else getattr(full_fixtures, request.param)(root)
    )
    return Path(cfg["steps"][0]["config"]["evidence_path"])


@pytest.mark.parametrize("rich", [False, True])
def test_package_parity(pack, tmp_path, rich):
    context = {
        "format": "invarlock/client-review-context-v1",
        "title": 'Review <x> & "ü"',
        "review_date": "2026-10-28",
        "byline": "Owner",
        "context": [["Use", "Retained evidence"]],
        "next_action": "Verify independently",
    }
    if rich:
        context.update(
            label="Review _ label",
            recommendation="Long context " * 80,
            decision="Pending <approval>",
            basis=[["Scope", "A | B"]],
        )
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    expected = tmp_path / "python"
    actual = tmp_path / "rust"
    reference = build_client_review(pack, context_path=path, output=expected)
    result = run(
        "report", pack, "--package", actual, "--review-context", path, "--json"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    for field in ["output", "brief", "archive"]:
        reference[field] = str(actual) + reference[field][len(str(expected)) :]
    assert payload == reference
    before, after = files(expected), files(actual)
    assert before.keys() == after.keys()
    for name in before:
        if name not in {"package-manifest.json", "handoff.zip"}:
            assert before[name] == after[name], name
    left, right = [json.loads(x["package-manifest.json"]) for x in [before, after]]
    assert left.pop("generator_source_sha256") != right.pop("generator_source_sha256")
    assert left == right
    with zipfile.ZipFile(actual / "handoff.zip") as archive:
        assert set(archive.namelist()) == set(after) - {"handoff.zip"}
        assert all(archive.read(name) == after[name] for name in archive.namelist())
    assert (
        run(
            "report", pack, "--package", actual, "--review-context", path, "--json"
        ).returncode
        != 0
    )
    assert files(actual) == after


def test_report_parity(pack, tmp_path):
    render_evidence(
        pack,
        html_path=tmp_path / "python.html",
        markdown_path=tmp_path / "python.md",
        junit_path=tmp_path / "python.xml",
        explain=True,
    )
    result = run(
        "report",
        pack,
        "--html",
        tmp_path / "rust.html",
        "--markdown",
        tmp_path / "rust.md",
        "--junit",
        tmp_path / "rust.xml",
        "--explain",
        "--json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for ext in ["html", "md", "xml"]:
        assert (tmp_path / f"python.{ext}").read_bytes() == (
            tmp_path / f"rust.{ext}"
        ).read_bytes()


@pytest.mark.parametrize("scenario", ["pass", "regression"])
@pytest.mark.parametrize("unsigned", [False, True])
def test_evaluate_parity(tmp_path, scenario, unsigned):
    root = tmp_path.resolve() / "fixture"
    full_fixtures.captured(root, 12, scenario)
    options = (
        {"unsigned": True, "signing_key_path": None}
        if unsigned
        else {"signing_key_path": root / "signer.pem"}
    )
    native_options = (
        ["--unsigned"] if unsigned else ["--signing-key", str(root / "signer.pem")]
    )
    request = load_evaluation_request(root / "request.json", output=root / "python")
    expected_pre = json.loads(
        preflight_evaluation_request(request, **options).as_json()
    )
    actual_pre = run(
        "evaluate",
        root / "request.json",
        "--output",
        root / "rust",
        "--preflight",
        *native_options,
        "--json",
    )
    assert actual_pre.returncode == 0, actual_pre.stdout + actual_pre.stderr
    expected_pre["output"] = str(root / "rust")
    assert json.loads(actual_pre.stdout) == expected_pre
    expected = json.loads(evaluate_request_file(request, **options).as_json())
    result = run(
        "evaluate",
        root / "request.json",
        "--output",
        root / "rust",
        *native_options,
        "--json",
        "--fail-on-policy",
    )
    assert result.returncode == (0 if expected["decision"] == "pass" else 7), (
        result.stdout + result.stderr
    )
    expected["evidence"] = str(root / "rust")
    assert json.loads(result.stdout) == expected
    assert files(root / "python") == files(root / "rust")


@pytest.mark.parametrize(
    "mutation", ["symlink", "fifo", "context-date", "context-fields", "context-empty"]
)
def test_package_refusals(tmp_path, mutation):
    root = tmp_path.resolve() / "fixture"
    full_fixtures.captured(root, 12, "pass")
    context = {
        "format": "invarlock/client-review-context-v1",
        "title": "Review",
        "review_date": "2026-10-28",
        "byline": "Owner",
        "context": [["Use", "Check"]],
        "next_action": "Check",
    }
    if mutation == "symlink":
        (root / "pack" / "injected").symlink_to(root / "policy.json")
    if mutation == "fifo":
        os.mkfifo(root / "pack" / "injected")
    if mutation == "context-date":
        context["review_date"] = "2026-02-30"
    if mutation == "context-fields":
        context["unknown"] = 1
    if mutation == "context-empty":
        context["context"] = []
    path = root / "context.json"
    path.write_text(json.dumps(context))
    output = root / "output"
    result = run(
        "report", root / "pack", "--package", output, "--review-context", path, "--json"
    )
    assert result.returncode != 0
    assert not output.exists()
    assert not list(root.glob(".invarlock-stage-*"))


@pytest.mark.parametrize("family", ["captured", "native", "judge", "evidence_set"])
def test_verify_public_cli_parity(tmp_path, family):
    root = tmp_path.resolve() / "fixture"
    cfg = (
        full_fixtures.captured(root, 12, "pass")
        if family == "captured"
        else getattr(full_fixtures, family)(root)
    )
    first = cfg["steps"][0]["config"]
    pack = Path(first["evidence_path"])
    args = []
    if family in ["judge", "evidence_set"]:
        args += ["--trust-profile", first["recipient_policy_path"]]
    else:
        for key, value in first.items():
            if key.startswith("expected_"):
                args += ["--" + key.replace("_", "-"), value]
        args += ["--policy", first["policy_path"]]
    if family == "judge":
        receipt_cfg = cfg["steps"][1]["config"]
        args += [
            "--verifier-signing-key",
            receipt_cfg["verifier_signing_key_path"],
            "--verifier-identity",
            receipt_cfg["verifier_identity"],
        ]
    elif family != "evidence_set":
        args += [
            "--verifier-signing-key",
            first["verifier_signing_key_path"],
            "--verifier-identity",
            first["verifier_identity"],
        ]
    before = subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "verify",
            str(pack),
            *args,
            "--receipt",
            str(root / "python-receipt.json"),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    after = run(
        "verify", pack, *args, "--receipt", root / "rust-receipt.json", "--json"
    )
    assert after.returncode == before.returncode, (
        before.stdout,
        before.stderr,
        after.stdout,
        after.stderr,
    )
    left = json.loads(before.stdout)
    right = json.loads(after.stdout)

    def normalized(v):
        if isinstance(v, dict):
            return {k: normalized(x) for k, x in v.items()}
        if isinstance(v, list):
            return [normalized(x) for x in v]
        if isinstance(v, str):
            return v.replace("python-receipt.json", "receipt.json").replace(
                "rust-receipt.json", "receipt.json"
            )
        return v

    assert normalized(left) == normalized(right)
    assert (root / "python-receipt.json").read_bytes() == (
        root / "rust-receipt.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "example", ["classification", "extraction", "judge", "native-judge"]
)
def test_setup_starters(tmp_path, example):
    from invarlock.cli.evaluation_setup import starter_artifacts

    output = tmp_path / "starter"
    result = run("evaluate", "--init", output, "--example", example, "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert files(output) == starter_artifacts(example)
    assert run("evaluate", "--init", output, "--json").returncode == 2


def test_keygen_and_cases(tmp_path):
    from cryptography.hazmat.primitives import serialization

    from invarlock.evidence_pack_integrity import public_key_fingerprint

    result = run("evaluate", "--keygen", tmp_path / "keys", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    key = serialization.load_pem_private_key(
        (tmp_path / "keys/private.pem").read_bytes(), None
    )
    public = serialization.load_pem_public_key(
        (tmp_path / "keys/public.pem").read_bytes()
    )
    assert key.public_key().public_bytes_raw() == public.public_bytes_raw()
    assert (
        public_key_fingerprint(public)
        == json.loads(result.stdout)["details"]["public_key_fingerprint"]
    )
    assert (tmp_path / "keys/private.pem").stat().st_mode & 0o777 == 0o600
    from invarlock.evaluation_records.cases import canonical_case_set, case_set_digest
    from invarlock.evidence_pack_contract import canonical_json_bytes

    value = {
        "format": "invarlock/evaluation-case-set-v1",
        "cases": [
            {"id": i, "input": "q", "expected": "a", "metadata": {}} for i in ["b", "a"]
        ],
    }
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(value))
    out = tmp_path / "canonical.json"
    result = run(
        "evaluate", "--freeze-cases", source, "--case-set-output", out, "--json"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.read_bytes() == canonical_json_bytes(canonical_case_set(value))
    assert json.loads(result.stdout)["details"]["case_set_digest"] == case_set_digest(
        value
    )


@pytest.mark.parametrize(
    "text",
    [
        "a: true",
        "a: false",
        "a: null",
        "a: 1",
        "a: -1",
        "a: 1.25",
        "a: 1e3",
        'a: "01"',
        "a: [1, 2]",
        "a: |\n  text\n",
        "a: yes",
        "a: True",
        "a: ~",
        "a: 01",
        "a: +1",
        "a: .inf",
        "a: 2026-01-01",
        "a: !!str text",
        "a: &ref value\nb: *ref",
        "a: 1\na: 2",
        "a: {<<: {b: 1}}",
        "1: value",
        "---\na: 1\n---\nb: 2",
    ],
)
def test_request_yaml_semantics(tmp_path, text):
    path = tmp_path / "request.yaml"
    path.write_text(text)
    try:
        expected = _load_yaml(text.encode())
    except ValueError:
        expected = None
    # Timestamp values parse in PyYAML, but are rejected by request value validation.
    import datetime

    if expected and any(
        isinstance(v, (datetime.date, datetime.datetime)) for v in expected.values()
    ):
        expected = None
    result = run("decode-request", "--request", path)
    if expected is None:
        assert result.returncode != 0, (text, result.stdout)
    else:
        assert result.returncode == 0 and json.loads(result.stdout) == expected, (
            text,
            result.stdout,
            result.stderr,
        )
