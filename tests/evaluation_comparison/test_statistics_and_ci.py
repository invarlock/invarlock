"""Cross-check arithmetic, uncertainty, missingness and executable CI behavior."""

import copy
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.cli import app
from tests._evaluation_support import (
    EvaluationRecordsError,
    captured_request_digest,
    compare_runs,
    digest,
    example_project,
    materialize_captured_request,
)


def test_scalar_unit_conversion_preserves_decision_and_scales_interval():
    base, candidate, policy = example_project("judge")
    for i, row in enumerate(candidate["records"]):
        row["scores"]["latency_ms"] += i % 3
    original = compare_runs(base, candidate, policy)
    p2, b2, c2 = copy.deepcopy((policy, base, candidate))
    p2["metrics"][1]["unit"] = "seconds"
    p2["metrics"][1]["accepted_provenance"]["unit"] = "seconds"
    for key in ("maximum_regression", "maximum_interval_width", "subject_maximum"):
        p2["metrics"][1][key] /= 1000
    for run in (b2, c2):
        run["score_provenance"]["latency_ms"]["unit"] = "seconds"
        for row in run["records"]:
            row["scores"]["latency_ms"] /= 1000
    converted = compare_runs(b2, c2, p2)
    assert original["decision"] == converted["decision"]
    for key in ("lower", "upper"):
        assert converted["metrics"][1]["interval"][key] == pytest.approx(
            original["metrics"][1]["interval"][key] / 1000
        )


def test_swapping_paired_runs_reverses_delta_and_interval():
    base, candidate, policy = example_project("judge")
    for i, row in enumerate(candidate["records"]):
        row["scores"]["quality"] += (i % 5 - 2) / 100
    forward = compare_runs(base, candidate, policy)["metrics"][0]
    reverse = compare_runs(candidate, base, policy)["metrics"][0]
    assert forward["delta"] == pytest.approx(-reverse["delta"])
    assert forward["interval"]["lower"] == pytest.approx(-reverse["interval"]["upper"])
    assert forward["interval"]["upper"] == pytest.approx(-reverse["interval"]["lower"])


def test_latency_regression_rejects_candidate_below_absolute_ceiling():
    base, candidate, policy = example_project("judge")
    for row in candidate["records"]:
        row["scores"]["latency_ms"] = 150.0
    result = compare_runs(base, candidate, policy)
    latency = result["metrics"][1]
    assert latency["subject_mean"] < policy["metrics"][1]["subject_maximum"]
    assert latency["decision"] == result["decision"] == "regression"
    assert latency["reasons"] == ["upper interval bound exceeds allowed regression"]


def test_wide_interval_cannot_pass_an_acceptable_mean():
    base, candidate, policy = example_project("judge")
    policy["metrics"][0]["maximum_interval_width"] = 0.001
    for i, row in enumerate(candidate["records"]):
        row["scores"]["quality"] += 0.05 if i % 2 else -0.05
    result = compare_runs(base, candidate, policy)
    quality = result["metrics"][0]
    assert quality["delta"] == pytest.approx(0)
    assert quality["decision"] == result["decision"] == "insufficient_evidence"
    assert quality["reasons"] == ["interval is too wide"]


def test_invalid_reference_identifies_metric_and_record():
    base, candidate, policy = example_project("extraction")
    for run in (base, candidate):
        del run["records"][0]["expected"]["currency"]
    with pytest.raises(
        EvaluationRecordsError, match="metric quality, record case-0: reference"
    ):
        compare_runs(base, candidate, policy)


@pytest.mark.parametrize(
    "mutation", ["unit", "nan", "boolean", "empty_slice", "missing_score"]
)
def test_no_false_pass_from_invalid_or_incomplete_measurements(mutation):
    base, candidate, policy = example_project("judge")
    if mutation == "unit":
        candidate["score_provenance"]["quality"]["unit"] = "percent"
    elif mutation in ("nan", "boolean"):
        candidate["records"][0]["scores"]["quality"] = (
            float("nan") if mutation == "nan" else True
        )
    elif mutation == "empty_slice":
        policy["slices"][0]["where"] = {"category": "nonexistent"}
    else:
        del candidate["records"][0]["scores"]["quality"]
    if mutation in ("unit", "nan", "boolean"):
        with pytest.raises(EvaluationRecordsError):
            compare_runs(base, candidate, policy)
    else:
        assert (
            compare_runs(base, candidate, policy)["decision"] == "insufficient_evidence"
        )


def test_real_cli_sign_verify_and_all_decision_exit_codes(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    assert (
        runner.invoke(
            app, ["evaluate", "--init", str(project), "--example", "judge"]
        ).exit_code
        == 0
    )
    key_directory = tmp_path / "keys"
    key = key_directory / "private.pem"
    generated = runner.invoke(
        app, ["evaluate", "--keygen", str(key_directory), "--json"]
    )
    assert generated.exit_code == 0, generated.output
    assert key.stat().st_mode & 0o077 == 0
    signer = json.loads(generated.stdout)["details"]["public_key_fingerprint"]
    assert (
        runner.invoke(
            app, ["evaluate", "--keygen", str(key_directory), "--json"]
        ).exit_code
        == 2
    )
    request = project / "request.yaml"
    request_value = json.loads(request.read_text())
    request_value["output"]["evidence"] = "signed"
    request.chmod(0o644)
    request.write_text(json.dumps(request_value))
    compared = runner.invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(key),
            "--json",
        ],
    )
    assert compared.exit_code == 0, compared.output
    base = json.loads((project / "inputs/baseline.json").read_text())
    candidate = json.loads((project / "inputs/subject.json").read_text())
    policy = json.loads((project / "policy.json").read_text())
    verifier_directory = tmp_path / "verifier"
    assert (
        runner.invoke(
            app, ["evaluate", "--keygen", str(verifier_directory), "--json"]
        ).exit_code
        == 0
    )
    args = [
        "verify",
        str(project / "signed"),
        "--policy",
        str(project / "policy.json"),
        "--expected-baseline-run",
        digest(base),
        "--expected-subject-run",
        digest(candidate),
        "--expected-request-digest",
        captured_request_digest(base, candidate, policy),
        "--expected-signer",
        signer,
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--verifier-signing-key",
        str(verifier_directory / "private.pem"),
        "--verifier-identity",
        "statistics-test",
        "--json",
    ]
    verified = runner.invoke(app, args)
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["ok"] is True
    args[args.index("--expected-subject-run") + 1] = "sha256:" + "a" * 64
    args[args.index("--receipt") + 1] = str(tmp_path / "bad-receipt.json")
    assert runner.invoke(app, args).exit_code == 6
    for row in candidate["records"]:
        row["scores"]["quality"] = 0.1
    subject_path = project / "inputs/subject.json"
    subject_path.chmod(0o644)
    subject_path.write_text(json.dumps(candidate))
    request_value["output"]["evidence"] = "regression"
    request.chmod(0o644)
    request.write_text(json.dumps(request_value))
    regressed = runner.invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(key),
            "--json",
        ],
    )
    assert regressed.exit_code == 0, regressed.output
    assert json.loads(regressed.stdout)["policy_verdict"] == "regression"
    args[1] = str(project / "regression")
    args[args.index("--expected-subject-run") + 1] = digest(candidate)
    args[args.index("--expected-request-digest") + 1] = captured_request_digest(
        base, candidate, policy
    )
    args[args.index("--receipt") + 1] = str(tmp_path / "regression-receipt.json")
    rejected = runner.invoke(app, args)
    assert rejected.exit_code == 7, rejected.output
    assert json.loads(rejected.stdout)["decision"] == "regression"
    assert (tmp_path / "regression-receipt.json").is_file()

    for metric in policy["metrics"]:
        metric["minimum_count"] = 100
    policy_path = project / "policy.json"
    policy_path.chmod(0o644)
    policy_path.write_text(json.dumps(policy, indent=2))
    request_value["output"]["evidence"] = "insufficient"
    request.write_text(json.dumps(request_value))
    insufficient = runner.invoke(
        app, ["evaluate", str(request), "--signing-key", str(key), "--json"]
    )
    assert insufficient.exit_code == 0, insufficient.output
    assert json.loads(insufficient.stdout)["policy_verdict"] == "insufficient_evidence"
    args[1] = str(project / "insufficient")
    args[args.index("--expected-request-digest") + 1] = captured_request_digest(
        base, candidate, policy
    )
    args[args.index("--receipt") + 1] = str(tmp_path / "insufficient-receipt.json")
    rejected = runner.invoke(app, args)
    assert rejected.exit_code == 7, rejected.output
    assert json.loads(rejected.stdout)["decision"] == "insufficient_evidence"
    assert (tmp_path / "insufficient-receipt.json").is_file()


@pytest.mark.parametrize("decision", ["pass", "regression", "insufficient_evidence"])
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("gate", [False, True])
def test_cli_policy_gate_preserves_published_evidence(
    tmp_path, decision, unsigned, gate
):
    base, subject, policy = example_project("judge")
    if decision == "regression":
        for row in subject["records"]:
            row["scores"]["quality"] = 0.1
    elif decision == "insufficient_evidence":
        for metric in policy["metrics"]:
            metric["minimum_count"] = 100
    root = tmp_path / "approved"
    request = materialize_captured_request(root, base, subject, policy)
    key = tmp_path / "key.pem"
    key.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    args = ["evaluate", str(request), "--json"]
    args.extend(["--unsigned"] if unsigned else ["--signing-key", str(key)])
    if gate:
        args.append("--fail-on-policy")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (7 if gate and decision != "pass" else 0), result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["policy_verdict"] == decision
    assert (root / "evidence/manifest.json").is_file()
    assert (root / "evidence/manifest.signature.json").exists() is not unsigned


def test_preflight_cannot_apply_policy_gate(tmp_path):
    base, subject, policy = example_project("judge")
    request = materialize_captured_request(tmp_path / "approved", base, subject, policy)
    result = CliRunner().invoke(
        app, ["evaluate", str(request), "--unsigned", "--preflight", "--fail-on-policy"]
    )
    assert result.exit_code == 2
    assert not (request.parent / "evidence").exists()
