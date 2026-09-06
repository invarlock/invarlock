"""Externally planned membership survives paired omission and signed replay."""

import base64
import json
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline import (
    PipelineError,
    canonical_case_set,
    case_set_digest,
    cli,
    compare_runs,
    comparison,
    contracts,
    create_evidence,
    verify_evidence,
)
from invarlock.pipeline.cases import CASE_SET_FORMAT, validate_run_case_set
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import digest
from invarlock.pipeline.evidence import DOMAIN
from invarlock.pipeline.templates import example_project


def planned(run):
    return {
        "format": "invarlock/pipeline-case-set-v1",
        "cases": sorted(
            [
                {
                    key: deepcopy(row[key])
                    for key in ("id", "input", "expected", "metadata")
                }
                for row in run["records"]
            ],
            key=lambda row: row["id"],
        ),
    }


def test_planned_pin_rejects_same_case_omitted_from_both_runs():
    baseline, candidate, policy = example_project("extraction")
    pin = digest(planned(baseline))
    baseline["records"].pop()
    candidate["records"].pop()
    # Pair equality and the sample minimum alone cannot detect this omission.
    assert compare_runs(baseline, candidate, policy)["decision"] == "pass"
    policy["expected_case_set_digest"] = pin
    with pytest.raises(PipelineError, match="planned case set"):
        compare_runs(baseline, candidate, policy)


def test_complete_planned_case_set_accepts_all_records():
    baseline, candidate, policy = example_project("extraction")
    policy["expected_case_set_digest"] = digest(planned(baseline))
    assert compare_runs(baseline, candidate, policy)["decision"] == "pass"


def test_canonical_order_detachment_and_run_order_independence():
    baseline, candidate, policy = example_project("extraction")
    value = planned(baseline)
    value["cases"].reverse()
    normalized = canonical_case_set(value)
    pin = case_set_digest(value)
    assert normalized == planned(baseline)
    assert pin == digest(normalized)
    value["cases"][0]["metadata"]["changed"] = "yes"
    assert "changed" not in normalized["cases"][-1]["metadata"]
    candidate["records"].reverse()
    assert digest(candidate) != digest(baseline)
    policy["expected_case_set_digest"] = pin
    assert compare_runs(baseline, candidate, policy)["decision"] == "pass"


@pytest.mark.parametrize(
    "left,right", [(1, 1.0), (1, True), ("é", "e\u0301"), ([1, 2], [2, 1]), ("1", 1)]
)
def test_membership_preserves_json_types_text_and_nested_order(left, right):
    value = {
        "format": CASE_SET_FORMAT,
        "cases": [{"id": "one", "input": left, "expected": None, "metadata": {}}],
    }
    changed = deepcopy(value)
    changed["cases"][0]["input"] = right
    assert case_set_digest(value) != case_set_digest(changed)


def test_numeric_lexical_forms_follow_parsed_json_semantics():
    value = {
        "format": CASE_SET_FORMAT,
        "cases": [
            {"id": "one", "input": json.loads("1e0"), "expected": None, "metadata": {}}
        ],
    }
    changed = deepcopy(value)
    changed["cases"][0]["input"] = json.loads("1.0")
    assert case_set_digest(value) == case_set_digest(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.update(extra=True),
        lambda v: v.update(format="unknown"),
        lambda v: v.update(cases=[]),
        lambda v: v["cases"].append(deepcopy(v["cases"][0])),
        lambda v: v["cases"][0].update(output="answer"),
        lambda v: v["cases"][0].pop("expected"),
        lambda v: v["cases"][0].update(id=""),
        lambda v: v["cases"][0].update(id="a" * 129),
        lambda v: v["cases"][0].update(id="bad\nID"),
        lambda v: v["cases"][0].update(metadata={"cluster": 3}),
        lambda v: v["cases"][0].update(input=float("nan")),
    ],
)
def test_closed_case_contract_rejects_invalid_values(mutation):
    value = planned(example_project("extraction")[0])
    mutation(value)
    with pytest.raises(PipelineError):
        canonical_case_set(value)


def test_case_count_and_byte_limits(monkeypatch):
    value = {
        "format": CASE_SET_FORMAT,
        "cases": [
            {"id": str(i), "input": None, "expected": None, "metadata": {}}
            for i in range(10000)
        ],
    }
    assert len(canonical_case_set(value)["cases"]) == 10000
    value["cases"].append(
        {"id": "extra", "input": None, "expected": None, "metadata": {}}
    )
    with pytest.raises(PipelineError, match="too long"):
        canonical_case_set(value)
    value["cases"].pop()
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", 100)
    with pytest.raises(PipelineError, match="byte limit"):
        canonical_case_set(value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["records"].pop(),
        lambda r: r["records"].append(
            {**deepcopy(r["records"][0]), "id": "additional"}
        ),
        lambda r: r["records"][0].update(input="changed"),
        lambda r: r["records"][0].update(expected="changed"),
        lambda r: r["records"][0]["metadata"].update(cluster="changed"),
    ],
)
def test_shared_mutation_rejected_before_scoring(mutation, monkeypatch):
    baseline, candidate, policy = example_project("extraction")
    policy["expected_case_set_digest"] = case_set_digest(planned(baseline))
    mutation(baseline)
    mutation(candidate)

    def forbidden(*args, **kwargs):
        pytest.fail("metric arithmetic must not start for a changed planned set")

    monkeypatch.setattr(comparison, "score", forbidden)
    with pytest.raises(PipelineError, match="planned case set"):
        compare_runs(baseline, candidate, policy)


@pytest.mark.parametrize("side", [0, 1])
def test_each_run_must_match_planned_set(side):
    runs = list(example_project("extraction"))
    runs[2]["expected_case_set_digest"] = case_set_digest(planned(runs[0]))
    runs[side]["records"].pop()
    with pytest.raises(PipelineError, match="planned case set"):
        compare_runs(*runs)


def test_membership_excludes_execution_but_keeps_error_rows():
    baseline, candidate, policy = example_project("extraction")
    pin = case_set_digest(planned(baseline))
    candidate["artifact_digest"] = "sha256:" + "a" * 64
    row = candidate["records"][0]
    row.update(
        output="malformed",
        scores={"extra": 0},
        error="truncated",
        context={"model": "other"},
    )
    validate_run_case_set(candidate, pin)
    policy["expected_case_set_digest"] = pin
    assert (
        compare_runs(baseline, candidate, policy)["decision"] == "insufficient_evidence"
    )
    assert candidate["records"][0]["error"] == "truncated"


@pytest.mark.parametrize(
    "pin", [None, 1, "sha256:" + "A" * 64, "sha256:" + "a" * 64 + "\n"]
)
def test_helper_rejects_noncanonical_digest(pin):
    with pytest.raises(PipelineError, match="lowercase sha256"):
        validate_run_case_set(example_project("extraction")[0], pin)


def write_json(path, value):
    path.write_bytes(canonical_json_bytes(value))
    return path


def test_cli_case_set_roundtrip_and_no_replace(tmp_path):
    value = planned(example_project("extraction")[0])
    value["cases"].reverse()
    source = write_json(tmp_path / "planned.json", value)
    output = tmp_path / "canonical.json"
    runner = CliRunner()
    result = runner.invoke(app, ["case-set", str(source), "--output", str(output)])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["case_count"] == 40
    assert summary["expected_case_set_digest"] == digest(
        json.loads(output.read_bytes())
    )
    assert output.read_bytes() == canonical_json_bytes(canonical_case_set(value))
    again = runner.invoke(app, ["case-set", str(output)])
    assert again.exit_code == 0
    assert (
        json.loads(again.output)["expected_case_set_digest"]
        == summary["expected_case_set_digest"]
    )
    before = output.read_bytes()
    refused = runner.invoke(app, ["case-set", str(source), "--output", str(output)])
    assert refused.exit_code == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "content", [b'{"format":1,"format":2}', b"{}", b'{"input":NaN}']
)
def test_cli_case_set_rejects_invalid_json_without_publication(tmp_path, content):
    source = tmp_path / "source.json"
    source.write_bytes(content)
    output = tmp_path / "out.json"
    result = CliRunner().invoke(app, ["case-set", str(source), "--output", str(output)])
    assert result.exit_code == 2
    assert not output.exists()


def test_cli_case_set_rejects_symlink(tmp_path):
    source = write_json(
        tmp_path / "source.json", planned(example_project("extraction")[0])
    )
    link = tmp_path / "link.json"
    link.symlink_to(source)
    assert CliRunner().invoke(app, ["case-set", str(link)]).exit_code == 2


def project(tmp_path, baseline, candidate, policy):
    for name, value in (
        ("baseline", baseline),
        ("candidate", candidate),
        ("policy", policy),
    ):
        write_json(tmp_path / f"{name}.json", value)
    return write_json(
        tmp_path / "project.json",
        {
            "format": "invarlock/pipeline-project-v1",
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "candidate": {"path": "candidate.json", "adapter": "invarlock"},
            "policy": "policy.json",
        },
    )


@pytest.mark.parametrize("change", ["omission", "null", "bad", "newline", "override"])
def test_cli_rejects_membership_before_loading_private_key(
    tmp_path, monkeypatch, change
):
    baseline, candidate, policy = example_project("extraction")
    policy["expected_case_set_digest"] = case_set_digest(planned(baseline))
    extra = []
    if change == "omission":
        baseline["records"].pop()
        candidate["records"].pop()
    elif change == "override":
        replacement = deepcopy(candidate)
        replacement["records"].pop()
        extra = [
            "--candidate",
            str(write_json(tmp_path / "replacement.json", replacement)),
        ]
    else:
        policy["expected_case_set_digest"] = {
            "null": None,
            "bad": "invalid",
            "newline": policy["expected_case_set_digest"] + "\n",
        }[change]
    path = project(tmp_path, baseline, candidate, policy)

    def forbidden(*args):
        pytest.fail("a rejected planned set must not load the private key")

    monkeypatch.setattr(cli, "_private", forbidden)
    output = tmp_path / "result"
    result = CliRunner().invoke(
        app,
        [
            "compare",
            str(path),
            "--output",
            str(output),
            "--signing-key",
            str(tmp_path / "private.pem"),
            *extra,
        ],
    )
    assert result.exit_code == 2, result.output
    assert not output.exists()


def test_cli_signed_case_pin_handoff(tmp_path):
    baseline, candidate, policy = example_project("extraction")
    policy["expected_case_set_digest"] = case_set_digest(planned(baseline))
    path = project(tmp_path, baseline, candidate, policy)
    runner = CliRunner()
    keys = tmp_path / "keys"
    assert runner.invoke(app, ["keygen", str(keys)]).exit_code == 0
    output = tmp_path / "result"
    result = runner.invoke(
        app,
        [
            "compare",
            str(path),
            "--output",
            str(output),
            "--signing-key",
            str(keys / "private.pem"),
        ],
    )
    assert result.exit_code == 0, result.output
    verified = runner.invoke(
        app,
        [
            "verify",
            str(output / "evidence.json"),
            "--public-key",
            str(keys / "public.pem"),
            "--policy",
            str(tmp_path / "policy.json"),
            "--expected-baseline",
            digest(baseline),
            "--expected-candidate",
            digest(candidate),
        ],
    )
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.output)["authenticated"] is True


def test_resigned_mutual_omission_cannot_override_recipient_case_pin():
    baseline, candidate, policy = example_project("extraction")
    policy["expected_case_set_digest"] = case_set_digest(planned(baseline))
    key = Ed25519PrivateKey.generate()
    original = create_evidence(baseline, candidate, policy, key)
    forged = deepcopy(original)
    forged["baseline"]["records"].pop()
    forged["candidate"]["records"].pop()
    # An authorized signer fabricates a internally consistent truncated report.
    # Updated test-only full-run anchors deliberately reach the membership guard.
    unpinned = {k: v for k, v in policy.items() if k != "expected_case_set_digest"}
    forged["comparison"] = compare_runs(
        forged["baseline"], forged["candidate"], unpinned
    )
    payload = {k: v for k, v in forged.items() if k != "signature"}
    forged["signature"]["value"] = base64.b64encode(
        key.sign(DOMAIN + canonical_json_bytes(payload))
    ).decode()
    with pytest.raises(PipelineError, match="planned case set"):
        verify_evidence(
            forged,
            public_key=key.public_key(),
            expected_baseline=digest(forged["baseline"]),
            expected_candidate=digest(forged["candidate"]),
            policy=policy,
        )
    # Removing the policy pin cannot bypass the recipient's original policy.
    changed = create_evidence(forged["baseline"], forged["candidate"], unpinned, key)
    with pytest.raises(PipelineError, match="recipient.*policy"):
        verify_evidence(
            changed,
            public_key=key.public_key(),
            expected_baseline=digest(changed["baseline"]),
            expected_candidate=digest(changed["candidate"]),
            policy=policy,
        )


def test_legacy_unpinned_comparison_and_signature_identity():
    baseline, candidate, policy = example_project("extraction")
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    assert (
        digest(compare_runs(baseline, candidate, policy))
        == "sha256:c518c4fa06a537c60954142b1b06a74e2c5e8b6e2be43c4bc380c4889d05013b"
    )
    evidence = create_evidence(baseline, candidate, policy, key)
    assert (
        digest(evidence)
        == "sha256:a2b9d6f14b9c5cd7610f316fb88354b749d10fbc6a461b8b3c7bfacf854474dd"
    )
    assert (
        verify_evidence(
            evidence,
            public_key=key.public_key(),
            expected_baseline=digest(baseline),
            expected_candidate=digest(candidate),
            policy=policy,
        )["decision"]
        == "pass"
    )


def test_source_and_packaged_case_contracts_match():
    root = Path(__file__).resolve().parents[2]
    for name in ("case_set", "policy", "evidence"):
        filename = f"pipeline_{name}.schema.json"
        assert (root / "contracts" / filename).read_bytes() == (
            root / "src/invarlock/_data/contracts" / filename
        ).read_bytes()
