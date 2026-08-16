from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from scripts.release import release_reference_journey as journey


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def completed_reference(
    repo_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, object], Path, dict[str, object], dict[str, str]]:
    workspace = tmp_path_factory.mktemp("release-reference") / "clean-consumer"

    summary = journey.run_release_reference_journey(
        repo_root=repo_root,
        command=(sys.executable, "-m", "invarlock.cli.app"),
        workspace=workspace,
        allow_checkout_source=True,
    )
    output = workspace.parent.resolve() / workspace.name
    receipt = json.loads((output / "verification.receipt.json").read_bytes())
    verifier = receipt["statement"]["verifier"]
    verification = {
        "trust_profile_digest": verifier["trust_profile_digest"],
        "verifier_fingerprint": verifier["signing_key_fingerprint"],
    }
    return summary, output, journey._load_reference(repo_root), verification


def test_retained_reference_replays_through_current_cli(
    repo_root: Path,
    completed_reference: tuple[
        dict[str, object], Path, dict[str, object], dict[str, str]
    ],
) -> None:
    summary, output, _config, _verification = completed_reference

    assert summary == {
        "comparison_id": "comparison-f22a4c52f0172fb7c17e33a37f46a75d",
        "format": "invarlock/release-reference-result-v1",
        "ok": True,
        "pack_manifest_digest": (
            "sha256:355ed56ce95a79faa384ce2f2de6d4e3804fba81c243b82d37cac397fd4a454d"
        ),
        "policy_verdict": "pass",
        "receipt_sha256": summary["receipt_sha256"],
        "reference_id": "qwen3.8-27b-bf16-to-q5-k-m-gguf",
        "report_sha256": summary["report_sha256"],
        "verification_scope": "paired_comparison",
    }
    assert isinstance(summary["report_sha256"], str)
    assert journey._DIGEST.fullmatch(summary["report_sha256"])
    assert isinstance(summary["receipt_sha256"], str)
    assert journey._DIGEST.fullmatch(summary["receipt_sha256"])
    assert (output / "verification.receipt.json").is_file()
    assert not (output / "trust" / "verifier.pem").exists()
    assert (output / "report-a.html").read_bytes() == (
        output / "report-b.html"
    ).read_bytes()
    assert str(repo_root) not in json.dumps(summary, sort_keys=True)


def test_every_public_pack_replays_through_current_cli(
    repo_root: Path, tmp_path: Path
) -> None:
    summary = journey.run_public_evidence_compatibility(
        repo_root=repo_root,
        command=(sys.executable, "-m", "invarlock.cli.app"),
        workspace=tmp_path / "all-public-evidence",
        allow_checkout_source=True,
    )

    expected_ids = {
        path.name
        for path in (repo_root / "public_evidence" / "evidence").iterdir()
        if path.is_dir()
    }
    references = summary["references"]
    assert isinstance(references, list)
    assert summary == {
        "format": "invarlock/release-public-evidence-compatibility-result-v1",
        "ok": True,
        "pack_count": len(expected_ids),
        "references": references,
    }
    assert {item["reference_id"] for item in references} == expected_ids
    assert all(item["ok"] is True for item in references)
    assert all(journey._DIGEST.fullmatch(item["receipt_sha256"]) for item in references)
    assert all(journey._DIGEST.fullmatch(item["report_sha256"]) for item in references)
    assert not list((tmp_path / "all-public-evidence").rglob("verifier.pem"))


def test_every_evaluator_transaction_replays_through_current_cli(
    repo_root: Path, tmp_path: Path
) -> None:
    summary = journey.run_evaluator_qualification_compatibility(
        repo_root=repo_root,
        command=(sys.executable, "-m", "invarlock.cli.app"),
        workspace=tmp_path / "all-evaluator-transactions",
        allow_checkout_source=True,
    )

    transactions_root = (
        repo_root / "examples/evaluator-qualification/signed-transactions"
    )
    expected_ids = {path.name for path in transactions_root.iterdir() if path.is_dir()}
    references = summary["references"]
    assert isinstance(references, list)
    assert summary == {
        "format": ("invarlock/release-evaluator-qualification-compatibility-result-v1"),
        "ok": True,
        "pack_count": len(expected_ids),
        "references": references,
    }
    assert {item["reference_id"] for item in references} == expected_ids
    assert {item["policy_verdict"] for item in references} == {"fail", "pass"}
    assert all(item["ok"] is True for item in references)
    assert not list((tmp_path / "all-evaluator-transactions").rglob("verifier.pem"))


def test_all_retained_evidence_combines_both_closed_sets(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    def public(**kwargs: object) -> dict[str, object]:
        calls.append(("public", kwargs["workspace"]))  # type: ignore[arg-type]
        return {"ok": True, "pack_count": 7}

    def evaluator(**kwargs: object) -> dict[str, object]:
        calls.append(("evaluator", kwargs["workspace"]))  # type: ignore[arg-type]
        return {"ok": True, "pack_count": 4}

    monkeypatch.setattr(journey, "run_public_evidence_compatibility", public)
    monkeypatch.setattr(journey, "run_evaluator_qualification_compatibility", evaluator)
    workspace = tmp_path / "all-retained-evidence"
    summary = journey.run_retained_evidence_compatibility(
        repo_root=repo_root,
        command=(sys.executable,),
        workspace=workspace,
        allow_checkout_source=True,
    )

    assert summary == {
        "evaluator_qualification": {"ok": True, "pack_count": 4},
        "format": "invarlock/release-retained-evidence-compatibility-result-v1",
        "ok": True,
        "pack_count": 11,
        "public_evidence": {"ok": True, "pack_count": 7},
    }
    assert [name for name, _workspace in calls] == ["public", "evaluator"]


def _valid_verification(config: dict[str, object]) -> dict[str, object]:
    evidence = config["evidence"]
    expected = config["expected"]
    anchors = config["anchors"]
    policy = config["policy"]
    assert isinstance(evidence, dict)
    assert isinstance(expected, dict)
    assert isinstance(anchors, dict)
    assert isinstance(policy, dict)
    digest = "sha256:" + "1" * 64
    return {
        "assurance_status": "verified",
        "authenticity": "pinned",
        "comparison_id": evidence["comparison_id"],
        "errors": [],
        "format_version": journey.VERIFY_FORMAT,
        "integrity_ok": True,
        "ok": expected["policy_verdict"] == "pass",
        "pack_format": "invarlock/evidence-pack-v1",
        "pack_manifest_digest": evidence["pack_manifest_digest"],
        "policy_verdict": expected["policy_verdict"],
        "reports_verified": True,
        "request_digest": anchors["request_digest"],
        "verification_scope": expected["verification_scope"],
        "verifier_identity": "release-reference-verifier",
        "warnings": [],
        "anchors": journey._expected_result_anchors(anchors, policy["digest"]),
        "trust_profile_digest": digest,
        "verifier_fingerprint": digest,
    }


def test_reference_configuration_pins_manifest_and_external_policy(
    repo_root: Path,
) -> None:
    config = journey._load_reference(repo_root)
    evidence = repo_root / config["evidence"]["path"]
    policy = repo_root / config["policy"]["path"]

    assert (
        "sha256:"
        + hashlib.sha256((evidence / "manifest.json").read_bytes()).hexdigest()
        == config["evidence"]["pack_manifest_digest"]
    )
    assert (
        "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest()
        == config["policy"]["digest"]
    )
    assert policy.read_bytes() == journey._canonical_json_bytes(
        json.loads(policy.read_bytes())
    )


def test_reference_workspace_must_be_new_and_outside_checkout(
    repo_root: Path,
) -> None:
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="outside"):
        journey.run_release_reference_journey(
            repo_root=repo_root,
            command=(sys.executable, "-m", "invarlock.cli.app"),
            workspace=repo_root / "release-reference-must-not-exist",
            allow_checkout_source=True,
        )


@pytest.mark.parametrize(
    "raw",
    (b'{"ok":true,"ok":false}', b'{"value":NaN}', b"[]"),
)
def test_candidate_json_is_strict_and_object_shaped(raw: bytes) -> None:
    with pytest.raises(journey.ReleaseReferenceJourneyError):
        journey._strict_json_object(raw, label="candidate")


def test_json_helpers_reject_empty_oversize_and_noncanonical_values() -> None:
    for raw in (b"", b"{}" + b" " * journey.MAX_JSON_BYTES):
        with pytest.raises(journey.ReleaseReferenceJourneyError, match="byte length"):
            journey._strict_json_object(raw, label="candidate")
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="not canonical"):
        journey._canonical_json_bytes({"invalid": object()})


@pytest.mark.parametrize("value", (None, "", "../escape", "/absolute", "bad\\path"))
def test_relative_paths_reject_traversal_and_platform_ambiguity(value: object) -> None:
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="safe relative"):
        journey._relative_parts(value, label="path")


def test_checkout_path_resolution_rejects_missing_symlink_and_wrong_kind(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular"
    regular.write_text("data", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()
    link = tmp_path / "link"
    link.symlink_to(regular)

    for value, directory_expected, message in (
        ("missing", False, "unavailable"),
        ("link", False, "symlinks"),
        ("regular", True, "directory"),
        ("directory", False, "regular file"),
    ):
        with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
            journey._resolve_checkout_path(
                tmp_path,
                value,
                label="input",
                directory=directory_expected,
            )


def test_regular_file_reader_rejects_missing_oversize_and_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="unavailable"):
        journey._read_regular_file(missing, label="input", maximum=1)

    source = tmp_path / "source"
    source.write_bytes(b"ab")
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="bounded"):
        journey._read_regular_file(source, label="input", maximum=1)

    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: b"a" if self == source else original(self),
    )
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="changed"):
        journey._read_regular_file(source, label="input", maximum=2)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("fields", "fields"),
        ("format", "format"),
        ("anchors", "anchors"),
        ("anchor_digest", "sha256"),
        ("evidence", "evidence"),
        ("comparison_id", "comparison_id"),
        ("expected", "expectations"),
        ("verdict", "verdict"),
        ("scope", "scope"),
        ("policy", "policy"),
    ),
)
def test_reference_configuration_is_closed(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    config = copy.deepcopy(journey._load_reference(repo_root))
    if case == "fields":
        config["extra"] = True
    elif case == "format":
        config["format"] = "wrong"
    elif case == "anchors":
        config["anchors"] = {}
    elif case == "anchor_digest":
        config["anchors"]["request_digest"] = "wrong"
    elif case == "evidence":
        config["evidence"] = {}
    elif case == "comparison_id":
        config["evidence"]["comparison_id"] = " "
    elif case == "expected":
        config["expected"] = {}
    elif case == "verdict":
        config["expected"]["policy_verdict"] = "other"
    elif case == "scope":
        config["expected"]["verification_scope"] = "other"
    elif case == "policy":
        config["policy"] = {}
    monkeypatch.setattr(
        journey,
        "_strict_json_object",
        lambda *_args, **_kwargs: config,
    )

    with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
        journey._load_reference(repo_root)


def test_workspace_rejects_existing_missing_parent_and_creation_failure(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="must be new"):
        journey._prepare_workspace(repo_root, existing)
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="parent"):
        journey._prepare_workspace(repo_root, tmp_path / "missing" / "workspace")

    target = tmp_path / "cannot-create"
    original = Path.mkdir

    def reject(self: Path, *args: object, **kwargs: object) -> None:
        if self == target:
            raise OSError("denied")
        original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", reject)
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="could not"):
        journey._prepare_workspace(repo_root, target)


@pytest.mark.parametrize(
    "outcome",
    (OSError("launch"), subprocess.TimeoutExpired(["candidate"], 1)),
)
def test_candidate_launch_failures_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: Exception
) -> None:
    monkeypatch.setattr(
        journey.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(outcome),
    )
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="command failed"):
        journey._run_cli((sys.executable,), (), cwd=tmp_path, environment={})


def test_candidate_nonzero_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        journey.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 2, "{}", "private"),
    )
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="unexpected status"):
        journey._run_cli((sys.executable,), (), cwd=tmp_path, environment={})


def test_expected_policy_rejection_status_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        journey.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 7, "{}", ""),
    )

    assert (
        journey._run_cli(
            (sys.executable,),
            (),
            cwd=tmp_path,
            environment={},
            expected_status=7,
        )
        == {}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("diagnostic", "diagnostics"),
        ("anchors", "anchors changed"),
        ("fingerprint", "sha256"),
    ),
)
def test_verification_result_rejects_diagnostics_anchor_or_identity_drift(
    repo_root: Path, mutation: str, message: str
) -> None:
    config = journey._load_reference(repo_root)
    result = _valid_verification(config)
    if mutation == "diagnostic":
        result["warnings"] = ["changed"]
    elif mutation == "anchors":
        result["anchors"] = {}
    else:
        result["verifier_fingerprint"] = "wrong"
    with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
        journey._validate_verification_result(result, config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("noncanonical", "not canonical"),
        ("shape", "receipt is invalid"),
        ("statement", "statement changed"),
        ("verifier", "verifier changed"),
        ("signature", "signature is invalid"),
        ("public_key", "public key is invalid"),
        ("signed_value", "did not verify"),
    ),
)
def test_independent_receipt_validation_rejects_mutation(
    completed_reference: tuple[
        dict[str, object], Path, dict[str, object], dict[str, str]
    ],
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    _summary, output, config, verification = completed_reference
    receipt = json.loads((output / "verification.receipt.json").read_bytes())
    if mutation == "shape":
        receipt["statement"] = []
    elif mutation == "statement":
        receipt["statement"]["format"] = "wrong"
    elif mutation == "verifier":
        receipt["statement"]["verifier"] = {}
    elif mutation == "signature":
        receipt["signature"]["algorithm"] = "wrong"
    elif mutation == "public_key":
        receipt["signature"]["public_key"] = {}
    elif mutation == "signed_value":
        receipt["signature"]["value"] = "AAAA"
    path = tmp_path / f"{mutation}.json"
    raw = journey._canonical_json_bytes(receipt)
    if mutation == "noncanonical":
        raw += b" "
    path.write_bytes(raw)

    with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
        journey._validate_receipt(path, verification, config)


def test_receipt_public_key_must_match_declared_verifier_fingerprint(
    completed_reference: tuple[
        dict[str, object], Path, dict[str, object], dict[str, str]
    ],
    tmp_path: Path,
) -> None:
    _summary, output, config, verification = completed_reference
    receipt = json.loads((output / "verification.receipt.json").read_bytes())
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    statement = receipt["statement"]
    receipt["signature"]["public_key"]["value"] = public_key.decode("ascii")
    receipt["signature"]["value"] = base64.b64encode(
        private_key.sign(journey._canonical_json_bytes(statement))
    ).decode("ascii")
    path = tmp_path / "different-verifier-key.json"
    path.write_bytes(journey._canonical_json_bytes(receipt))

    with pytest.raises(journey.ReleaseReferenceJourneyError, match="fingerprint"):
        journey._validate_receipt(path, verification, config)


def test_report_result_is_closed() -> None:
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="inconsistent"):
        journey._validate_report_result(
            {"ok": True},
            html_path=Path("report.html"),
            manifest_digest="sha256:" + "0" * 64,
        )


def test_candidate_result_must_match_every_pinned_field(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(journey, "_run_cli", lambda *_args, **_kwargs: {"ok": True})
    workspace = tmp_path / "bad-candidate"

    with pytest.raises(
        journey.ReleaseReferenceJourneyError,
        match="verification result is inconsistent",
    ):
        journey.run_release_reference_journey(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=workspace,
            allow_checkout_source=True,
        )
    assert not (
        workspace.parent.resolve() / workspace.name / "trust/verifier.pem"
    ).exists()


@pytest.mark.parametrize("pin", ("manifest", "policy"))
def test_reference_source_bytes_must_match_their_pins(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pin: str,
) -> None:
    config = copy.deepcopy(journey._load_reference(repo_root))
    if pin == "manifest":
        config["evidence"]["pack_manifest_digest"] = "sha256:" + "0" * 64
    else:
        config["policy"]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(journey, "_load_reference", lambda _root, **_kwargs: config)

    with pytest.raises(journey.ReleaseReferenceJourneyError, match="pin"):
        journey.run_release_reference_journey(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=tmp_path / pin,
            allow_checkout_source=True,
        )


def test_reference_invocation_rejects_missing_or_invalid_executable(
    repo_root: Path, tmp_path: Path
) -> None:
    for root, command, message in (
        (tmp_path / "missing", (sys.executable,), "checkout"),
        (repo_root, (), "invocation"),
        (repo_root, (str(tmp_path / "missing-command"),), "unavailable"),
    ):
        with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
            journey.run_release_reference_journey(
                repo_root=root,
                command=command,
                workspace=tmp_path / (message + "-workspace"),
                allow_checkout_source=True,
            )

    non_executable = tmp_path / "not-executable"
    non_executable.write_text("data", encoding="utf-8")
    non_executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(journey.ReleaseReferenceJourneyError, match="invalid"):
        journey.run_release_reference_journey(
            repo_root=repo_root,
            command=(str(non_executable),),
            workspace=tmp_path / "non-executable-workspace",
            allow_checkout_source=True,
        )


def test_repeated_report_rendering_must_be_byte_identical(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = 0

    def command(
        _command: object,
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal reports
        if arguments[0] == "report":
            path = Path(arguments[arguments.index("--html") + 1])
            path.write_text(f"report-{reports}", encoding="utf-8")
            reports += 1
        return {}

    monkeypatch.setattr(journey, "_run_cli", command)
    monkeypatch.setattr(journey, "_validate_verification_result", lambda *_: None)
    monkeypatch.setattr(journey, "_validate_receipt", lambda *_: None)
    monkeypatch.setattr(
        journey, "_validate_report_result", lambda *_args, **_kwargs: None
    )

    with pytest.raises(journey.ReleaseReferenceJourneyError, match="deterministic"):
        journey.run_release_reference_journey(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=tmp_path / "nondeterministic",
            allow_checkout_source=True,
        )


def test_main_supports_source_candidate_and_error_paths(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(journey, "run_release_reference_journey", run)
    assert journey.main(["--repo-root", str(repo_root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert calls[-1]["allow_checkout_source"] is True

    workspace = tmp_path / "explicit-workspace"
    assert (
        journey.main(
            [
                "--repo-root",
                str(repo_root),
                "--invarlock-cli",
                sys.executable,
                "--workspace",
                str(workspace),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "Release reference journey passed.\n"
    assert calls[-1]["allow_checkout_source"] is False
    assert calls[-1]["workspace"] == workspace

    assert journey.main(["--repo-root", str(tmp_path / "missing")]) == 1
    assert "release reference journey rejected" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "attribute"),
    (
        ("--all-public-evidence", "run_public_evidence_compatibility"),
        (
            "--all-evaluator-qualification",
            "run_evaluator_qualification_compatibility",
        ),
        ("--all-retained-evidence", "run_retained_evidence_compatibility"),
    ),
)
def test_main_dispatches_retained_evidence_selection(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    attribute: str,
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(journey, attribute, run)
    assert journey.main(["--repo-root", str(repo_root), option, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert calls[-1]["allow_checkout_source"] is True


def test_public_evidence_compatibility_rejects_unsafe_entries(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    public_root = repo_root / "public_evidence" / "evidence"
    public_root.mkdir(parents=True)
    (public_root / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(journey.ReleaseReferenceJourneyError, match="unsafe"):
        journey.run_public_evidence_compatibility(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=tmp_path / "unsafe-workspace",
            allow_checkout_source=True,
        )


def test_evaluator_qualification_compatibility_rejects_unsafe_entries(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    transactions_root = (
        repo_root / "examples/evaluator-qualification/signed-transactions"
    )
    transactions_root.mkdir(parents=True)
    (transactions_root / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(journey.ReleaseReferenceJourneyError, match="unsafe"):
        journey.run_evaluator_qualification_compatibility(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=tmp_path / "unsafe-evaluator-workspace",
            allow_checkout_source=True,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate", "duplicated"),
        ("path", "path is inconsistent"),
        ("coverage", "do not cover"),
    ),
)
def test_public_evidence_compatibility_fails_closed_on_configuration_drift(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    config = copy.deepcopy(journey._load_reference(repo_root))
    if case == "duplicate":
        configs = (journey.REFERENCE_CONFIG, journey.REFERENCE_CONFIG)
    else:
        configs = (journey.REFERENCE_CONFIG,)
    if case == "path":
        config["evidence"]["path"] = "public_evidence/evidence/other/evidence"

    monkeypatch.setattr(journey, "PUBLIC_EVIDENCE_REFERENCE_CONFIGS", configs)
    monkeypatch.setattr(
        journey,
        "_load_reference",
        lambda _root, **_kwargs: copy.deepcopy(config),
    )
    monkeypatch.setattr(
        journey,
        "run_release_reference_journey",
        lambda **_kwargs: {"reference_id": config["evidence"]["reference_id"]},
    )

    with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
        journey.run_public_evidence_compatibility(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=tmp_path / f"{case}-workspace",
            allow_checkout_source=True,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate", "duplicated"),
        ("path", "path is inconsistent"),
        ("coverage", "do not cover"),
    ),
)
def test_evaluator_qualification_fails_closed_on_configuration_drift(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    reference = journey.EVALUATOR_QUALIFICATION_REFERENCE_CONFIGS[0]
    config = copy.deepcopy(
        journey._load_reference(repo_root, reference_config=reference)
    )
    if case == "duplicate":
        configs = (reference, reference)
    else:
        configs = (reference,)
    if case == "path":
        config["evidence"]["path"] = (
            "examples/evaluator-qualification/signed-transactions/other/evidence"
        )

    monkeypatch.setattr(journey, "EVALUATOR_QUALIFICATION_REFERENCE_CONFIGS", configs)
    monkeypatch.setattr(
        journey,
        "_load_reference",
        lambda _root, **_kwargs: copy.deepcopy(config),
    )
    monkeypatch.setattr(
        journey,
        "run_release_reference_journey",
        lambda **_kwargs: {"reference_id": config["evidence"]["reference_id"]},
    )

    with pytest.raises(journey.ReleaseReferenceJourneyError, match=message):
        journey.run_evaluator_qualification_compatibility(
            repo_root=repo_root,
            command=(sys.executable,),
            workspace=tmp_path / f"{case}-evaluator-workspace",
            allow_checkout_source=True,
        )


def test_reference_environment_removes_runtime_bypasses(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/untrusted")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "1")

    installed = journey._sanitized_environment(repo_root, allow_checkout_source=False)
    source = journey._sanitized_environment(repo_root, allow_checkout_source=True)

    assert "PYTHONHOME" not in installed
    assert "PYTHONPATH" not in installed
    assert "INVARLOCK_ALLOW_NETWORK" not in installed
    assert installed["PYTHONNOUSERSITE"] == "1"
    assert installed["PYTHONSAFEPATH"] == "1"
    assert source["PYTHONPATH"] == os.fspath(repo_root / "src")
