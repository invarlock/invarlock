from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import runtime_qualification

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "runtime_qualification.py"
PACK_DIGEST = "sha256:" + "a" * 64
CANARY_PACK_DIGEST = "sha256:" + "b" * 64
TRUST_DIGEST = "sha256:" + "c" * 64
RUNTIME_DIGEST = "sha256:" + "d" * 64
SOURCE_COMMIT = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
_SOURCE_BUNDLE_CACHE: bytes | None = None


def _source_bundle_bytes() -> bytes:
    global _SOURCE_BUNDLE_CACHE
    if _SOURCE_BUNDLE_CACHE is not None:
        return _SOURCE_BUNDLE_CACHE
    _SOURCE_BUNDLE_CACHE = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", SOURCE_COMMIT],
        check=True,
        capture_output=True,
    ).stdout
    return _SOURCE_BUNDLE_CACHE


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "qualification-python"
    log = tmp_path / "qualification-log.jsonl"
    executable.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

raw_arguments = sys.argv[1:]
if raw_arguments[:3] == ["-I", "-S", "-c"]:
    mode, target, *remaining = raw_arguments[4:]
    if mode == "module":
        arguments = ["-m", target, *remaining]
    elif mode == "path":
        arguments = [target, *remaining]
    elif mode == "code":
        arguments = ["-c", target, *remaining]
    else:
        raise SystemExit("unsupported fake bootstrap mode")
else:
    arguments = raw_arguments
root = Path(__file__).parent
control = json.loads(root.joinpath("qualification-control.json").read_text(encoding="utf-8"))
stage = control.get("failure_stage")
if (
    stage == "container_engine_mutation"
    and arguments[:3] == ["-m", "invarlock", "evaluate"]
    and "--preflight" in arguments
):
    engine = Path(shutil.which("docker")).resolve()
    engine.write_text("mutated container engine\\n", encoding="utf-8")
with root.joinpath("qualification-log.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{
        "argv0": sys.argv[0],
        "arguments": arguments,
        "cwd": os.getcwd(),
        "engine_path": shutil.which("docker"),
        "engine_resolved_path": (
            str(Path(shutil.which("docker")).resolve())
            if shutil.which("docker") else None
        ),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }}) + "\\n")

def fail(name: str) -> None:
    if stage == name:
        if name == "trust_precheck":
            print("ModuleNotFoundError: No module named 'invarlock'", file=sys.stderr)
        else:
            print(json.dumps({{"errors": [name + " sentinel"], "ok": False}}))
        raise SystemExit(3)

if arguments[:1] == ["-c"]:
    print(json.dumps({{
        "distributions": json.loads(os.environ["INVARLOCK_CANDIDATE_DISTRIBUTIONS"]),
        "format_version": "invarlock/qualification-candidate-probe-v1",
        "ok": True,
        "providers": [],
    }}))
elif arguments and arguments[0].endswith("qualification_precheck.py"):
    fail("trust_precheck")
    print(json.dumps({{
        "artifact_digests": {{
            "baseline": "sha256:" + "3" * 64,
            "subject": "sha256:" + "4" * 64,
        }},
        "evidence_signer_fingerprint": "sha256:" + "5" * 64,
        "request_digest": "sha256:" + "7" * 64,
        "format_version": "invarlock/qualification-precheck-v1",
        "ok": True,
        "policy_digest": "sha256:" + "1" * 64,
        "receipt": arguments[arguments.index("--receipt") + 1],
        "runtime_digests": {{
            "baseline": {RUNTIME_DIGEST!r},
            "subject": {RUNTIME_DIGEST!r},
        }},
        "schedule_digest": "sha256:" + "2" * 64,
        "trust_profile_digest": control.get("precheck_trust", {TRUST_DIGEST!r}),
        "verifier_fingerprint": "sha256:" + "6" * 64,
        "verifier_identity": "qualification-verifier",
    }}))
elif arguments and arguments[0].endswith("qualification_receipt_check.py"):
    canary = "--expected-runtime-image-digest" in arguments
    fail("canary_prerequisite" if canary else "receipt_verification")
    receipt = Path(arguments[arguments.index("--receipt") + 1])
    result = {{
        "format_version": "invarlock/qualification-receipt-check-v1",
        "ok": True,
        "pack_manifest_digest": {CANARY_PACK_DIGEST!r} if canary else {PACK_DIGEST!r},
        "receipt_sha256": "sha256:" + hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "verifier_fingerprint": "sha256:" + "6" * 64,
    }}
    if canary:
        result["runtime_image_digest"] = control.get(
            "canary_runtime_digest"
        ) or arguments[arguments.index("--expected-runtime-image-digest") + 1]
        result["compatibility"] = control.get("canary_compatibility") or {{
            "acceptance": {{"kind": "builtin_metric", "metric": "exact_match"}},
            "device_classes": {{"baseline": "cuda", "subject": "cuda"}},
            "providers": {{"baseline": "hf_transformers", "subject": "hf_transformers"}},
            "task": "text_causal",
        }}
    print(json.dumps(result))
elif arguments[:3] == ["-m", "invarlock", "evaluate"]:
    if "--preflight" in arguments:
        fail("preflight")
        print(json.dumps({{
            "format_version": "invarlock/evaluation-preflight-v2",
            "ok": True,
            "output": control.get("preflight_output") or control["evidence"],
            "artifact_digests": {{
                "baseline": "sha256:" + "3" * 64,
                "subject": "sha256:" + "4" * 64,
            }},
            "evidence_signer_fingerprint": "sha256:" + "5" * 64,
            "request_digest": "sha256:" + "7" * 64,
            "policy_digest": "sha256:" + "1" * 64,
            "runtime_image_digests": {{
                "baseline": {RUNTIME_DIGEST!r},
                "subject": {RUNTIME_DIGEST!r},
            }},
            "schedule_digest": "sha256:" + "2" * 64,
        }}))
    else:
        fail("evaluation")
        Path(control["evidence"]).mkdir()
        if control.get("binding_mutation") == "request_changed":
            captured_request = Path(arguments[3])
            captured_request.unlink()
            captured_request.write_text("changed request\\n", encoding="utf-8")
        published_evidence = control["evidence"]
        if control.get("binding_mutation") == "evaluation_relative":
            published_evidence = Path(published_evidence).name
        elif control.get("binding_mutation") == "evaluation_destination":
            published_evidence = str(
                Path(published_evidence).with_name("other-evidence")
            )
        print(json.dumps({{
            "evidence": published_evidence,
            "format_version": "invarlock/evaluation-result-v1",
            "ok": True,
            "pack_manifest_digest": {PACK_DIGEST!r},
        }}))
elif arguments[:3] == ["-m", "invarlock", "verify"]:
    fail("verification")
    receipt = Path(arguments[arguments.index("--receipt") + 1])
    receipt.write_text("signed receipt\\n", encoding="utf-8")
    verification = {{
        "anchors": {{
            "artifact_digests": {{
                "baseline": "sha256:" + "3" * 64,
                "subject": "sha256:" + "4" * 64,
            }},
            "policy_digest": "sha256:" + "1" * 64,
            "runtime_digests": {{
                "baseline": {RUNTIME_DIGEST!r},
                "subject": {RUNTIME_DIGEST!r},
            }},
            "schedule_digest": "sha256:" + "2" * 64,
            "signer_fingerprint": "sha256:" + "5" * 64,
        }},
        "assurance_status": "verified",
        "authenticity": "pinned",
        "errors": [],
        "format_version": "invarlock/evidence-pack-verify-v1",
        "integrity_ok": True,
        "ok": True,
        "pack_manifest_digest": (
            "sha256:" + "9" * 64
            if control.get("binding_mutation") == "verification_pack"
            else {PACK_DIGEST!r}
        ),
        "policy_verdict": "pass",
        "request_digest": "sha256:" + "7" * 64,
        "reports_verified": True,
        "signer_fingerprint": "sha256:" + "5" * 64,
        "signed_receipt": receipt.name,
        "trust_profile_digest": control.get("verified_trust") or {TRUST_DIGEST!r},
        "verifier_fingerprint": "sha256:" + "6" * 64,
        "verifier_identity": "qualification-verifier",
        "verification_scope": "paired_comparison",
        "warnings": [],
    }}
    mutation = control.get("binding_mutation")
    if mutation == "artifact_digests":
        verification["anchors"]["artifact_digests"]["subject"] = "sha256:" + "9" * 64
    elif mutation == "runtime_digests":
        verification["anchors"]["runtime_digests"]["subject"] = "sha256:" + "9" * 64
    elif mutation in {{"policy_digest", "schedule_digest", "signer_fingerprint"}}:
        verification["anchors"][mutation] = "sha256:" + "9" * 64
    elif mutation in {{"trust_profile_digest", "verifier_fingerprint", "verifier_identity", "signed_receipt"}}:
        verification[mutation] = "changed"
    elif mutation == "request_digest":
        verification[mutation] = "sha256:" + "9" * 64
    elif mutation == "observed_signer_fingerprint":
        verification["signer_fingerprint"] = "sha256:" + "9" * 64
    elif mutation == "strict_errors":
        verification["errors"] = ["sentinel"]
    elif mutation == "strict_authenticity":
        verification["authenticity"] = "mismatch"
    print(json.dumps(verification))
elif arguments[:3] == ["-m", "invarlock", "report"]:
    fail("report")
    report = Path(arguments[arguments.index("--html") + 1])
    if control.get("binding_mutation") == "receipt_changed":
        Path(control["receipt"]).write_text("changed receipt\\n", encoding="utf-8")
    report.write_text("<html>qualified</html>\\n", encoding="utf-8")
    report_pack = (
        "sha256:" + "9" * 64
        if control.get("binding_mutation") == "report_pack"
        else {PACK_DIGEST!r}
    )
    print(json.dumps({{
        "format_version": "invarlock/evidence-report-v1",
        "html": (
            str(report.with_name("other-report.html"))
            if control.get("binding_mutation") == "report_destination"
            else str(report)
        ),
        "ok": True,
        "pack_manifest_digest": report_pack,
    }}))
else:
    print("unexpected qualification invocation", file=sys.stderr)
    raise SystemExit(4)
""",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    container_engine = tmp_path / "docker"
    container_engine.write_text(
        f"""#!{sys.executable}
import json
from pathlib import Path
control = json.loads(Path(__file__).with_name("qualification-control.json").read_text())
if control.get("failure_stage") == "runtime_source":
    raise SystemExit("runtime_source sentinel")
print(json.dumps([{{"Config": {{"Labels": {{
    "dev.invarlock.source-bundle-sha256": control["source_bundle_digest"],
    "org.opencontainers.image.revision": control["runtime_source_commit"],
}}}}}}]))
""",
        encoding="utf-8",
    )
    container_engine.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return executable, log


def _candidate_wheel(tmp_path: Path, source_bundle: Path) -> tuple[Path, Path]:
    wheel = tmp_path / "invarlock-0.12.1-py3-none-any.whl"
    with (
        tarfile.open(source_bundle, "r:*") as source,
        zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as destination,
    ):
        for member in source.getmembers():
            if not member.isfile() or not member.name.startswith("src/invarlock/"):
                continue
            extracted = source.extractfile(member)
            assert extracted is not None
            destination.writestr(member.name.removeprefix("src/"), extracted.read())
        destination.writestr(
            "invarlock-0.12.1.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: invarlock\nVersion: 0.12.1\n",
        )
    manifest = tmp_path / "candidate-wheels.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": "invarlock/qualification-candidate-wheels-v1",
                "wheels": [
                    {
                        "path": str(wheel),
                        "sha256": "sha256:"
                        + hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return wheel, manifest


def _inputs(tmp_path: Path) -> dict[str, Path]:
    request = tmp_path / "request.yaml"
    request.write_text(
        "format_version: invarlock/evaluation-request-v1\n", encoding="utf-8"
    )
    signing_key = tmp_path / "evidence-signer.pem"
    signing_key.write_text("private fixture\n", encoding="utf-8")
    trust_profile = tmp_path / "trusted-inputs.json"
    trust_profile.write_text("{}\n", encoding="utf-8")
    source_bundle = tmp_path / "source.tar.gz"
    source_bundle.write_bytes(_source_bundle_bytes())
    _wheel, candidate_manifest = _candidate_wheel(tmp_path, source_bundle)
    canary_evidence = tmp_path / "canary-evidence"
    canary_evidence.mkdir()
    canary_receipt = tmp_path / "canary-receipt.json"
    canary_receipt.write_text("signed canary receipt\n", encoding="utf-8")
    canary_trust_profile = tmp_path / "canary-trust-inputs.json"
    canary_trust_profile.write_text("{}\n", encoding="utf-8")
    return {
        "request": request,
        "signing_key": signing_key,
        "trust_profile": trust_profile,
        "source_bundle": source_bundle,
        "candidate_manifest": candidate_manifest,
        "evidence": tmp_path / "evidence",
        "receipt": tmp_path / "verification-receipt.json",
        "canary_evidence": canary_evidence,
        "canary_receipt": canary_receipt,
        "canary_trust_profile": canary_trust_profile,
        "report": tmp_path / "report.html",
        "summary": tmp_path / "qualification-summary.json",
    }


def _write_control(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    failure_stage: str | None = None,
    verified_trust: str | None = None,
    binding_mutation: str | None = None,
    runtime_source_commit: str = SOURCE_COMMIT,
    preflight_output: str | None = None,
    canary_runtime_digest: str | None = None,
) -> None:
    tmp_path.joinpath("qualification-control.json").write_text(
        json.dumps(
            {
                "binding_mutation": binding_mutation,
                "canary_runtime_digest": canary_runtime_digest,
                "evidence": str(paths["evidence"].absolute()),
                "failure_stage": failure_stage,
                "preflight_output": preflight_output,
                "receipt": str(paths["receipt"]),
                "runtime_source_commit": runtime_source_commit,
                "source_bundle_digest": "sha256:"
                + hashlib.sha256(paths["source_bundle"].read_bytes()).hexdigest(),
                "verified_trust": verified_trust,
            }
        ),
        encoding="utf-8",
    )


def _command(
    tmp_path: Path,
    *,
    mode: str,
    python: Path,
    paths: dict[str, Path],
    report: bool = False,
    source_bundle_digest: str | None = None,
) -> list[str]:
    bundle_digest = source_bundle_digest or (
        "sha256:" + hashlib.sha256(paths["source_bundle"].read_bytes()).hexdigest()
    )
    command = [
        sys.executable,
        str(DRIVER),
        mode,
        "--python",
        str(python),
        "--request",
        str(paths["request"]),
        "--signing-key",
        str(paths["signing_key"]),
        "--runtime-image",
        RUNTIME_DIGEST,
        "--runtime-image-digest",
        RUNTIME_DIGEST,
        "--evidence",
        str(paths["evidence"]),
        "--trust-profile",
        str(paths["trust_profile"]),
        "--receipt",
        str(paths["receipt"]),
        "--source-commit",
        SOURCE_COMMIT,
        "--source-bundle",
        str(paths["source_bundle"]),
        "--source-bundle-sha256",
        bundle_digest,
        "--candidate-wheel-manifest",
        str(paths["candidate_manifest"]),
        "--runtime-device",
        "cuda:0",
        "--runtime-cpus",
        "4",
        "--runtime-memory-mib",
        "8192",
        "--runtime-user",
        "65532:65532",
    ]
    if mode != "canary":
        command.extend(
            (
                "--canary-evidence",
                str(paths["canary_evidence"]),
                "--canary-receipt",
                str(paths["canary_receipt"]),
                "--canary-trust-profile",
                str(paths["canary_trust_profile"]),
            )
        )
    if mode in {"run", "canary"}:
        command.extend(("--summary", str(paths["summary"])))
    if report:
        command.extend(("--report", str(paths["report"])))
    return command


def _execute(
    tmp_path: Path,
    *,
    mode: str = "run",
    failure_stage: str | None = None,
    verified_trust: str | None = None,
    binding_mutation: str | None = None,
    report: bool = False,
    runtime_source_commit: str = SOURCE_COMMIT,
    preflight_output: str = "bound",
    canary_runtime_digest: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Path], Path]:
    python, log = _fake_python(tmp_path)
    paths = _inputs(tmp_path)
    if preflight_output == "bound":
        controlled_preflight_output = None
    elif preflight_output == "relative":
        controlled_preflight_output = paths["evidence"].name
    elif preflight_output == "mismatched":
        controlled_preflight_output = str(tmp_path / "other-evidence")
    else:
        raise ValueError(f"unknown preflight output mode: {preflight_output}")
    _write_control(
        tmp_path,
        paths,
        failure_stage=failure_stage,
        verified_trust=verified_trust,
        binding_mutation=binding_mutation,
        runtime_source_commit=runtime_source_commit,
        preflight_output=controlled_preflight_output,
        canary_runtime_digest=canary_runtime_digest,
    )
    completed = subprocess.run(
        _command(tmp_path, mode=mode, python=python, paths=paths, report=report),
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, paths, log


def _qualified_invocations(log: Path) -> list[dict[str, object]]:
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    probe, *invocations = entries
    assert probe["arguments"][0] == "-c"
    return invocations


def test_qualification_python_preserves_a_venv_symlink(tmp_path: Path) -> None:
    executable, _log = _fake_python(tmp_path)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    linked_python = venv / "python"
    linked_python.symlink_to(executable)

    selected = runtime_qualification.qualification_python(linked_python)

    assert selected == str(linked_python.absolute())
    assert Path(selected).is_symlink()


def test_readiness_runs_preflight_and_repository_precheck_only(tmp_path: Path) -> None:
    completed, paths, log = _execute(tmp_path, mode="readiness")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["mode"] == "readiness"
    assert result["stage"] == "ready"
    assert result["verification_inputs"]["trust_profile_digest"] == TRUST_DIGEST
    assert result["request_sha256"] == (
        "sha256:" + hashlib.sha256(paths["request"].read_bytes()).hexdigest()
    )
    invocations = _qualified_invocations(log)
    assert len(invocations) == 3
    assert invocations[0]["arguments"][0].endswith("qualification_receipt_check.py")
    assert invocations[0]["arguments"][
        invocations[0]["arguments"].index(
            "--expected-runtime-image-digest"
        ) : invocations[0]["arguments"].index("--expected-runtime-image-digest") + 2
    ] == ["--expected-runtime-image-digest", RUNTIME_DIGEST]
    captured_request = Path(
        invocations[0]["arguments"][
            invocations[0]["arguments"].index("--expected-request") + 1
        ]
    )
    captured_root = Path(
        invocations[0]["arguments"][
            invocations[0]["arguments"].index("--expected-request-root") + 1
        ]
    )
    assert captured_request.name == paths["request"].name
    assert captured_root.resolve() == paths["request"].parent.resolve()
    assert captured_request.parent != captured_root
    assert invocations[1]["arguments"][:3] == ["-m", "invarlock", "evaluate"]
    assert "--preflight" in invocations[1]["arguments"]
    assert invocations[2]["arguments"][0].endswith("qualification_precheck.py")
    child_directories = {entry["cwd"] for entry in invocations}
    assert len(child_directories) == 1
    child_directory = Path(child_directories.pop())
    assert child_directory.name == "work"
    assert child_directory != ROOT
    assert all(entry["pythonpath"] is None for entry in invocations)
    authenticated_helpers = child_directory.parent / "source" / "scripts"
    assert (
        Path(invocations[0]["arguments"][0]).parent.resolve()
        == authenticated_helpers.resolve()
    )
    assert (
        Path(invocations[2]["arguments"][0]).parent.resolve()
        == authenticated_helpers.resolve()
    )


def test_canary_bootstrap_runs_without_a_prior_canary(tmp_path: Path) -> None:
    completed, _paths, log = _execute(tmp_path, mode="canary")

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mode"] == "canary"
    invocations = _qualified_invocations(log)
    assert not any(
        "--expected-runtime-image-digest" in entry["arguments"] for entry in invocations
    )
    assert invocations[0]["arguments"][:3] == ["-m", "invarlock", "evaluate"]


def test_canary_failure_stops_before_lane_preflight(tmp_path: Path) -> None:
    completed, paths, log = _execute(
        tmp_path,
        failure_stage="canary_prerequisite",
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "canary_prerequisite"
    invocations = _qualified_invocations(log)
    assert len(invocations) == 1
    assert invocations[0]["arguments"][0].endswith("qualification_receipt_check.py")
    assert not paths["evidence"].exists()


def test_canary_image_mismatch_stops_before_lane_preflight(tmp_path: Path) -> None:
    completed, paths, log = _execute(
        tmp_path,
        canary_runtime_digest="sha256:" + "9" * 64,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "canary_prerequisite"
    assert "exact qualification image" in failure["errors"][0]
    invocations = _qualified_invocations(log)
    assert len(invocations) == 1
    assert not paths["evidence"].exists()


def test_readiness_resolves_relative_output_against_original_request_root(
    tmp_path: Path,
) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        mode="readiness",
        preflight_output="relative",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["evidence"] == str(paths["evidence"])


def test_readiness_rejects_a_preflight_bound_to_another_destination(
    tmp_path: Path,
) -> None:
    completed, paths, log = _execute(
        tmp_path,
        mode="readiness",
        preflight_output="mismatched",
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "preflight_binding"
    assert "does not match --evidence" in failure["errors"][0]
    invocations = _qualified_invocations(log)
    assert len(invocations) == 2
    assert "--preflight" in invocations[-1]["arguments"]
    assert not paths["evidence"].exists()


def test_missing_precheck_dependency_is_a_structured_stage_failure(
    tmp_path: Path,
) -> None:
    completed, _paths, _log = _execute(
        tmp_path, mode="readiness", failure_stage="trust_precheck"
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["ok"] is False
    assert failure["stage"] == "trust_precheck"
    assert "ModuleNotFoundError" in failure["diagnostic"]["output"]


def test_readiness_accepts_a_request_in_a_read_only_directory(tmp_path: Path) -> None:
    python, log = _fake_python(tmp_path)
    paths = _inputs(tmp_path)
    request_root = tmp_path / "fixed-inputs"
    request_root.mkdir()
    request = request_root / "request.yaml"
    paths["request"].replace(request)
    paths["request"] = request
    request_root.chmod(0o500)
    try:
        _write_control(tmp_path, paths)
        completed = subprocess.run(
            _command(tmp_path, mode="readiness", python=python, paths=paths),
            cwd=ROOT,
            env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        request_root.chmod(0o700)

    assert completed.returncode == 0, completed.stderr
    assert log.exists()


def test_source_bundle_mismatch_fails_before_any_qualification_command(
    tmp_path: Path,
) -> None:
    python, log = _fake_python(tmp_path)
    paths = _inputs(tmp_path)
    original_digest = (
        "sha256:" + hashlib.sha256(paths["source_bundle"].read_bytes()).hexdigest()
    )
    paths["source_bundle"].write_bytes(b"substituted bundle\n")
    _write_control(tmp_path, paths)

    completed = subprocess.run(
        _command(
            tmp_path,
            mode="readiness",
            python=python,
            paths=paths,
            source_bundle_digest=original_digest,
        ),
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "configuration"
    assert "source bundle does not match" in failure["errors"][0]
    assert not log.exists()


def test_source_bundle_and_runtime_image_must_identify_the_declared_commit(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    bundle_digest = (
        "sha256:" + hashlib.sha256(paths["source_bundle"].read_bytes()).hexdigest()
    )
    with pytest.raises(
        runtime_qualification.QualificationError,
        match="source reference does not identify",
    ):
        runtime_qualification._authenticate_source_bundle(  # noqa: SLF001
            paths["source_bundle"],
            declared_digest=bundle_digest,
            source_commit="a" * 40,
            root=ROOT,
        )

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    completed, _paths, _log = _execute(
        runtime_root,
        mode="readiness",
        runtime_source_commit="a" * 40,
    )
    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "runtime_source"
    assert "does not match frozen source" in failure["errors"][0]


@pytest.mark.parametrize(
    "relative",
    ["scripts/runtime_qualification.py", "scripts/qualification_source.py"],
)
def test_live_qualification_code_must_match_authenticated_source(
    tmp_path: Path,
    relative: str,
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    for name in ("runtime_qualification.py", "qualification_source.py"):
        scripts.joinpath(name).write_text(
            f'IDENTITY = "{name}:frozen"\n', encoding="utf-8"
        )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "frozen"],
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bundle = tmp_path / "source.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            f"--output={bundle}",
            source_commit,
        ],
        check=True,
    )
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    repository.joinpath(relative).write_text(
        'IDENTITY = "live-mutation"\n', encoding="utf-8"
    )

    with pytest.raises(
        runtime_qualification.QualificationError,
        match=rf"live qualification file {re.escape(relative)} does not match",
    ):
        runtime_qualification._authenticate_source_bundle(  # noqa: SLF001
            bundle,
            declared_digest=digest,
            source_commit=source_commit,
            root=repository,
        )


def test_source_bundle_pax_comment_cannot_forge_git_commit_bytes(
    tmp_path: Path,
) -> None:
    committed = runtime_qualification._source_archive_files(  # noqa: SLF001
        _source_bundle_bytes(), source_commit=SOURCE_COMMIT
    )
    relative = sorted(committed)[0]
    committed[relative] += b"\nforged\n"
    source_bundle = tmp_path / "forged-source.tar"
    with tarfile.open(
        source_bundle,
        mode="w",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": SOURCE_COMMIT},
    ) as archive:
        for name, payload in sorted(committed.items()):
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    digest = "sha256:" + hashlib.sha256(source_bundle.read_bytes()).hexdigest()

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="do not match the selected Git commit",
    ):
        runtime_qualification._authenticate_source_bundle(  # noqa: SLF001
            source_bundle,
            declared_digest=digest,
            source_commit=SOURCE_COMMIT,
            root=ROOT,
        )


def test_git_replace_ref_cannot_change_authenticated_commit_bytes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    probe = repository / "src" / "invarlock" / "probe.py"
    probe.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    probe.write_text('VALUE = "clean"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "clean"],
        check=True,
    )
    clean = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    probe.write_text('VALUE = "malicious"\n', encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qam", "malicious"], check=True
    )
    malicious = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", clean, malicious], check=True
    )
    source_bundle = tmp_path / "clean.tar"
    subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            f"--output={source_bundle}",
            clean,
        ],
        check=True,
    )
    digest = "sha256:" + hashlib.sha256(source_bundle.read_bytes()).hexdigest()

    committed = runtime_qualification._authenticated_execution_sources(  # noqa: SLF001
        source_bundle,
        declared_digest=digest,
        source_commit=clean,
        root=repository,
    )

    assert committed["src/invarlock/probe.py"] == b'VALUE = "clean"\n'


def test_run_rejects_trust_substitution_before_summary(tmp_path: Path) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        verified_trust="sha256:" + "f" * 64,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "verification_binding"
    assert "trust-profile" in failure["errors"][0]
    assert not paths["summary"].exists()


@pytest.mark.parametrize(
    "binding_mutation",
    (
        "artifact_digests",
        "policy_digest",
        "runtime_digests",
        "schedule_digest",
        "signer_fingerprint",
        "request_digest",
        "observed_signer_fingerprint",
        "strict_errors",
        "strict_authenticity",
        "verifier_fingerprint",
        "verifier_identity",
        "signed_receipt",
    ),
)
def test_run_rejects_every_expanded_trust_binding_substitution(
    tmp_path: Path, binding_mutation: str
) -> None:
    completed, paths, _log = _execute(tmp_path, binding_mutation=binding_mutation)

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "verification_binding"
    assert not paths["summary"].exists()


def test_existing_summary_fails_before_any_qualification_command(
    tmp_path: Path,
) -> None:
    python, log = _fake_python(tmp_path)
    paths = _inputs(tmp_path)
    paths["summary"].write_text("existing\n", encoding="utf-8")
    _write_control(tmp_path, paths)

    completed = subprocess.run(
        _command(tmp_path, mode="run", python=python, paths=paths),
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "configuration"
    assert "summary already exists" in failure["errors"][0]
    assert not log.exists()


@pytest.mark.parametrize(
    ("failure_stage", "expected_stage", "with_report"),
    (
        ("preflight", "preflight", False),
        ("runtime_source", "runtime_source", False),
        ("evaluation", "evaluation", False),
        ("verification", "verification", False),
        ("receipt_verification", "receipt_verification", False),
        ("report", "report", True),
    ),
)
def test_child_failures_preserve_structured_stage_diagnostics(
    tmp_path: Path,
    failure_stage: str,
    expected_stage: str,
    with_report: bool,
) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        failure_stage=failure_stage,
        report=with_report,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == expected_stage
    assert failure_stage + " sentinel" in json.dumps(failure["diagnostic"])
    assert not paths["summary"].exists()


def test_successful_run_writes_a_private_complete_no_clobber_summary(
    tmp_path: Path,
) -> None:
    completed, paths, log = _execute(tmp_path, report=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary == result
    assert summary["stage"] == "complete"
    source_bundle_digest = (
        "sha256:" + hashlib.sha256(paths["source_bundle"].read_bytes()).hexdigest()
    )
    assert summary["source"] == {
        "bundle_sha256": source_bundle_digest,
        "commit": SOURCE_COMMIT,
        "execution_tree_sha256": runtime_qualification._authenticate_source_bundle(  # noqa: SLF001
            paths["source_bundle"],
            declared_digest=source_bundle_digest,
            source_commit=SOURCE_COMMIT,
            root=ROOT,
        ),
    }
    assert summary["runtime"] == {
        "image": RUNTIME_DIGEST,
        "image_digest": RUNTIME_DIGEST,
    }
    manifest_bytes = paths["candidate_manifest"].read_bytes()
    manifest = json.loads(manifest_bytes)
    assert summary["host_runtime"] == {
        "candidate_manifest_sha256": "sha256:"
        + hashlib.sha256(manifest_bytes).hexdigest(),
        "candidate_wheels": [
            {
                "distribution": "invarlock",
                "filename": Path(manifest["wheels"][0]["path"]).name,
                "sha256": manifest["wheels"][0]["sha256"],
                "version": "0.12.1",
            }
        ],
        "python": {
            "path": str(tmp_path / "qualification-python"),
            "resolved_path": str(tmp_path / "qualification-python"),
            "sha256": "sha256:"
            + hashlib.sha256(
                (tmp_path / "qualification-python").read_bytes()
            ).hexdigest(),
        },
    }
    assert summary["evidence"]["pack_manifest_digest"] == PACK_DIGEST
    assert summary["canary"] == {
        "compatibility": {
            "acceptance": {"kind": "builtin_metric", "metric": "exact_match"},
            "device_classes": {"baseline": "cuda", "subject": "cuda"},
            "providers": {
                "baseline": "hf_transformers",
                "subject": "hf_transformers",
            },
            "task": "text_causal",
        },
        "pack_manifest_digest": CANARY_PACK_DIGEST,
        "receipt_sha256": "sha256:"
        + hashlib.sha256(paths["canary_receipt"].read_bytes()).hexdigest(),
        "runtime_image_digest": RUNTIME_DIGEST,
    }
    assert summary["verification_inputs"]["trust_profile_digest"] == TRUST_DIGEST
    assert summary["receipt"]["sha256"] == (
        "sha256:" + hashlib.sha256(paths["receipt"].read_bytes()).hexdigest()
    )
    assert summary["report"]["sha256"] == (
        "sha256:" + hashlib.sha256(paths["report"].read_bytes()).hexdigest()
    )
    assert stat.S_IMODE(paths["summary"].stat().st_mode) == 0o600
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    assert {Path(entry["engine_path"]).parent.name for entry in invocations} == {
        "engine-bin"
    }
    assert {entry["engine_resolved_path"] for entry in invocations} == {
        str(tmp_path / "docker")
    }
    canary_check = next(
        entry
        for entry in invocations
        if entry["arguments"][0].endswith("qualification_receipt_check.py")
        and "--expected-runtime-image-digest" in entry["arguments"]
    )
    assert (
        canary_check["arguments"][
            canary_check["arguments"].index("--expected-runtime-device") + 1
        ]
        == "cuda:0"
    )
    assert [
        entry["arguments"][2]
        for entry in invocations
        if entry["arguments"][:2] == ["-m", "invarlock"]
    ] == [
        "evaluate",
        "evaluate",
        "verify",
        "report",
    ]
    assert (
        sum(
            entry["arguments"][0].endswith("qualification_receipt_check.py")
            for entry in invocations
        )
        == 3
    )


def test_preflight_rejects_container_engine_mutation(tmp_path: Path) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        failure_stage="container_engine_mutation",
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "preflight"
    assert failure["errors"] == ["container engine changed after configuration"]
    assert not paths["summary"].exists()


def test_run_rejects_a_renderer_pack_substitution(tmp_path: Path) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        report=True,
        binding_mutation="report_pack",
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["stage"] == "report_binding"
    assert not paths["summary"].exists()


@pytest.mark.parametrize(
    ("mutation", "stage", "report"),
    (
        ("evaluation_destination", "evaluation_binding", False),
        ("request_changed", "evaluation_binding", False),
        ("verification_pack", "verification_binding", False),
        ("report_destination", "report_binding", True),
        ("receipt_changed", "report_binding", True),
    ),
)
def test_run_rejects_cross_stage_binding_drift(
    tmp_path: Path,
    mutation: str,
    stage: str,
    report: bool,
) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        binding_mutation=mutation,
        report=report,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stderr)["stage"] == stage
    assert not paths["summary"].exists()


def test_run_accepts_request_relative_evidence_publication(tmp_path: Path) -> None:
    completed, paths, _log = _execute(
        tmp_path,
        binding_mutation="evaluation_relative",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["evidence"]["pack_manifest_digest"] == (
        PACK_DIGEST
    )
    assert paths["summary"].is_file()


def test_child_environment_drops_caller_path_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source-tree"
    root.joinpath("src", "invarlock").mkdir(parents=True)
    shadow_bin = tmp_path / "shadow-bin"
    shadow_bin.mkdir()
    monkeypatch.setenv("PATH", f"{shadow_bin}{os.pathsep}{os.defpath}")
    working = tmp_path / "work"
    working.mkdir()
    candidate_site = tmp_path / "candidate-site"
    candidate_site.mkdir()
    context = runtime_qualification.ExecutionContext(
        source_root=root,
        working_directory=working,
        child_path=os.defpath.split(os.pathsep)[0],
        candidate_site=candidate_site,
        candidate_manifest_sha256="sha256:" + "1" * 64,
        candidate_wheels=(
            runtime_qualification.CandidateWheelIdentity(
                distribution="invarlock",
                version="0.12.1",
                filename="invarlock.whl",
                sha256="sha256:" + "2" * 64,
            ),
        ),
        python_identity=runtime_qualification._python_identity(  # noqa: SLF001
            sys.executable
        ),
    )
    environment = runtime_qualification._child_environment(context)  # noqa: SLF001

    assert environment["PATH"] == context.child_path
    assert "PYTHONPATH" not in environment
    assert environment["INVARLOCK_QUALIFICATION_CANDIDATE_SITE"] == str(candidate_site)


def test_child_execution_uses_candidate_invarlock_not_snapshot_or_cwd(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source-tree"
    authenticated = root / "src" / "invarlock"
    working = tmp_path / "work"
    shadow = working / "invarlock"
    authenticated.mkdir(parents=True)
    shadow.mkdir(parents=True)
    authenticated.joinpath("__init__.py").write_text("", encoding="utf-8")
    authenticated.joinpath("__main__.py").write_text(
        'print("snapshot")\n', encoding="utf-8"
    )
    shadow.joinpath("__init__.py").write_text("", encoding="utf-8")
    shadow.joinpath("__main__.py").write_text('print("shadowed")\n', encoding="utf-8")
    candidate_site = tmp_path / "candidate-site"
    candidate_package = candidate_site / "invarlock"
    candidate_package.mkdir(parents=True)
    candidate_package.joinpath("__init__.py").write_text("", encoding="utf-8")
    context = runtime_qualification.ExecutionContext(
        source_root=root,
        working_directory=working,
        child_path=os.defpath.split(os.pathsep)[0],
        candidate_site=candidate_site,
        candidate_manifest_sha256="sha256:" + "1" * 64,
        candidate_wheels=(
            runtime_qualification.CandidateWheelIdentity(
                distribution="invarlock",
                version="0.12.1",
                filename="invarlock.whl",
                sha256="sha256:" + "2" * 64,
            ),
        ),
        python_identity=runtime_qualification._python_identity(  # noqa: SLF001
            sys.executable
        ),
    )
    completed = runtime_qualification._run(  # noqa: SLF001
        [
            sys.executable,
            "-c",
            "import invarlock; print(invarlock.__file__)",
        ],
        context=context,
        stage="test",
    )

    imported = Path(completed.stdout.strip()).resolve()
    assert imported.is_relative_to(candidate_site)
    assert completed.args[1:4] == ["-I", "-S", "-c"]


def test_isolated_bootstrap_does_not_execute_site_hooks(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = (
        venv / "Lib" / "site-packages"
        if os.name == "nt"
        else venv / "lib" / version / "site-packages"
    )
    site_packages.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / "site-hook-ran"
    payload = f"from pathlib import Path; Path({str(sentinel)!r}).write_text('ran')\n"
    site_packages.joinpath("sitecustomize.py").write_text(payload, encoding="utf-8")
    site_packages.joinpath("malicious.py").write_text(payload, encoding="utf-8")
    site_packages.joinpath("malicious.pth").write_text(
        "import malicious\n", encoding="utf-8"
    )
    candidate_site = tmp_path / "candidate-site"
    package = candidate_site / "invarlock"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text("", encoding="utf-8")
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    context = runtime_qualification.ExecutionContext(
        source_root=tmp_path,
        working_directory=tmp_path,
        child_path=os.defpath.split(os.pathsep)[0],
        candidate_site=candidate_site,
        candidate_manifest_sha256="sha256:" + "1" * 64,
        candidate_wheels=(
            runtime_qualification.CandidateWheelIdentity(
                distribution="invarlock",
                version="0.12.1",
                filename="invarlock.whl",
                sha256="sha256:" + "2" * 64,
            ),
        ),
        python_identity=runtime_qualification._python_identity(  # noqa: SLF001
            str(python)
        ),
    )

    completed = runtime_qualification._run(  # noqa: SLF001
        [str(python), "-c", "import invarlock; print(invarlock.__file__)"],
        context=context,
        stage="test",
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve().is_relative_to(candidate_site)
    assert not sentinel.exists()


def test_candidate_wheel_digest_mismatch_fails_before_extraction(
    tmp_path: Path,
) -> None:
    source_bundle = tmp_path / "source.tar"
    source_bundle.write_bytes(_source_bundle_bytes())
    wheel, _manifest = _candidate_wheel(tmp_path, source_bundle)
    candidate_site = tmp_path / "candidate-site"
    candidate_site.mkdir()
    archived = runtime_qualification._source_archive_files(  # noqa: SLF001
        source_bundle.read_bytes(),
        source_commit=SOURCE_COMMIT,
    )

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="digest does not match manifest",
    ):
        runtime_qualification._capture_candidate_wheel(  # noqa: SLF001
            runtime_qualification.CandidateWheelSpec(
                path=wheel,
                sha256="sha256:" + "0" * 64,
            ),
            archived=archived,
            candidate_site=candidate_site,
        )

    assert not any(candidate_site.iterdir())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            '{"format_version":"invarlock/qualification-candidate-wheels-v1",'
            '"format_version":"invarlock/qualification-candidate-wheels-v1",'
            '"wheels":[]}',
            "strict JSON",
        ),
        ('{"format_version":NaN,"wheels":[]}', "strict JSON"),
        ("[]", "JSON object"),
        ("{}", "contract is invalid"),
        (
            '{"format_version":"invarlock/qualification-candidate-wheels-v1",'
            '"wheels":[]}',
            "inventory is invalid",
        ),
        (
            '{"format_version":"invarlock/qualification-candidate-wheels-v1",'
            '"wheels":[{}]}',
            "entry is invalid",
        ),
        (
            '{"format_version":"invarlock/qualification-candidate-wheels-v1",'
            '"wheels":[{"path":null,"sha256":"sha256:' + "0" * 64 + '"}]}',
            "path is invalid",
        ),
        (
            '{"format_version":"invarlock/qualification-candidate-wheels-v1",'
            '"wheels":[{"path":"missing.whl","sha256":"sha256:' + "0" * 64 + '"}]}',
            "wheel is unavailable",
        ),
    ],
)
def test_candidate_wheel_manifest_rejects_malformed_or_incomplete_input(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    manifest = tmp_path / "candidate-wheels.json"
    manifest.write_text(payload, encoding="utf-8")

    with pytest.raises(runtime_qualification.QualificationError, match=message):
        runtime_qualification._candidate_wheel_specs(manifest)  # noqa: SLF001


def test_candidate_wheel_source_substitution_fails_with_matching_digest(
    tmp_path: Path,
) -> None:
    source_bundle = tmp_path / "source.tar"
    source_bundle.write_bytes(_source_bundle_bytes())
    wheel, _manifest = _candidate_wheel(tmp_path, source_bundle)
    substituted = tmp_path / "substituted.whl"
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(substituted, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for member in source.infolist():
            payload = source.read(member)
            if member.filename == "invarlock/__init__.py":
                payload += b"\nTAMPERED = True\n"
            target.writestr(member, payload)
    archived = runtime_qualification._source_archive_files(  # noqa: SLF001
        source_bundle.read_bytes(),
        source_commit=SOURCE_COMMIT,
    )
    candidate_site = tmp_path / "candidate-site"
    candidate_site.mkdir()

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="sources do not match authenticated source",
    ):
        runtime_qualification._capture_candidate_wheel(  # noqa: SLF001
            runtime_qualification.CandidateWheelSpec(
                path=substituted,
                sha256="sha256:" + hashlib.sha256(substituted.read_bytes()).hexdigest(),
            ),
            archived=archived,
            candidate_site=candidate_site,
        )


def test_candidate_wheel_change_during_capture_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bundle = tmp_path / "source.tar"
    source_bundle.write_bytes(_source_bundle_bytes())
    wheel, _manifest = _candidate_wheel(tmp_path, source_bundle)
    archived = runtime_qualification._source_archive_files(  # noqa: SLF001
        source_bundle.read_bytes(),
        source_commit=SOURCE_COMMIT,
    )
    candidate_site = tmp_path / "candidate-site"
    candidate_site.mkdir()
    real_write = runtime_qualification._write_candidate_member  # noqa: SLF001
    changed = False

    def write_then_change(destination: Path, payload: bytes) -> None:
        nonlocal changed
        real_write(destination, payload)
        if not changed:
            with wheel.open("ab") as handle:
                handle.write(b"changed")
            changed = True

    monkeypatch.setattr(
        runtime_qualification,
        "_write_candidate_member",
        write_then_change,
    )

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="changed while captured",
    ):
        runtime_qualification._capture_candidate_wheel(  # noqa: SLF001
            runtime_qualification.CandidateWheelSpec(
                path=wheel,
                sha256="sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
            ),
            archived=archived,
            candidate_site=candidate_site,
        )


def test_python_identity_rejects_executable_mutation(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    shutil.copy2(sys.executable, executable)
    executable.chmod(0o700)
    identity = runtime_qualification._python_identity(str(executable))  # noqa: SLF001
    with executable.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="changed after binding",
    ):
        runtime_qualification._assert_python_identity(  # noqa: SLF001
            identity,
            stage="test",
        )
