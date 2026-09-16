"""The image receipt requires observed execution and explicit artifact review."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.qualification import k2_runtime_finalize as finalize


def write(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    image = "sha256:" + "a" * 64
    inputs = {
        "core_wheel_filename": "invarlock-0.15.0-py3-none-any.whl",
        "core_distribution_version": "0.15.0",
        "source_commit": finalize.source.COMMIT,
        "source_archive_sha256": finalize.source.ARCHIVE_SHA256,
        "source_derivation_sha256": "b" * 64,
        "derived_distribution_version": finalize.source.DERIVED_VERSION,
        "input_sha256": {
            "requirements.txt": "c" * 64,
            "core/invarlock-0.15.0-py3-none-any.whl": "d" * 64,
        },
    }
    build_hash = "e" * 64
    observed = {
        "packages": {
            "example": "1.0",
            "invarlock": "0.15.0",
            "sglang": finalize.source.DERIVED_VERSION,
        },
        "files": {},
        "os_packages": "",
    }
    probe = {"status": "cpu_imports_passed_not_gpu_qualified", "gpu_execution": False}
    monkeypatch.setattr(finalize, "verify_context", lambda *_: (inputs, build_hash))
    monkeypatch.setattr(finalize, "observe", lambda *_: (observed, probe))
    python_report = {"components": 1, "findings": []}
    os_report = {
        "ArtifactType": "container_image",
        "Metadata": {"ImageID": image, "OS": {"Family": "ubuntu", "Name": "24.04"}},
        "Results": [
            {
                "Class": "os-pkgs",
                "Type": "ubuntu",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-1",
                        "PkgName": "linux-libc-dev",
                        "InstalledVersion": "1",
                        "Severity": "CRITICAL",
                    },
                    {
                        "VulnerabilityID": "CVE-2026-2",
                        "PkgName": "libexpat1",
                        "InstalledVersion": "2",
                        "Severity": "MEDIUM",
                    },
                ],
            }
        ],
    }
    source_report = {
        "source_commit": inputs["source_commit"],
        "archive_sha256": inputs["source_archive_sha256"],
        "source_derivation_manifest_sha256": inputs["source_derivation_sha256"],
        "all_derived_hashes_verified": True,
    }
    artifacts = {}
    for kind, value in (
        ("python", python_report),
        ("os", os_report),
        ("source", source_report),
    ):
        name = kind + ".json"
        artifacts[kind] = {"path": name, "sha256": write(tmp_path / name, value)}
    applicability = {
        "format": "invarlock/k2-runtime-applicability-v1",
        "image_digest": image,
        "os_scan_sha256": artifacts["os"]["sha256"],
        "scope": "offline_fixed_qualification",
        "findings": [
            {
                "advisory": row["VulnerabilityID"],
                "package": row["PkgName"],
                "installed_version": row["InstalledVersion"],
                "scanner_severity": row["Severity"],
                "decision": "unresolved"
                if row["PkgName"] == "libexpat1"
                else "not_applicable",
                "rationale": "Example applicability assessment for this exact scope.",
            }
            for row in os_report["Results"][0]["Vulnerabilities"]
        ],
    }
    artifacts["applicability"] = {
        "path": "applicability.json",
        "sha256": write(tmp_path / "applicability.json", applicability),
    }
    review = {
        "format": "invarlock/k2-runtime-security-review-v1",
        "image_digest": image,
        "build_inputs_sha256": build_hash,
        "requirements_sha256": inputs["input_sha256"]["requirements.txt"],
        "decision": "blocked",
        "reviewer": "Example reviewer",
        "rationale": "Runtime Expat findings remain unresolved.",
        "unresolved_findings": ["CVE-2026-2"],
        "artifacts": artifacts,
    }
    review_path = tmp_path / "review.json"
    write(review_path, review)
    return tmp_path, image, review_path, review, inputs, observed, probe


def test_blocked_review_retains_cpu_success_and_raw_critical(case):
    root, image, review_path, *_ = case
    result = finalize.finalize(
        root, root / "archive", image, review_path, root / "result"
    )
    assert result["status"] == "blocked"
    assert result["cpu_checks"] == "passed"
    assert result["gpu_qualified"] is False
    assert result["security"]["os_severities"] == {"CRITICAL": 1, "MEDIUM": 1}
    assert result["security"]["decision"] == "blocked"
    assert (root / "result" / "native-probe.json").is_file()


def test_ready_requires_explicit_accepted_review_and_no_blockers(case):
    root, image, path, review, *_ = case
    applicability = json.loads((root / "applicability.json").read_text())
    for row in applicability["findings"]:
        row["decision"] = "accepted_for_scope"
    review["artifacts"]["applicability"]["sha256"] = write(
        root / "applicability.json", applicability
    )
    review.update(decision="accepted_for_bounded_gpu_preflight", unresolved_findings=[])
    write(path, review)
    result = finalize.finalize(root, root / "archive", image, path, root / "result")
    assert result["status"] == "ready"
    assert result["security"]["os_finding_count"] == 2
    assert result["gpu_qualified"] is False
    assert "local observation" in result["assurance_limit"]


@pytest.mark.parametrize(
    "change",
    [
        {"image_digest": "sha256:" + "1" * 64},
        {"build_inputs_sha256": "1" * 64},
        {"requirements_sha256": "1" * 64},
        {"decision": "clean"},
        {"reviewer": ""},
        {"rationale": ""},
        {"extra": True},
        {"unresolved_findings": "none"},
        {"decision": "accepted_for_bounded_gpu_preflight"},
    ],
)
def test_invalid_review_cannot_publish(case, change):
    root, image, path, review, *_ = case
    review.update(change)
    write(path, review)
    with pytest.raises(ValueError):
        finalize.finalize(root, root / "archive", image, path, root / "result")
    assert not (root / "result").exists()


@pytest.mark.parametrize(
    "kind,change",
    [
        ("os", {"Metadata": {"ImageID": "sha256:" + "0" * 64}}),
        ("os", {"Results": []}),
        ("os", {"ArtifactType": "filesystem"}),
        ("python", {"components": 0}),
        ("python", {"findings": "invalid"}),
        ("source", {"all_derived_hashes_verified": False}),
        ("source", {"source_commit": "0" * 40}),
    ],
)
def test_unbound_or_failing_reports_reject(case, kind, change):
    root, image, path, review, *_ = case
    artifact = root / review["artifacts"][kind]["path"]
    data = json.loads(artifact.read_text())
    data.update(change)
    review["artifacts"][kind]["sha256"] = write(artifact, data)
    write(path, review)
    with pytest.raises(ValueError):
        finalize.finalize(root, root / "archive", image, path, root / "result")


def test_changed_review_artifact_rejects(case):
    root, image, path, *_ = case
    (root / "os.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact identity"):
        finalize.finalize(root, root / "archive", image, path, root / "result")


@pytest.mark.parametrize(
    "name", ["../os.json", "/os.json", "a/../os.json", "a\\os.json"]
)
def test_review_cannot_escape_artifact_directory(case, name):
    root, image, path, review, *_ = case
    review["artifacts"]["os"]["path"] = name
    write(path, review)
    with pytest.raises(ValueError, match="path"):
        finalize.finalize(root, root / "archive", image, path, root / "result")


def test_cpu_failure_never_publishes(case, monkeypatch):
    root, image, path, *_ = case

    def fail(*_):
        raise ValueError("CPU probe failed")

    monkeypatch.setattr(finalize, "observe", fail)
    with pytest.raises(ValueError, match="CPU probe failed"):
        finalize.finalize(root, root / "archive", image, path, root / "result")
    assert not (root / "result").exists()


def test_json_rejects_duplicate_fields_and_symlinks(tmp_path):
    path = tmp_path / "input"
    path.write_text('{"decision":"blocked","decision":"accepted"}')
    with pytest.raises(ValueError, match="duplicate"):
        finalize.read_json(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        finalize.read_json(link)


def test_bounded_runner_limits_output_and_runtime():
    assert finalize.run([sys.executable, "-c", "print('ok')"], timeout=5) == b"ok\n"
    with pytest.raises(ValueError, match="output"):
        finalize.run([sys.executable, "-c", "print('x'*10000)"], timeout=5, limit=100)
    with pytest.raises(subprocess.TimeoutExpired):
        finalize.run(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.01
        )
    with pytest.raises(subprocess.CalledProcessError):
        finalize.run([sys.executable, "-c", "raise SystemExit(3)"], timeout=5)


def test_container_cleanup_uses_observed_id_and_offline_limits(monkeypatch):
    calls = []

    def fake(command, **_):
        calls.append(command)
        if command[1] == "create":
            return ("f" * 64 + "\n").encode()
        if command[1] == "inspect":
            return b'{"ExitCode": 0, "OOMKilled": false}'
        return b"{}"

    monkeypatch.setattr(finalize, "run", fake)
    assert finalize.container("sha256:" + "a" * 64, "python", ["-c", "pass"]) == b"{}"
    assert calls[-1] == ["docker", "rm", "--force", "f" * 64]
    command = calls[0]
    assert "/tmp:rw,nosuid,nodev,exec,size=1g" in command
    assert "none" in command and "--read-only" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command and "--gpus" not in command
    assert "65532:65532" in command and "--pids-limit" in command


def test_container_failure_still_removes_exact_id(monkeypatch):
    calls = []

    def fake(command, **_):
        calls.append(command)
        if command[1] == "create":
            return b"f" * 64
        if command[1] == "start":
            raise subprocess.TimeoutExpired(command, 1)
        return b""

    monkeypatch.setattr(finalize, "run", fake)
    with pytest.raises(subprocess.TimeoutExpired):
        finalize.container("sha256:" + "a" * 64, "python", [])
    assert calls[-1][-1] == "f" * 64


def test_cli_reports_input_failure(tmp_path, monkeypatch, capsys):
    def fail(*_):
        raise ValueError("bad image")

    monkeypatch.setattr(finalize, "finalize", fail)
    with pytest.raises(SystemExit) as error:
        finalize.main(
            [
                "--context",
                str(tmp_path),
                "--archive",
                "source.tar.gz",
                "--image",
                "bad",
                "--review",
                "review.json",
                "--output",
                "result",
            ]
        )
    assert error.value.code == 2
    assert "bad image" in capsys.readouterr().err


def test_missing_review_and_output_collision(case):
    root, image, *_ = case
    result = finalize.finalize(root, root / "archive", image, None, root / "result")
    assert result["status"] == "blocked"
    assert result["security"]["decision"] == "missing"
    with pytest.raises(FileExistsError):
        finalize.finalize(root, root / "archive", image, None, root / "result")


def test_python_findings_preserved_when_blocked_and_never_accepted(case):
    root, image, path, review, *_ = case
    report = {"components": 1, "findings": [{"id": "CVE-2026-3"}]}
    review["artifacts"]["python"]["sha256"] = write(root / "python.json", report)
    write(path, review)
    result = finalize.finalize(root, root / "archive", image, path, root / "result")
    assert result["security"]["python_finding_count"] == 1
    assert json.loads((root / "result/python-review.json").read_text()) == report
    applicability = json.loads((root / "applicability.json").read_text())
    for row in applicability["findings"]:
        row["decision"] = "accepted_for_scope"
    review["artifacts"]["applicability"]["sha256"] = write(
        root / "applicability.json", applicability
    )
    review.update(decision="accepted_for_bounded_gpu_preflight", unresolved_findings=[])
    write(path, review)
    with pytest.raises(ValueError, match="Python findings"):
        finalize.finalize(root, root / "archive", image, path, root / "other")


@pytest.mark.parametrize(
    "change",
    [
        {"unresolved_findings": [3]},
        {"artifacts": {}},
        {"artifacts": {"os": {}, "python": {}, "source": {}, "applicability": {}}},
    ],
)
def test_malformed_review_fields(case, change):
    root, image, path, review, *_ = case
    review.update(change)
    write(path, review)
    with pytest.raises(ValueError):
        finalize.finalize(root, root / "archive", image, path, root / "result")


def test_bounded_file_and_json_types(tmp_path):
    path = tmp_path / "data"
    path.write_text("12345")
    with pytest.raises(ValueError, match="bounded regular"):
        finalize.read(path, 3)
    with pytest.raises(ValueError, match="object"):
        finalize.decode(b"[]")
    with pytest.raises(ValueError, match="bounded regular"):
        finalize.read(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        finalize.read(link / "data")


@pytest.mark.parametrize(
    "state", [{"ExitCode": 3, "OOMKilled": False}, {"ExitCode": 0, "OOMKilled": True}]
)
def test_failed_container_state_is_removed(monkeypatch, state):
    calls = []

    def fake(command, **_):
        calls.append(command)
        return b"f" * 64 if command[1] == "create" else json.dumps(state).encode()

    monkeypatch.setattr(finalize, "run", fake)
    with pytest.raises(ValueError, match="execution failed"):
        finalize.container("sha256:" + "a" * 64, "python", [])
    assert calls[-1] == ["docker", "rm", "--force", "f" * 64]


def test_unrecognized_container_identifier_cannot_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(
        finalize, "run", lambda command, **_: calls.append(command) or b"mutable-tag"
    )
    with pytest.raises(ValueError, match="exact container ID"):
        finalize.container("sha256:" + "a" * 64, "python", [])
    assert len(calls) == 1


@pytest.fixture
def observation(tmp_path, monkeypatch):
    image = "sha256:" + "a" * 64
    lock = finalize.read(finalize.build.LOCK)
    (tmp_path / "requirements.txt").write_bytes(lock)
    pins = finalize.lock_packages(lock)
    packages = {
        **pins,
        "sglang": finalize.source.DERIVED_VERSION,
        "invarlock": "0.15.0",
    }
    inputs = {
        "core_wheel_filename": "invarlock-0.15.0-py3-none-any.whl",
        "core_distribution_version": "0.15.0",
        "source_derivation_sha256": "b" * 64,
        "input_sha256": {
            "requirements.txt": finalize.sha(lock),
            "core/invarlock-0.15.0-py3-none-any.whl": "c" * 64,
            "os-security-pins.txt": "d" * 64,
            "native_probe.py": "e" * 64,
            "apt/package-verification.json": "f" * 64,
            "apt/debs/test.deb": "0" * 64,
        },
    }
    campaign_files = {
        name: finalize.sha(name.encode()) for name in finalize.CAMPAIGN_FILES
    }
    inputs["input_sha256"].update(campaign_files)
    build_hash = "1" * 64
    observed = {
        "campaign_files": dict(campaign_files),
        "packages": packages,
        "files": {
            "build-inputs.json": build_hash,
            "source-derivation.json": "b" * 64,
            "requirements.txt": finalize.sha(lock),
            "os-security-pins.txt": "d" * 64,
            "package-verification.json": "f" * 64,
            "expat-built/build-report.json": "2" * 64,
            "os-packages.txt": finalize.sha(b"test\t1\n"),
        },
        "os_packages": "test\t1\n",
        "native_probe_sha256": "e" * 64,
        "wheel_artifacts": "c" * 64 + "  /tmp/core/invarlock-0.15.0-py3-none-any.whl\n",
    }
    expat_observation = {
        "source_version": finalize.build.expat.VERSION,
        "package_version": finalize.build.expat.PACKAGE_VERSION,
        "pyexpat_version": "expat_" + finalize.build.expat.VERSION,
        "build_report_sha256": "2" * 64,
    }
    observed["expat"] = expat_observation
    probe = {
        "status": "cpu_imports_passed_not_gpu_qualified",
        "gpu_execution": False,
        "build_inputs_sha256": build_hash,
        "packages": dict(packages),
    }
    probe["host_compiler"] = {
        "status": "fixed_cpu_host_compile_and_call_passed",
        "source_sha256": finalize.sha(finalize.native.HOST_SOURCE),
        "compiled_library_sha256": "3" * 64,
        "triton_version": packages["triton"],
        "result": 42,
        "gpu_execution": False,
    }
    inspection = {"Id": image, "Os": "linux", "Architecture": "amd64"}
    monkeypatch.setattr(
        finalize, "run", lambda *_, **__: json.dumps(inspection).encode()
    )
    monkeypatch.setattr(
        finalize,
        "container",
        lambda _image, _entry, args: json.dumps(
            observed
            if args[0] == "-c"
            else expat_observation
            if args[-1] == "verify"
            else probe
        ).encode(),
    )
    return tmp_path, image, inputs, build_hash, observed, probe, inspection


def test_observe_reads_exact_installed_lock_and_cpu_probe(observation):
    root, image, inputs, digest, observed, probe, _ = observation
    assert finalize.observe(image, inputs, digest, root) == (observed, probe)


@pytest.mark.parametrize(
    "change",
    [
        "tag",
        "image",
        "platform",
        "artifact",
        "version",
        "extra",
        "sglang",
        "os",
        "probe",
        "probe_hash",
        "core",
    ],
)
def test_observation_rejects_identity_drift(observation, change):
    root, image, inputs, digest, observed, probe, inspection = observation
    if change == "tag":
        image = "image:latest"
    elif change == "image":
        inspection["Id"] = "sha256:" + "0" * 64
    elif change == "platform":
        inspection["Architecture"] = "arm64"
    elif change == "artifact":
        observed["files"]["requirements.txt"] = "0" * 64
    elif change == "version":
        observed["packages"]["pip"] = "0"
    elif change == "extra":
        observed["packages"]["diskcache"] = "1"
    elif change == "sglang":
        observed["packages"]["sglang"] = "0"
    elif change == "os":
        observed["os_packages"] = "modified"
    elif change == "probe":
        probe["gpu_execution"] = True
    elif change == "probe_hash":
        observed["native_probe_sha256"] = "0" * 64
    else:
        observed["wheel_artifacts"] = (
            "0" * 64 + "  /tmp/core/invarlock-0.15.0-py3-none-any.whl\n"
        )
    with pytest.raises(ValueError):
        finalize.observe(image, inputs, digest, root)


def test_lock_rejects_missing_or_duplicate_pins():
    with pytest.raises(ValueError, match="204 unique pins"):
        finalize.lock_packages(b"test==1 \\\n")
    lock = finalize.read(finalize.build.LOCK)
    with pytest.raises(ValueError, match="204 unique pins"):
        finalize.lock_packages(lock + b"pip==26.2 \\\n")


@pytest.fixture
def context_case(tmp_path, monkeypatch):
    context = tmp_path / "prepared"
    context.mkdir()
    (context / "source").mkdir()
    source_hash = finalize.sha(b"source\n")
    manifest = {"derived_files": {"module.py": source_hash}}
    source_digest = write(context / "source/source-derivation.json", manifest)
    (context / "source/module.py").write_bytes(b"source\n")
    (context / "core").mkdir()
    (context / "core/invarlock-0.15.0-py3-none-any.whl").write_bytes(b"wheel")
    (context / "apt").mkdir()
    (context / "apt/deb-artifacts.sha256").write_bytes(b"debs")
    inputs = {
        "core_wheel_filename": "invarlock-0.15.0-py3-none-any.whl",
        "core_distribution_version": "0.15.0",
        "format": "invarlock/k2-runtime-build-inputs-v1",
        "status": "prepared_not_built",
        "source_commit": finalize.source.COMMIT,
        "source_archive_sha256": finalize.source.ARCHIVE_SHA256,
        "derived_distribution_version": finalize.source.DERIVED_VERSION,
        "source_derivation_sha256": source_digest,
        "input_sha256": {
            "core/invarlock-0.15.0-py3-none-any.whl": finalize.sha(b"wheel"),
            "apt/deb-artifacts.sha256": finalize.sha(b"debs"),
        },
    }
    write(context / "build-inputs.json", inputs)
    expected = json.loads(json.dumps(inputs))
    for name in ("bootstrap", "examples", "preparation", "expat"):
        (context / name).mkdir()

    def prepare(_archive, _wheel, digest, output, **kwargs):
        assert digest == finalize.sha(b"wheel")
        assert kwargs["expected_apt_manifest"] == finalize.sha(b"debs")
        for name in ("core", "apt", "bootstrap", "examples", "preparation", "expat"):
            (output / name).mkdir(parents=True)
        (output / "core/invarlock-0.15.0-py3-none-any.whl").write_bytes(b"wheel")
        (output / "apt/deb-artifacts.sha256").write_bytes(b"debs")
        (output / "source").mkdir(parents=True)
        (output / "source/module.py").write_bytes(b"source\n")
        write(output / "source/source-derivation.json", manifest)
        return expected

    monkeypatch.setattr(finalize.build, "prepare", prepare)
    return context, inputs


def test_reconstructed_context_and_each_derived_file(context_case):
    context, inputs = context_case
    assert finalize.verify_context(context, context / "archive") == (
        inputs,
        finalize.sha(finalize.read(context / "build-inputs.json")),
    )


@pytest.mark.parametrize(
    "change", ["identity", "manifest", "input", "source_manifest", "source"]
)
def test_modified_context_cannot_finalize(context_case, change):
    context, inputs = context_case
    if change == "identity":
        inputs["source_commit"] = "0" * 40
        write(context / "build-inputs.json", inputs)
    elif change == "manifest":
        inputs["extra"] = True
        write(context / "build-inputs.json", inputs)
    elif change == "input":
        (context / "core/invarlock-0.15.0-py3-none-any.whl").write_bytes(b"changed")
    elif change == "source_manifest":
        write(context / "source/source-derivation.json", {})
    else:
        (context / "source/module.py").write_bytes(b"changed")
    with pytest.raises(ValueError):
        finalize.verify_context(context, context / "archive")


def test_partial_publication_is_removed(case, monkeypatch):
    root, image, path, *_ = case
    original = Path.open

    def fail(self, *args, **kwargs):
        if self.name == "native-probe.json":
            raise OSError("write failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError, match="write failed"):
        finalize.finalize(root, root / "archive", image, path, root / "result")
    assert not (root / "result").exists()


def test_cli_success(case, capsys):
    root, image, path, *_ = case
    assert (
        finalize.main(
            [
                "--context",
                str(root),
                "--archive",
                "archive",
                "--image",
                image,
                "--review",
                str(path),
                "--output",
                str(root / "result"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "gpu_qualified": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        "image",
        "scan",
        "scope",
        "rows",
        "fields",
        "decision",
        "rationale",
        "omit",
        "severity",
        "missing_unresolved",
    ],
)
def test_applicability_must_bind_and_cover_every_raw_finding(case, change):
    root, image, path, review, *_ = case
    artifact = root / "applicability.json"
    data = json.loads(artifact.read_text())
    if change == "image":
        data["image_digest"] = "sha256:" + "0" * 64
    elif change == "scan":
        data["os_scan_sha256"] = "0" * 64
    elif change == "scope":
        data["scope"] = "public_serving"
    elif change == "rows":
        data["findings"] = "none"
    elif change == "fields":
        data["findings"][0]["extra"] = True
    elif change == "decision":
        data["findings"][0]["decision"] = "clean"
    elif change == "rationale":
        data["findings"][0]["rationale"] = ""
    elif change == "omit":
        data["findings"].pop()
    elif change == "severity":
        data["findings"][0]["scanner_severity"] = "LOW"
    else:
        review["unresolved_findings"] = []
    review["artifacts"]["applicability"]["sha256"] = write(artifact, data)
    write(path, review)
    with pytest.raises(ValueError, match="applicability"):
        finalize.finalize(root, root / "archive", image, path, root / "result")
    assert not (root / "result").exists()


def test_accepted_cli_is_scoped_and_still_not_gpu_qualified(case, capsys):
    root, image, path, review, *_ = case
    artifact = root / "applicability.json"
    data = json.loads(artifact.read_text())
    for row in data["findings"]:
        row["decision"] = "accepted_for_scope"
    review["artifacts"]["applicability"]["sha256"] = write(artifact, data)
    review.update(decision="accepted_for_bounded_gpu_preflight", unresolved_findings=[])
    write(path, review)
    assert (
        finalize.main(
            [
                "--context",
                str(root),
                "--archive",
                "archive",
                "--image",
                image,
                "--review",
                str(path),
                "--output",
                str(root / "result"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "ready",
        "gpu_qualified": False,
    }
    result = json.loads((root / "result/runtime-build.json").read_text())
    assert result["security"]["scope"] == "offline_fixed_qualification"
    assert result["security"]["os_severities"]["CRITICAL"] == 1


def test_context_core_name_cannot_include_directory(context_case):
    context, inputs = context_case
    inputs["core_wheel_filename"] = "sub/invarlock-0.15.0-py3-none-any.whl"
    write(context / "build-inputs.json", inputs)
    with pytest.raises(ValueError, match="basename"):
        finalize.verify_context(context, context / "archive")


def test_observed_core_version_follows_prepared_wheel(observation):
    root, image, inputs, digest, observed, probe, _ = observation
    previous = inputs["core_wheel_filename"]
    inputs.update(
        core_wheel_filename="invarlock-0.16.0-py3-none-any.whl",
        core_distribution_version="0.16.0",
    )
    inputs["input_sha256"]["core/" + inputs["core_wheel_filename"]] = inputs[
        "input_sha256"
    ].pop("core/" + previous)
    observed["wheel_artifacts"] = observed["wheel_artifacts"].replace(
        previous, inputs["core_wheel_filename"]
    )
    observed["packages"]["invarlock"] = "0.16.0"
    probe["packages"]["invarlock"] = "0.16.0"
    assert finalize.observe(image, inputs, digest, root) == (observed, probe)
    observed["packages"]["invarlock"] = "0.15.0"
    with pytest.raises(ValueError, match="inventory"):
        finalize.observe(image, inputs, digest, root)


@pytest.mark.parametrize("name", ["source/extra.py", "core/extra.whl", "expat/json.py"])
def test_extra_copied_input_is_rejected(context_case, name):
    context, _ = context_case
    (context / name).write_bytes(b"unexpected executable input")
    with pytest.raises(ValueError, match="unexpected entries"):
        finalize.verify_context(context, context / "archive")


def test_input_tree_rejects_symlinks(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="bounded regular"):
        finalize.tree_entries(tmp_path)


def test_input_tree_includes_nested_directories_and_rejects_linked_root(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/file").write_bytes(b"data")
    assert finalize.tree_entries(tmp_path) == {
        "nested": "directory",
        "nested/file": "file",
    }
    (tmp_path / "alias").symlink_to(tmp_path / "nested")
    with pytest.raises(ValueError, match="real directory"):
        finalize.tree_entries(tmp_path / "alias")


@pytest.mark.parametrize("name", [".dockerignore", "Dockerfile.dockerignore"])
def test_implicit_docker_ignore_input_is_rejected(context_case, name):
    context, _ = context_case
    (context / name).write_text("source/python/sglang/optional.py\n")
    with pytest.raises(ValueError, match="Docker ignore"):
        finalize.verify_context(context, context / "archive")


@pytest.mark.parametrize("name", finalize.CAMPAIGN_FILES)
@pytest.mark.parametrize("change", ["missing", "changed"])
def test_observation_rejects_missing_or_changed_installed_campaign(
    observation, name, change
):
    root, image, inputs, digest, observed, *_ = observation
    if change == "missing":
        observed["campaign_files"].pop(name)
    else:
        observed["campaign_files"][name] = "0" * 64
    with pytest.raises(ValueError, match="installed campaign files"):
        finalize.observe(image, inputs, digest, root)


def test_observation_requires_campaign_inventory(observation):
    root, image, inputs, digest, observed, *_ = observation
    observed.pop("campaign_files")
    with pytest.raises(ValueError, match="installed campaign files"):
        finalize.observe(image, inputs, digest, root)


def test_embedded_inventory_rejects_real_canonical_duplicates(tmp_path):
    import ast
    import re
    from importlib.metadata import distributions

    definition = next(
        node
        for node in ast.parse(finalize.INVENTORY).body
        if isinstance(node, ast.FunctionDef) and node.name == "package_inventory"
    )
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), "<inventory>", "exec"),
        namespace,
    )
    roots = []
    for index, (name, version) in enumerate(
        (("Example.Package", "0.0.0"), ("example_package", "26.0"))
    ):
        root = tmp_path / str(index)
        metadata = root / f"example_{index}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        roots.append(str(root))
    first = list(distributions(path=roots[:1]))
    assert namespace["package_inventory"](first) == {"example-package": "0.0.0"}
    with pytest.raises(ValueError, match="duplicate installed distribution"):
        namespace["package_inventory"](distributions(path=roots))


def test_installed_expat_observation_cannot_use_old_version(observation):
    root, image, inputs, digest, observed, *_ = observation
    observed["expat"]["pyexpat_version"] = "expat_2.6.1"
    with pytest.raises(ValueError, match="Expat observation"):
        finalize.observe(image, inputs, digest, root)


def test_image_side_expat_module_injection_is_rejected_before_helper(
    observation, monkeypatch
):
    root, image, inputs, digest, observed, *_ = observation
    observed["files"]["expat/json.py"] = "0" * 64
    calls = []

    def container(_image, _entry, args):
        calls.append(args)
        assert args[0] == "-c"
        return json.dumps(observed).encode()

    monkeypatch.setattr(finalize, "container", container)
    with pytest.raises(ValueError, match="Expat input tree"):
        finalize.observe(image, inputs, digest, root)
    assert len(calls) == 1


def test_native_probe_cannot_omit_required_host_compiler_check(observation):
    root, image, inputs, digest, _, probe, _ = observation
    probe.pop("host_compiler")
    with pytest.raises(ValueError, match="host compiler observation"):
        finalize.observe(image, inputs, digest, root)
