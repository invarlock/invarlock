"""Observe an exact local K2 image and bind an explicit operator security review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from examples.qualification import k2_runtime_build as build
from examples.qualification import k2_runtime_source as source

LIMIT = 64 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or "\\" in name
    ):
        raise ValueError("artifact path is not canonical and relative")
    return path


def read(path, limit=LIMIT):
    # Check directory components as well as the leaf; descriptor traversal rejects
    # directory symlink replacement between checking and opening the next component.
    path = Path(path).absolute()
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        leaf = os.open(
            path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=descriptor
        )
        try:
            info = os.fstat(leaf)
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("artifact is not a bounded regular file")
            with os.fdopen(leaf, "rb", closefd=False) as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError("artifact exceeds size bound")
            return data
        finally:
            os.close(leaf)
    finally:
        os.close(descriptor)


def objects(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def decode(data):
    result = json.loads(data, object_pairs_hook=objects)
    if not isinstance(result, dict):
        raise ValueError("JSON artifact must be an object")
    return result


def read_json(path):
    return decode(read(path))


def run(command, *, timeout=300, limit=LIMIT):
    """Bound both pipe output and elapsed execution; never accumulate stderr."""
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as process:
        chunks, size, deadline = [], 0, time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(command, timeout)
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("command output exceeds size bound")
                    chunks.append(chunk)
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return b"".join(chunks)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def container(image, entrypoint, arguments):
    identifier = (
        run(
            [
                "docker",
                "create",
                "--pull=never",
                "--network",
                "none",
                "--read-only",
                "--user",
                "65532:65532",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "256",
                "--memory",
                "8g",
                "--cpus",
                "4",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=1g",
                "--env",
                "HOME=/tmp",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--entrypoint",
                entrypoint,
                image,
                *arguments,
            ],
            timeout=60,
            limit=1024,
        )
        .decode()
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("engine did not return an exact container ID")
    try:
        output = run(["docker", "start", "--attach", identifier])
        state = decode(
            run(
                ["docker", "inspect", "--format", "{{json .State}}", identifier],
                timeout=30,
            )
        )
        if state.get("ExitCode") != 0 or state.get("OOMKilled") is not False:
            raise ValueError("container execution failed")
        return output
    finally:
        run(["docker", "rm", "--force", identifier], timeout=30, limit=4096)


def tree_entries(root):
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("prepared input tree root must be a real directory")
    entries, pending = {}, [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                info = child.stat(follow_symlinks=False)
                name = str(Path(child.path).relative_to(root))
                if len(entries) >= 20000 or not (
                    stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
                ):
                    raise ValueError(
                        "prepared input tree is not bounded regular files and directories"
                    )
                entries[name] = "directory" if stat.S_ISDIR(info.st_mode) else "file"
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(child.path))
    return entries


def verify_context(context, archive):
    if any(
        os.path.lexists(context / name)
        for name in (".dockerignore", "Dockerfile.dockerignore")
    ):
        raise ValueError("unexpected Docker ignore input")
    data = read(context / "build-inputs.json", 1024 * 1024)
    inputs = decode(data)
    if (
        inputs.get("format") != "invarlock/k2-runtime-build-inputs-v1"
        or inputs.get("status") != "prepared_not_built"
        or inputs.get("source_commit") != source.COMMIT
        or inputs.get("source_archive_sha256") != source.ARCHIVE_SHA256
        or inputs.get("derived_distribution_version") != source.DERIVED_VERSION
    ):
        raise ValueError("prepared source identity differs")
    wheel_name = str(relative(inputs["core_wheel_filename"]))
    if "/" in wheel_name:
        raise ValueError("core wheel filename must be a basename")
    wheel_path = "core/" + wheel_name
    # Recreate the prepared manifest with the maintained derivation, authenticated
    # APT metadata, and current launcher/lock bytes. The wheel hash is subsequently
    # bound by the operator review, not treated as independent provenance.
    with tempfile.TemporaryDirectory(prefix="k2-finalize-") as temporary:
        expected = build.prepare(
            archive,
            context / wheel_path,
            inputs["input_sha256"][wheel_path],
            Path(temporary).resolve() / "context",
            apt_bundle=context / "apt",
            pip_wheel=context / "bootstrap" / build.PIP_WHEEL,
            expected_apt_manifest=inputs["input_sha256"]["apt/deb-artifacts.sha256"],
        )
        for name in ("source", "core", "apt", "bootstrap", "examples", "preparation"):
            if tree_entries(context / name) != tree_entries(
                Path(temporary).resolve() / "context" / name
            ):
                raise ValueError(
                    "prepared input tree contains missing or unexpected entries"
                )
        if inputs != expected:
            raise ValueError("prepared build inputs differ from reconstructed inputs")
        for name, digest in inputs["input_sha256"].items():
            if sha(read(context / str(relative(name)), 256 * 1024 * 1024)) != digest:
                raise ValueError("prepared input identity differs")
        manifest = read_json(
            Path(temporary).resolve() / "context/source/source-derivation.json"
        )
        if (
            sha(read(context / "source/source-derivation.json"))
            != inputs["source_derivation_sha256"]
        ):
            raise ValueError("source derivation identity differs")
        for name, digest in manifest["derived_files"].items():
            if sha(read(context / "source" / str(relative(name)))) != digest:
                raise ValueError("derived source file identity differs")
    return inputs, sha(data)


def lock_packages(data):
    entries = re.findall(
        r"^([A-Za-z0-9_.-]+)==([^\s\\]+) \\", data.decode(), re.MULTILINE
    )
    pins = {re.sub(r"[-_.]+", "-", name.lower()): version for name, version in entries}
    if len(pins) != 204 or len(pins) != len(entries):
        raise ValueError("runtime lock must contain the complete 204 unique pins")
    return pins


CAMPAIGN_FILES = (
    "examples/qualification/k2_campaign.py",
    "examples/qualification/k2_producer.py",
    "examples/qualification/k2-horizon/catalog.json",
)

INVENTORY = """import hashlib, importlib.metadata, json, os, pathlib, re, stat, subprocess

def package_inventory(distributions):
    packages = {}
    for distribution in distributions:
        name = re.sub(r'[-_.]+', '-', distribution.metadata['Name'].lower())
        if name in packages:
            raise ValueError(f'duplicate installed distribution identity: {name}')
        packages[name] = distribution.version
    return packages

def campaign_hash(name):
    descriptor = os.open('/opt/campaign/' + name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise ValueError('campaign input must be a bounded regular file')
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError('campaign input exceeds size bound')
    return hashlib.sha256(data).hexdigest()

campaign_files = {name: campaign_hash(name) for name in CAMPAIGN_FILES}
root = pathlib.Path('/usr/share/invarlock-k2')
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
packages = package_inventory(importlib.metadata.distributions())
print(json.dumps({'campaign_files': campaign_files, 'files': files, 'packages': packages, 'os_packages': subprocess.check_output(['dpkg-query', '-W']).decode(), 'wheel_artifacts': (root / 'wheel-artifacts.sha256').read_text(), 'native_probe_sha256': hashlib.sha256(pathlib.Path('/opt/campaign/native_probe.py').read_bytes()).hexdigest()}))
""".replace("CAMPAIGN_FILES", repr(CAMPAIGN_FILES))


def observe(image, inputs, build_hash, context):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("image must be an exact immutable local image ID")
    inspection = decode(
        run(["docker", "image", "inspect", "--format", "{{json .}}", image], timeout=30)
    )
    if (
        inspection.get("Id") != image
        or inspection.get("Os") != "linux"
        or inspection.get("Architecture") != "amd64"
    ):
        raise ValueError("local image identity or platform differs")
    observed = decode(container(image, "/opt/k2/bin/python", ["-c", INVENTORY]))
    if observed.get("campaign_files") != {
        name: inputs["input_sha256"][name] for name in CAMPAIGN_FILES
    }:
        raise ValueError("installed campaign files differ from prepared identities")
    files = observed["files"]
    expected = {
        "build-inputs.json": build_hash,
        "source-derivation.json": inputs["source_derivation_sha256"],
        "requirements.txt": inputs["input_sha256"]["requirements.txt"],
    }
    expected.update(
        {
            name.removeprefix("apt/"): digest
            for name, digest in inputs["input_sha256"].items()
            if name.startswith("apt/") and not name.startswith("apt/debs/")
        }
    )
    expected.update(
        {
            name: digest
            for name, digest in inputs["input_sha256"].items()
            if name.startswith("bootstrap/")
        }
    )
    expected["os-security-pins.txt"] = inputs["input_sha256"]["os-security-pins.txt"]
    if any(files.get(name) != digest for name, digest in expected.items()):
        raise ValueError("installed build artifacts differ")
    if observed.get("native_probe_sha256") != inputs["input_sha256"][
        "native_probe.py"
    ] or not re.search(
        r"^"
        + inputs["input_sha256"]["core/" + inputs["core_wheel_filename"]]
        + r"  /tmp/core/"
        + re.escape(inputs["core_wheel_filename"])
        + r"$",
        observed.get("wheel_artifacts", ""),
        re.MULTILINE,
    ):
        raise ValueError("installed probe or core wheel record differs")
    packages = observed["packages"]
    pins = lock_packages(read(context / "requirements.txt", 1024 * 1024))
    if (
        any(packages.get(name) != version for name, version in pins.items())
        or set(packages) != set(pins) | {"sglang", "invarlock"}
        or packages.get("sglang") != source.DERIVED_VERSION
        or packages.get("invarlock") != inputs["core_distribution_version"]
    ):
        raise ValueError("installed Python inventory differs from the exact lock")
    if files.get("os-packages.txt") != sha(observed["os_packages"].encode()):
        raise ValueError("installed OS inventory differs from build inventory")
    probe = decode(
        container(image, "/opt/k2/bin/python", ["/opt/campaign/native_probe.py"])
    )
    if (
        probe.get("status") != "cpu_imports_passed_not_gpu_qualified"
        or probe.get("gpu_execution") is not False
        or probe.get("build_inputs_sha256") != build_hash
        or probe.get("packages") != packages
    ):
        raise ValueError("native CPU probe identity or result differs")
    return observed, probe


def security_review(path, image, inputs, build_hash, observed):
    raw = read(path, 1024 * 1024)
    review = decode(raw)
    fields = {
        "format",
        "image_digest",
        "build_inputs_sha256",
        "requirements_sha256",
        "decision",
        "reviewer",
        "rationale",
        "unresolved_findings",
        "artifacts",
    }
    if (
        set(review) != fields
        or review["format"] != "invarlock/k2-runtime-security-review-v1"
        or review["image_digest"] != image
        or review["build_inputs_sha256"] != build_hash
        or review["requirements_sha256"] != inputs["input_sha256"]["requirements.txt"]
        or review["decision"]
        not in {"blocked", "rejected", "accepted_for_bounded_gpu_preflight"}
        or any(
            not isinstance(review[key], str) or not review[key].strip()
            for key in ("reviewer", "rationale")
        )
        or not isinstance(review["unresolved_findings"], list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in review["unresolved_findings"]
        )
        or (
            review["decision"] == "accepted_for_bounded_gpu_preflight"
            and review["unresolved_findings"]
        )
        or not isinstance(review["artifacts"], dict)
        or set(review["artifacts"]) != {"python", "os", "source", "applicability"}
    ):
        raise ValueError("security review is invalid or unbound")
    reports, payloads = {}, {"security-review.json": raw}
    for kind, artifact in review["artifacts"].items():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise ValueError("security artifact reference is invalid")
        data = read(path.parent / str(relative(artifact["path"])))
        if sha(data) != artifact["sha256"]:
            raise ValueError("security artifact identity differs")
        reports[kind] = decode(data)
        payloads[kind + "-review.json"] = data
    python, os_report, source_report = (
        reports[key] for key in ("python", "os", "source")
    )
    # The Python report's count is only a scan result; observed versions above
    # supply the inventory and the review binds the exact lock and raw report.
    if (
        type(python.get("components")) is not int
        or python["components"] != len(observed["packages"]) - 2
        or not isinstance(python.get("findings"), list)
    ):
        raise ValueError("Python scan does not cover the locked closure")
    if (
        os_report.get("ArtifactType") != "container_image"
        or os_report.get("Metadata", {}).get("ImageID") != image
        or not isinstance(os_report.get("Results"), list)
        or not any(item.get("Class") == "os-pkgs" for item in os_report["Results"])
    ):
        raise ValueError("OS scan is not bound to the exact image")
    if (
        source_report.get("source_commit") != inputs["source_commit"]
        or source_report.get("archive_sha256") != inputs["source_archive_sha256"]
        or source_report.get("source_derivation_manifest_sha256")
        != inputs["source_derivation_sha256"]
        or source_report.get("all_derived_hashes_verified") is not True
    ):
        raise ValueError("source review identity differs")
    findings = [
        finding
        for item in os_report["Results"]
        if item.get("Class") == "os-pkgs"
        for finding in item.get("Vulnerabilities", [])
    ]
    applicability = reports["applicability"]
    if (
        set(applicability)
        != {"format", "image_digest", "os_scan_sha256", "scope", "findings"}
        or applicability["format"] != "invarlock/k2-runtime-applicability-v1"
        or applicability["image_digest"] != image
        or applicability["os_scan_sha256"] != review["artifacts"]["os"]["sha256"]
        or applicability["scope"] != "offline_fixed_qualification"
        or not isinstance(applicability["findings"], list)
    ):
        raise ValueError("applicability review is invalid or unbound")
    rows, unresolved = [], set()
    for item in applicability["findings"]:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "advisory",
                "package",
                "installed_version",
                "scanner_severity",
                "decision",
                "rationale",
            }
            or any(
                not isinstance(value, str) or not value.strip()
                for value in item.values()
            )
            or item["decision"]
            not in {"unresolved", "not_applicable", "accepted_for_scope"}
        ):
            raise ValueError("applicability disposition is invalid")
        rows.append(
            tuple(
                item[key]
                for key in (
                    "advisory",
                    "package",
                    "installed_version",
                    "scanner_severity",
                )
            )
        )
        if item["decision"] == "unresolved":
            unresolved.add(item["advisory"])
    if Counter(rows) != Counter(
        tuple(
            item[key]
            for key in ("VulnerabilityID", "PkgName", "InstalledVersion", "Severity")
        )
        for item in findings
    ):
        raise ValueError("applicability dispositions do not cover every raw finding")
    if not unresolved.issubset(review["unresolved_findings"]) or (
        unresolved and review["decision"] == "accepted_for_bounded_gpu_preflight"
    ):
        raise ValueError(
            "unresolved applicability findings cannot be accepted or omitted"
        )
    if (
        python["findings"]
        and review["decision"] == "accepted_for_bounded_gpu_preflight"
    ):
        raise ValueError("Python findings remain unresolved")
    summary = {
        key: review[key]
        for key in ("decision", "reviewer", "rationale", "unresolved_findings")
    }
    summary.update(
        python_finding_count=len(python["findings"]),
        os_finding_count=len(findings),
        os_severities=dict(Counter(item["Severity"] for item in findings)),
        scope=applicability["scope"],
        applicability_decisions=dict(
            Counter(item["decision"] for item in applicability["findings"])
        ),
    )
    return summary, payloads


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def finalize(context, archive, image, review, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs, build_hash = verify_context(context, archive)
    observed, probe = observe(image, inputs, build_hash, context)
    if review is None:
        summary, payloads = (
            {
                "decision": "missing",
                "unresolved_findings": ["Security review is missing."],
            },
            {},
        )
    else:
        summary, payloads = security_review(review, image, inputs, build_hash, observed)
    payloads.update(
        {
            "installed-inventory.json": encode(observed),
            "native-probe.json": encode(probe),
        }
    )
    result = {
        "format": "invarlock/k2-runtime-build-v1",
        "status": "ready"
        if summary["decision"] == "accepted_for_bounded_gpu_preflight"
        else "blocked",
        "cpu_checks": "passed",
        "gpu_qualified": False,
        "image_digest": image,
        "source_commit": inputs["source_commit"],
        "reviewed_source_files": inputs.get("reviewed_source_files", {}),
        "source_bundle_digest": "sha256:" + inputs["source_archive_sha256"],
        "dependency_inventory_digest": "sha256:"
        + sha(payloads["installed-inventory.json"]),
        "security_review_digest": "sha256:"
        + sha(payloads.get("security-review.json", b"")),
        "build_inputs_sha256": build_hash,
        "security": summary,
        "artifact_sha256": {name: sha(data) for name, data in payloads.items()},
        "assurance_limit": "Unsigned local observation and operator-attributed review; no independent execution attestation or GPU qualification.",
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in {**payloads, "runtime-build.json": encode(result)}.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("context", "archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        result = finalize(
            args.context, args.archive, args.image, args.review, args.output
        )
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(2, f"K2 runtime finalization: {error}\n")
    print(json.dumps({"status": result["status"], "gpu_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
