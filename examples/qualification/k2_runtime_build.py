"""Prepare a reviewable native K2 image context from authenticated inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

from examples.qualification import k2_runtime_apt as apt
from examples.qualification import k2_runtime_source as source

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "examples/qualification/k2-horizon/runtime"
LOCK = ROOT / "requirements/workflows/k2-campaign-py312.txt"


def _read(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("build input is not a bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("build input exceeds size bound")
    return data


def _apt_inputs(bundle, expected):
    manifest = _read(bundle / "deb-artifacts.sha256", 1024 * 1024)
    if hashlib.sha256(manifest).hexdigest() != expected:
        raise ValueError(
            "OS artifact manifest differs from the independent expected identity"
        )
    payloads = {"apt/deb-artifacts.sha256": manifest}
    total = 0
    for line in manifest.decode().splitlines():
        match = re.fullmatch(
            r"([0-9a-f]{64})  /out/debs/([A-Za-z0-9_.:+%~-]+\.deb)", line
        )
        if match is None:
            raise ValueError("OS artifact manifest contains an unsupported path")
        expected_digest, name = match.groups()
        key = f"apt/debs/{name}"
        if key in payloads:
            raise ValueError("OS artifact manifest repeats a package path")
        data = _read(bundle / "debs" / name, 256 * 1024 * 1024)
        total += len(data)
        if (
            total > 2 * 1024 * 1024 * 1024
            or hashlib.sha256(data).hexdigest() != expected_digest
        ):
            raise ValueError("OS artifact size or identity differs")
        payloads[key] = data
    if len(payloads) == 1:
        raise ValueError("OS artifact manifest is empty")
    report = apt.verify(
        bundle,
        {
            name.removeprefix("apt/debs/"): data
            for name, data in payloads.items()
            if name.startswith("apt/debs/")
        },
    )
    expected_files = {
        "deb-packages.tsv": report["package_table_sha256"],
        "package-indexes/ubuntu-archive-keyring.gpg": report["keyring_sha256"],
    }
    for item in report["indexes"]:
        expected_files["package-indexes/" + item["path"]] = item["sha256"]
        expected_files["repository-metadata/" + item["release"]] = item[
            "release_sha256"
        ]
    for name, expected_digest in expected_files.items():
        data = _read(bundle / name, 64 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise ValueError(
                "authenticated OS metadata changed before context preparation"
            )
        payloads["apt/" + name] = data
    payloads["apt/repository-metadata/ubuntu.sources"] = _read(
        bundle / "repository-metadata/ubuntu.sources", 65536
    )
    payloads["apt/package-verification.json"] = (
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    ).encode()
    return payloads


def prepare(
    archive,
    core_wheel,
    expected_core_wheel,
    output,
    *,
    apt_bundle,
    expected_apt_manifest,
):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_core_wheel):
        raise ValueError("expected core wheel must be an independently supplied SHA256")
    wheel = _read(core_wheel, 32 * 1024 * 1024)
    if hashlib.sha256(wheel).hexdigest() != expected_core_wheel:
        raise ValueError("candidate core wheel identity differs")
    payloads = {
        "Dockerfile": _read(RUNTIME / "Dockerfile", 65536),
        "requirements.txt": _read(LOCK, 1024 * 1024),
        "native_probe.py": _read(
            ROOT / "examples/qualification/k2_native_probe.py", 65536
        ),
        "core.whl": wheel,
        "os-security-pins.txt": _read(RUNTIME / "os-security-pins.txt", 65536),
        **_apt_inputs(apt_bundle, expected_apt_manifest),
    }
    for name in ("k2_runtime_source.py", "k2_runtime_build.py", "k2_runtime_apt.py"):
        payloads[f"preparation/{name}"] = _read(
            ROOT / "examples/qualification" / name, 65536
        )
    for relative in ("k2_campaign.py", "k2_producer.py", "k2-horizon/catalog.json"):
        payloads[f"examples/qualification/{relative}"] = _read(
            ROOT / "examples/qualification" / relative, 1024 * 1024
        )
    catalog = json.loads(payloads["examples/qualification/k2-horizon/catalog.json"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        derived = source.prepare(archive, output / "source")
        for name, data in payloads.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        inputs = {
            "format": "invarlock/k2-runtime-build-inputs-v1",
            "status": "prepared_not_built",
            "source_commit": source.COMMIT,
            "source_archive_sha256": source.ARCHIVE_SHA256,
            "derived_distribution_version": source.DERIVED_VERSION,
            "source_derivation_sha256": hashlib.sha256(
                (output / "source/source-derivation.json").read_bytes()
            ).hexdigest(),
            "input_sha256": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in sorted(payloads.items())
            },
            "reviewed_source_files": catalog["reviewed_source_files"],
            "excluded_operations": derived["excluded_operations"],
            "rust_extensions": "not built; selected Python HTTP path must pass native imports and GPU preflight",
            "kernel_loading": "bundled sgl_kernel FA3, pinned FlashInfer cubin and cu130 JIT cache, network disabled during execution",
        }
        (output / "build-inputs.json").write_text(
            json.dumps(inputs, sort_keys=True, indent=2) + "\n"
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--expected-core-wheel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apt-bundle", type=Path, required=True)
    parser.add_argument("--expected-apt-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(
            args.archive,
            args.core_wheel,
            args.expected_core_wheel_sha256,
            args.output,
            apt_bundle=args.apt_bundle,
            expected_apt_manifest=args.expected_apt_manifest_sha256,
        )
    except (
        ValueError,
        OSError,
        KeyError,
        subprocess.SubprocessError,
        lzma.LZMAError,
    ) as error:
        parser.exit(2, f"K2 runtime build: {error}\n")
    print(
        json.dumps(
            {"status": result["status"], "source_commit": result["source_commit"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
