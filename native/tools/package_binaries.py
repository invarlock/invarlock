"""Create a deterministic native Linux bundle from qualified release binaries."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        parser.error(
            "release bundles must be produced on the qualified Linux architecture"
        )
    members = {}
    for name in ("invarlock-native", "invarlock-native-verify"):
        path = args.binaries / name
        raw = path.read_bytes()
        if raw[:4] != b"\x7fELF":
            parser.error(f"{name} is not an ELF executable")
        members["bin/" + name] = (raw, 0o755)
    for path in [
        ROOT / "LICENSE",
        ROOT / "native/THIRD_PARTY_NOTICES.md",
        ROOT / "native/dependency-inventory.json",
        ROOT / "native/sbom.cdx.json",
        ROOT / "docs/reference/rust-native-installation.md",
        *sorted((ROOT / "native/licenses").rglob("*")),
    ]:
        if path.is_file():
            relative = path.relative_to(ROOT)
            members["share/invarlock/" + str(relative)] = (path.read_bytes(), 0o644)
    metadata = {
        "format": "invarlock/native-distribution-v1",
        "experimental": True,
        "version": "0.1.0",
        "architecture": platform.machine(),
        "os": "linux",
        "libc": list(platform.libc_ver()),
        "source_archive_sha256": digest(args.source_archive),
        "files": {
            name: hashlib.sha256(raw).hexdigest()
            for name, (raw, _) in sorted(members.items())
        },
    }
    members["manifest.json"] = (
        (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(),
        0o644,
    )
    checksums = "".join(
        f"{hashlib.sha256(raw).hexdigest()}  {name}\n"
        for name, (raw, _) in sorted(members.items())
    )
    members["SHA256SUMS"] = (checksums.encode(), 0o644)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as output,
        gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive,
    ):
        for name, (raw, mode) in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = mode
            info.mtime = 0
            archive.addfile(info, io.BytesIO(raw))
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": digest(args.output),
                "bytes": args.output.stat().st_size,
                "architecture": platform.machine(),
            }
        )
    )


if __name__ == "__main__":
    main()
