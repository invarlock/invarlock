"""Qualify the explicit source archive; building it is a separate retained check."""

import os
import tarfile
import tomllib
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("INVARLOCK_NATIVE_SDIST"),
    reason="requires a built native source archive",
)
def test_source_archive_retains_complete_workspace():
    source = Path(__file__).resolve().parents[1]
    expected = tomllib.loads((source / "Cargo.toml").read_text())
    with tarfile.open(os.environ["INVARLOCK_NATIVE_SDIST"]) as archive:
        members = archive.getmembers()
        assert all(item.isfile() or item.isdir() for item in members)
        root = members[0].name.split("/")[0]

        def read(relative):
            stream = archive.extractfile(f"{root}/{relative}")
            assert stream is not None
            return stream.read()

        actual = tomllib.loads(read("Cargo.toml").decode())
        assert actual["workspace"] == expected["workspace"]
        required = [
            Path(name)
            for name in (
                "Cargo.lock",
                "rust-toolchain.toml",
                "LICENSE",
                "README.md",
                "pyproject.toml",
                "dependency-inventory.json",
                "sbom.cdx.json",
                "THIRD_PARTY_NOTICES.md",
            )
        ]
        for name in expected["workspace"]["members"]:
            required.extend(
                path.relative_to(source)
                for path in (source / name).rglob("*")
                if path.is_file()
                and path.suffix in {".toml", ".rs", ".json", ".svg", ".css", ".js"}
            )
        for dependency in expected.get("patch", {}).get("crates-io", {}).values():
            if "path" in dependency:
                required.extend(
                    path.relative_to(source)
                    for path in (source / dependency["path"]).rglob("*")
                    if path.is_file()
                )
        required.extend(
            path.relative_to(source)
            for path in (source / "licenses").rglob("*")
            if path.is_file()
        )
        for relative in required:
            assert read(relative) == (source / relative).read_bytes(), str(relative)
