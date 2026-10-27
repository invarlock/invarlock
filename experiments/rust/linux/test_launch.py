"""Cheap local checks; never build native code or launch a comparison."""

import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "linux_launch", Path(__file__).with_name("launch.py")
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class LaunchChecks(unittest.TestCase):
    def test_rebinding_changes_only_matching_path_roots(self):
        original = {
            "files": {"/old/fixture/input": "sha256:fixed"},
            "request": [
                "/old/fixture",
                "$OUTPUT/receipt.json",
                "/old/fixtures/unrelated",
            ],
        }
        rebound = launch.rebind(original, "/old/fixture", "/linux/fixture")
        self.assertEqual(rebound["files"], {"/linux/fixture/input": "sha256:fixed"})
        self.assertEqual(
            rebound["request"],
            ["/linux/fixture", "$OUTPUT/receipt.json", "/old/fixtures/unrelated"],
        )
        self.assertIn("/old/fixture/input", original["files"])

    def test_bundle_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_text("retained")
            (root / "bundle.json").write_text(
                json.dumps({"files": {"file": launch.digest(root / "file")}})
            )
            launch.verify_bundle(root)
            (root / "file").write_text("changed")
            with self.assertRaises(ValueError):
                launch.verify_bundle(root)

    def test_fixture_copy_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_text("synthetic fixture")
            (root / "link").symlink_to(root / "file")
            with self.assertRaises(ValueError):
                launch.copy_verified(
                    root / "link", launch.digest(root / "file"), root / "copy"
                )

    def test_storage_rejects_host_sharing_and_overlay(self):
        for filesystem in ("virtiofs", "overlayfs", "tmpfs", "apfs"):
            with (
                self.subTest(filesystem=filesystem),
                patch.object(sys, "platform", "linux"),
                patch.object(launch, "capture", return_value=filesystem),
            ):
                with self.assertRaises(RuntimeError):
                    launch.require_linux_storage(Path("/work"))
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(launch, "capture", return_value="ext2/ext3"),
        ):
            self.assertEqual(launch.require_linux_storage(Path("/work")), "ext2/ext3")

    def test_source_archive_rejects_escape_and_links(self):
        for name, kind in (
            ("../outside", tarfile.REGTYPE),
            ("native/link", tarfile.SYMTYPE),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with tarfile.open(root / "source.tar", "w") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/outside" if kind == tarfile.SYMTYPE else ""
                    archive.addfile(member, io.BytesIO())
                with self.assertRaises(ValueError):
                    launch.extract_native(root / "source.tar", root / "source")


if __name__ == "__main__":
    unittest.main()
