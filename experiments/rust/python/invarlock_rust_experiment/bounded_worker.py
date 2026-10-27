"""Versioned worker with bounded installed-wheel identity checking.

Historical workers retain their original bytes. This worker changes only the
identity-check allocation boundary; transaction dispatch and output stay shared.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

from . import full_worker
from .worker import sha_file


def check_wheel_binding(metadata_path):
    """Bind every imported package member without allocating its entire contents."""
    metadata = json.loads(Path(metadata_path).read_bytes())
    bindings = {}
    for key, distribution_name, prefix in (
        ("root_wheel", "invarlock", "invarlock/"),
        ("native_wheel", "invarlock-native-experiment", "invarlock_native/"),
    ):
        item = metadata[key]
        if sha_file(item["path"]) != item["sha256"].removeprefix("sha256:"):
            raise RuntimeError(f"{key} wheel identity changed")
        distribution = importlib.metadata.distribution(distribution_name)
        checked = 0
        with zipfile.ZipFile(item["path"]) as wheel:
            for member in wheel.infolist():
                if member.filename.startswith(prefix) and not member.is_dir():
                    with wheel.open(member) as stream:
                        expected = hashlib.file_digest(stream, "sha256").hexdigest()
                    installed = distribution.locate_file(member.filename)
                    if expected != sha_file(installed):
                        raise RuntimeError(
                            f"installed bytes differ from {key}: {member.filename}"
                        )
                    checked += 1
        if checked == 0:
            raise RuntimeError(f"no package files bound for {key}")
        bindings[key] = {"sha256": item["sha256"], "installed_files_checked": checked}
    return bindings


def main():
    # A fresh worker owns this explicit checker selection. The historical module
    # remains unchanged for reproduction of earlier measurements.
    full_worker.check_wheel_binding = check_wheel_binding
    return full_worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
