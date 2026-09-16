"""Bind current example execution to its complete, locally declared public profile."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def current_profile(definition: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = [root / "runner_support.py", root / definition["runner"]]
    paths.extend(root / asset for asset in definition["runner_assets"])
    document = bytearray()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        body = path.read_bytes()
        document.extend(len(relative).to_bytes(8, "big"))
        document.extend(relative)
        document.extend(len(body).to_bytes(8, "big"))
        document.extend(body)
    return {
        "authority": definition["authority"],
        "execution": {
            "dependency_lock_sha256": "sha256:"
            + hashlib.sha256((root / definition["lock"]).read_bytes()).hexdigest(),
            "runner_sha256": "sha256:" + hashlib.sha256(document).hexdigest(),
        },
        "format": "invarlock/evaluator-qualification-profile-v1",
        "profile_id": definition["profile_id"],
        "upstream": {
            "package": definition["upstream"],
            "project_url": definition["project_url"],
        },
    }


def require_current_profile(
    profile: dict[str, Any], definition: dict[str, Any], root: Path
) -> None:
    if profile != current_profile(definition, root):
        raise ValueError(
            "execution profile does not match its complete current definition"
        )
