from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_text(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def build_source_repo_payload() -> dict[str, Any]:
    repo_root = _repo_root()
    remote_url = _git_text("config", "--get", "remote.origin.url")
    commit = _git_text("rev-parse", "HEAD")
    branch = _git_text("rev-parse", "--abbrev-ref", "HEAD")
    describe = _git_text("describe", "--tags", "--always", "--dirty")
    dirty = bool(_git_text("status", "--porcelain"))

    if remote_url:
        uri = f"git+{remote_url}"
    else:
        uri = repo_root.as_uri()

    payload: dict[str, Any] = {
        "uri": uri,
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
    }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write proof-pack source repository metadata."
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(build_source_repo_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
