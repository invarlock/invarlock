from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODULES = (
    "invarlock",
    "torch",
    "transformers",
    "datasets",
    "huggingface_hub",
    "accelerate",
    "yaml",
    "google.protobuf",
    "sentencepiece",
    "safetensors",
    "tiktoken",
)

CLI_CHECKS = (
    ("advanced", "evidence-pack", "--help"),
    ("report", "validate", "--help"),
)

REPO_ENTRYPOINTS = (
    "scripts/evidence_packs/run_suite.sh",
    "scripts/evidence_packs/run_pack.sh",
    "scripts/evidence_packs/verify_pack.sh",
    "scripts/evidence_packs/run_mini_pack_gate.sh",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check an evidence-pack remote environment."
    )
    parser.add_argument(
        "--module",
        dest="modules",
        action="append",
        default=[],
        help="Extra module import to require. Defaults cover the evidence-pack stack.",
    )
    parser.add_argument(
        "--cli",
        default="invarlock",
        help="CLI entrypoint to smoke-check (default: invarlock).",
    )
    parser.add_argument(
        "--only-runtime-provenance",
        action="store_true",
        help="Only validate runtime-image provenance readiness.",
    )
    parser.add_argument(
        "--skip-runtime-provenance-check",
        action="store_true",
        help="Skip runtime-image provenance validation.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repository root whose evidence-pack entrypoints must exist.",
    )
    return parser.parse_args(argv)


def required_modules(extra_modules: list[str]) -> tuple[str, ...]:
    if not extra_modules:
        return DEFAULT_MODULES
    return tuple(dict.fromkeys([*DEFAULT_MODULES, *extra_modules]))


def check_modules(modules: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            missing.append(module_name)
    return missing


def check_cli(cli_name: str) -> str | None:
    if shutil.which(cli_name) is None:
        return f"{cli_name} CLI not found on PATH."

    for args in CLI_CHECKS:
        proc = subprocess.run(
            [cli_name, *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (
                proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            )
            return f"{cli_name} {' '.join(args)} failed: {detail}"
    return None


def check_runtime_provenance() -> str | None:
    try:
        from invarlock.runtime_security import (
            RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT,
            RUNTIME_IMAGE_DIGEST_ENV,
            RUNTIME_IMAGE_LOCAL_DEFAULT,
            host_execution_allowed,
            resolve_runtime_image,
            resolve_runtime_image_digest,
            unverified_provenance_allowed,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return f"runtime security helpers unavailable: {exc}"

    if host_execution_allowed():
        return None

    image = resolve_runtime_image()
    digest = resolve_runtime_image_digest()
    if image in {RUNTIME_IMAGE_LOCAL_DEFAULT, RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT}:
        return None
    if "@sha256:" in image or digest:
        return None
    if unverified_provenance_allowed():
        return None

    return (
        f"runtime image {image!r} is not provenance-ready; set "
        f"{RUNTIME_IMAGE_DIGEST_ENV}, use a local invarlock-runtime image, or "
        "allow unverified provenance explicitly."
    )


def check_repo_root(repo_root: str) -> str | None:
    root = Path(repo_root)
    if not root.is_dir():
        return f"repo root {repo_root!r} does not exist."

    for relpath in REPO_ENTRYPOINTS:
        candidate = root / relpath
        if not candidate.is_file():
            return (
                f"repo root {repo_root!r} is missing required entrypoint {relpath!r}."
            )
        if not os.access(candidate, os.X_OK):
            return f"repo root {repo_root!r} has non-executable entrypoint {relpath!r}."
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.only_runtime_provenance:
        missing = check_modules(required_modules(list(args.modules)))
        if missing:
            print(
                "ERROR: Missing evidence-pack remote modules: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1

        cli_error = check_cli(args.cli)
        if cli_error:
            print(f"ERROR: {cli_error}", file=sys.stderr)
            return 1

        if args.repo_root:
            repo_error = check_repo_root(args.repo_root)
            if repo_error:
                print(f"ERROR: {repo_error}", file=sys.stderr)
                return 1

    if not args.skip_runtime_provenance_check:
        runtime_error = check_runtime_provenance()
        if runtime_error:
            print(f"ERROR: {runtime_error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
