from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys

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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
