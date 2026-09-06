"""Record real installed native imports and fail-closed optional operations."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


def package_inventory(distributions):
    packages = {}
    for distribution in distributions:
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"].lower())
        if name in packages:
            raise ValueError(f"duplicate installed distribution identity: {name}")
        packages[name] = distribution.version
    return packages


def inspect_native(root=Path("/usr/share/invarlock-k2")):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("probe requires the declared Linux x86_64 image")
    inputs = json.loads((root / "build-inputs.json").read_text())
    packages = package_inventory(importlib.metadata.distributions())
    if {"outlines", "outlines-core", "diskcache"} & packages.keys():
        raise ValueError("excluded grammar/cache distribution is installed")
    if packages.get("sglang") != inputs["derived_distribution_version"]:
        raise ValueError("installed derived distribution identity differs")
    modules = [
        "sglang",
        "sglang.srt.configs.k2_horizon",
        "sglang.srt.parser.reasoning_parser",
        "sglang.srt.models.xllm",
        "sglang.srt.entrypoints.http_server",
    ]
    for name in modules:
        importlib.import_module(name)
    package_root = Path(importlib.import_module("sglang").__file__).parent
    for name, expected in inputs["reviewed_source_files"].items():
        if name.startswith("python/sglang/"):
            actual = hashlib.sha256(
                (package_root / name.removeprefix("python/sglang/")).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise ValueError(f"installed native source identity differs: {name}")
    disabled = [
        "sglang.srt.constrained.outlines_backend",
        "sglang.srt.constrained.outlines_jump_forward",
    ]
    for name in disabled:
        try:
            importlib.import_module(name)
        except RuntimeError as error:
            if "unavailable in the restricted K2 runtime" not in str(error):
                raise
        else:
            raise ValueError(f"excluded operation unexpectedly imported: {name}")
    # Exercise the unchanged dispatcher with the upstream test context override.
    from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(grammar_backend="outlines")
    override.install()
    try:
        try:
            create_grammar_backend(None, None, 32000)
        except RuntimeError as error:
            if "unavailable in the restricted K2 runtime" not in str(error):
                raise
        else:
            raise ValueError("excluded grammar selection did not fail closed")
    finally:
        override.restore()
    help_result = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    if (
        "--model-path" not in help_result.stdout
        or "--grammar-backend" not in help_result.stdout
    ):
        raise ValueError("native help lacks the reviewed server arguments")
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        stdout=sys.stderr,
        timeout=60,
    )
    return {
        "format": "invarlock/k2-native-cpu-probe-v1",
        "status": "cpu_imports_passed_not_gpu_qualified",
        "build_inputs_sha256": hashlib.sha256(
            (root / "build-inputs.json").read_bytes()
        ).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "modules_imported": modules,
        "excluded_modules_rejected": disabled,
        "excluded_dispatch_rejected": True,
        "native_help_sha256": hashlib.sha256(help_result.stdout.encode()).hexdigest(),
        "packages": packages,
        "gpu_execution": False,
    }


if __name__ == "__main__":
    print(json.dumps(inspect_native(), sort_keys=True))
