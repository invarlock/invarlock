"""Record real installed native imports and fail-closed optional operations."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def package_inventory(distributions):
    packages = {}
    for distribution in distributions:
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"].lower())
        if name in packages:
            raise ValueError(f"duplicate installed distribution identity: {name}")
        packages[name] = distribution.version
    return packages


HOST_SOURCE = b"int fixed_host_probe(void) { return 42; }\n"
HOST_COMPILER_TIMEOUT = 60


def validate_host_compiler(report, triton_version):
    if (
        not isinstance(report, dict)
        or report.get("status") != "fixed_cpu_host_compile_and_call_passed"
        or report.get("source_sha256") != hashlib.sha256(HOST_SOURCE).hexdigest()
        or report.get("triton_version") != triton_version
        or report.get("result") != 42
        or report.get("gpu_execution") is not False
        or re.fullmatch(r"[0-9a-f]{64}", str(report.get("compiled_library_sha256")))
        is None
    ):
        raise ValueError("fixed CPU host compiler observation is missing or invalid")
    return report


def fixed_host_compile():
    import ctypes
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    from triton.runtime.build import _build

    with tempfile.TemporaryDirectory(prefix="k2-fixed-host-compile-") as directory:
        root = Path(directory).resolve()
        source = root / "fixed_host_probe.c"
        source.write_bytes(HOST_SOURCE)
        library = Path(
            _build("fixed_host_probe", str(source), str(root), [], [], [], [])
        )
        if (
            library.is_symlink()
            or library.resolve().parent != root
            or library.stat().st_size > 16 * 1024 * 1024
        ):
            raise ValueError(
                "host compiler output escaped its private bounded directory"
            )
        loaded = ctypes.CDLL(str(library))
        loaded.fixed_host_probe.restype = ctypes.c_int
        result = loaded.fixed_host_probe()
        report = {
            "status": "fixed_cpu_host_compile_and_call_passed",
            "source_sha256": hashlib.sha256(HOST_SOURCE).hexdigest(),
            "compiled_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "triton_version": importlib.metadata.version("triton"),
            "result": result,
            "gpu_execution": False,
        }
        return validate_host_compiler(report, report["triton_version"])


def host_compiler_probe():
    with tempfile.TemporaryDirectory(prefix="k2-host-compiler-result-") as directory:
        output, error = Path(directory) / "stdout", Path(directory) / "stderr"
        with output.open("wb") as stdout, error.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--host-compiler"],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=HOST_COMPILER_TIMEOUT)
            except BaseException:
                # Stop the compiler's private process group on timeout/interruption.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
            if code:
                with error.open("rb") as diagnostic:
                    excerpt = diagnostic.read(4096).decode("utf-8", errors="replace")
                raise ValueError(
                    "fixed CPU host compilation or library loading failed "
                    f"(exit {code}): {excerpt}"
                )
        if output.stat().st_size > 65536:
            raise ValueError("host compiler observation exceeds size bound")
        return validate_host_compiler(
            json.loads(output.read_bytes()), importlib.metadata.version("triton")
        )


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
    host_compiler = host_compiler_probe()
    return {
        "host_compiler": host_compiler,
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
    print(
        json.dumps(
            fixed_host_compile()
            if sys.argv[1:] == ["--host-compiler"]
            else inspect_native(),
            sort_keys=True,
        )
    )
