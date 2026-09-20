#!/usr/bin/env python3
"""Run the maintained core-wheel consumers outside the source checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGE_FILES = (
    "wheel_smoke.py",
    "request.yaml",
    "plan.json",
    "measurements.json",
    "baseline_run.json",
    "subject_run.json",
    "analysis_policy.json",
    "collection.json",
)
APPROVAL = "examples/evaluator-qualification/signed-transactions/deployment-approval-inspect-ai"
FILES = (
    ("examples/quickstart/run.py", "run.py"),
    ("examples/captured-results/wheel_smoke.py", "captured-wheel-smoke.py"),
    ("examples/captured-results/scorer_wheel_smoke.py", "scorer-wheel-smoke.py"),
    *((f"examples/judge-measurements/{name}", f"judge/{name}") for name in JUDGE_FILES),
    (
        f"{APPROVAL}/verification.receipt.json",
        "approval/incoming/verification.receipt.json",
    ),
)
DIRECTORIES = (
    ("examples/acceptance-handoff/golden", "golden"),
    ("examples/ci/standalone-consumer", "approval"),
    (f"{APPROVAL}/evidence", "approval/incoming/evidence"),
)


def check_core_install(cli: str) -> None:
    """Check the minimal installed package before exercising its consumers."""
    import sysconfig

    import invarlock

    site = Path(sysconfig.get_path("purelib")).resolve()
    assert Path(invarlock.__file__).resolve().is_relative_to(site)
    from importlib import import_module
    from importlib.metadata import distribution, version
    from importlib.util import find_spec

    assert version("invarlock") == invarlock.__version__
    assert find_spec("numpy") is None
    assert find_spec("PIL") is None

    import invarlock.cli.app
    import invarlock.judge_measurements as judge
    from invarlock.core.registry import CoreRegistry
    from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI

    assert Path(judge.__file__).resolve().is_relative_to(site)
    assert callable(judge.import_export) and callable(judge.prepare_collection)
    expected = {"hf_transformers", "hf_vision_text", "llama_cpp", "tensorrt_llm"}
    installed_entries = {
        entry.name
        for entry in distribution("invarlock").entry_points
        if entry.group == "invarlock.runtime_providers"
    }
    assert installed_entries == expected
    registry = CoreRegistry()
    for name in sorted(expected):
        provider = registry.get_runtime_provider(name)
        assert provider.name == name
        assert provider.abi_version == INVARLOCK_RUNTIME_PROVIDER_ABI
        info = registry.get_plugin_info(name, "runtime_providers")
        assert info["package"] == "invarlock"
        assert info["version"] == invarlock.__version__
        assert info["entry_point"] is None
        module = import_module(provider.__class__.__module__)
        assert Path(module.__file__).resolve().is_relative_to(site)
    for optional in (
        "torch",
        "transformers",
        "llama_cpp",
        "tensorrt_llm",
        "inspect_ai",
        "openai",
    ):
        assert optional not in sys.modules, f"optional backend imported: {optional}"
    subprocess.run([cli, "--help"], check=True)
    for module in ("llama_cpp_conformance", "tensorrt_llm_conformance"):
        subprocess.run(
            [sys.executable, "-I", "-m", f"invarlock.runtime_providers.{module}"],
            check=True,
        )


def run_consumers(cli: str) -> None:
    executable = shutil.which(cli)
    if executable is None:
        raise RuntimeError(
            "Install the candidate wheel and supply its invarlock executable"
        )
    executable = str(Path(executable).absolute())
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    with tempfile.TemporaryDirectory(prefix="invarlock-core-consumers-") as directory:
        root = Path(directory).resolve()
        if root.is_relative_to(ROOT):
            raise RuntimeError(
                "Wheel consumers require a temporary directory outside the checkout"
            )
        for source, destination in DIRECTORIES:
            shutil.copytree(ROOT / source, root / destination)
        for source, destination in FILES:
            target = root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
        commands = (
            (".", ("run.py", "--fixture", "golden")),
            (".", ("captured-wheel-smoke.py", "--cli", executable)),
            (".", ("judge/wheel_smoke.py", "--fixture", "judge", "--cli", executable)),
            (".", ("scorer-wheel-smoke.py", "--fixture", "judge", "--cli", executable)),
            (
                "approval",
                (
                    "review/verify_deployment_receipt.py",
                    "--approval-inputs",
                    "review/inspect-ai-deployment-approval-inputs.json",
                    "--evidence",
                    "incoming/evidence",
                    "--policy",
                    "review/policy/acceptance.json",
                    "--receipt",
                    "incoming/verification.receipt.json",
                    "--output",
                    "deployment-approval.json",
                ),
            ),
        )
        # Reject source imports before any consumer can produce misleading evidence.
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--mode",
                "check-core",
                "--cli",
                executable,
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        for working_directory, arguments in commands:
            subprocess.run(
                [sys.executable, *arguments],
                cwd=root / working_directory,
                env=environment,
                check=True,
            )


def run_optional_consumers(cli: str) -> None:
    """Exercise optional host features from an isolated installed wheel."""
    executable = shutil.which(cli)
    if executable is None:
        raise RuntimeError(
            "Install the candidate wheel and supply its invarlock executable"
        )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    with tempfile.TemporaryDirectory(
        prefix="invarlock-optional-consumers-"
    ) as directory:
        root = Path(directory).resolve()
        if root.is_relative_to(ROOT):
            raise RuntimeError(
                "Wheel consumers require a temporary directory outside the checkout"
            )
        commands = (
            (
                sys.executable,
                "-c",
                "from pathlib import Path; from sysconfig import get_path; "
                "import PIL, numpy, invarlock; "
                "site = Path(get_path('purelib')).resolve(); "
                "assert Path(invarlock.__file__).resolve().is_relative_to(site); "
                "assert Path(PIL.__file__).resolve().is_relative_to(site); "
                "assert Path(numpy.__file__).resolve().is_relative_to(site)",
            ),
            (
                sys.executable,
                "-c",
                "from invarlock.diagnostics import (DiagnosticInputError, "
                "canonical_observation_bytes, spectral_observation, "
                "rmt_observation, variance_observation)\n"
                "matrix = [[1.0, 0.0], [0.0, 1.0]]\n"
                "observations = [spectral_observation(matrix), "
                "rmt_observation(matrix), variance_observation([1.0, 3.0])]\n"
                "assert [o['kind'] for o in observations] == "
                "['spectral', 'rmt', 'variance']\n"
                "assert all(o['status'] == 'observation' for o in observations)\n"
                "assert observations[0]['rank'] == 2\n"
                "assert observations[1]['varying_feature_count'] == 2\n"
                "assert observations[2]['population_variance'] == 1.0\n"
                "for observation in observations:\n"
                "    assert canonical_observation_bytes(observation) == "
                "canonical_observation_bytes(dict(reversed(list(observation.items()))))\n"
                "for scorer in (spectral_observation, rmt_observation, "
                "variance_observation):\n"
                "    try:\n"
                "        scorer([[float('nan')]])\n"
                "    except DiagnosticInputError:\n"
                "        pass\n"
                "    else:\n"
                "        raise AssertionError('non-finite diagnostic input accepted')\n",
            ),
            (
                sys.executable,
                "-m",
                "invarlock.runtime_providers.hf_vision_text_conformance",
            ),
        )
        for command in commands:
            subprocess.run(command, cwd=root, env=environment, check=True)


def exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True)
    parser.add_argument(
        "--mode", choices=("core", "optional", "check-core"), default="core"
    )
    args = parser.parse_args()
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, exit_on_signal)
    if args.mode == "check-core":
        check_core_install(args.cli)
    elif args.mode == "optional":
        run_optional_consumers(args.cli)
    else:
        run_consumers(args.cli)


if __name__ == "__main__":
    main()
