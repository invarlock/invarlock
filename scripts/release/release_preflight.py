#!/usr/bin/env python3
"""Fail-closed local preflight for a candidate InvarLock release.

This command deliberately keeps artifact *shape* checks separate from release
approval.  It proves that the checked-out commit is the requested clean commit,
that the two built distributions have the expected metadata and hashes, and
that an isolated installed-wheel interpreter is not importing the checkout.
It then audits the current public-evidence index. It does not publish anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import site
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

PACKAGE_NAME = "invarlock"
PREFLIGHT_SCHEMA = "invarlock/release-preflight-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_METADATA_BYTES = 1_048_576
RUNTIME_PACKAGE_SUFFIXES = frozenset({".json", ".py", ".pyi", ".yaml", ".yml"})
RUNTIME_PACKAGE_FILENAMES = frozenset({"py.typed"})
IGNORED_RUNTIME_PACKAGE_FILENAMES = frozenset({".DS_Store"})
IMPORT_AFFECTING_SUFFIXES = frozenset({".pth"})
EXECUTABLE_PAYLOAD_SUFFIXES = frozenset(
    {".dll", ".dylib", ".exe", ".pyd", ".py", ".pyc", ".pyo", ".so"}
)
_IMPORT_PROBE = """
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import invarlock
import invarlock.cli.app
import invarlock.runtime_providers.hf_transformers
import invarlock.runtime_providers.gguf_identity
import invarlock.runtime_providers.tensorrt_llm_identity

distribution = metadata.distribution(\"invarlock\")
distribution_root = Path(distribution.locate_file(\"\")).resolve()
invarlock_modules = {
    name: str(Path(module.__file__).resolve())
    for name, module in sys.modules.items()
    if (name == \"invarlock\" or name.startswith(\"invarlock.\"))
    and getattr(module, \"__file__\", None)
}
print(json.dumps({
    \"module_file\": str(Path(invarlock.__file__).resolve()),
    \"module_version\": str(invarlock.__version__),
    \"distribution_name\": str(distribution.metadata[\"Name\"]),
    \"distribution_version\": str(distribution.version),
    \"distribution_root\": str(distribution_root),
    \"package_paths\": [str(Path(item).resolve()) for item in invarlock.__path__],
    \"invarlock_modules\": invarlock_modules,
}, sort_keys=True))
"""
_FIRST_PARTY_RUNTIME_PROVIDERS = {
    "hf_transformers": {
        "module": "invarlock.runtime_providers.hf_transformers",
        "connector_status": "ready",
        "backend_delivery": "python_extra",
        "runtime_qualification": "not_probed",
        "support_tier": "core_supported",
    },
}
_PROBED_INVARLOCK_MODULES = frozenset(
    {
        "invarlock",
        "invarlock.cli.app",
        "invarlock.runtime_providers.hf_transformers",
        "invarlock.runtime_providers.gguf_identity",
        "invarlock.runtime_providers.tensorrt_llm_identity",
    }
)


try:
    from scripts.release.release_distribution_validation import (  # noqa: F401
        DistributionArtifacts,
        InstalledWheelImport,
        ReleasePreflightConfig,
        ReleasePreflightError,
        _CaseSensitiveConfigParser,
        _checkout_runtime_files,
        _directory_is_needed,
        _entry_point_group,
        _expected_entry_points,
        _find_distribution_artifacts,
        _is_import_affecting_path,
        _is_within,
        _load_hash_manifest,
        _parse_entry_points,
        _parse_package_metadata,
        _require_executable_file,
        _require_regular_file,
        _resolve_from_repo,
        _safe_archive_member_name,
        _sha256,
        _tar_member_sha256,
        _validate_checkout_bound_sdist_member,
        _validate_distribution_checkout_binding,
        _validate_egg_info_member,
        _validate_entry_points,
        _validate_generated_sdist_setup_cfg,
        _validate_sdist_entry_points,
        _validate_sdist_metadata,
        _validate_sdist_surface,
        _validate_wheel_entry_points,
        _validate_wheel_metadata,
        _validate_wheel_package_directories,
        _validate_wheel_record,
        _validate_wheel_surface,
        _zip_member_sha256,
        validate_distributions,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from release_distribution_validation import (  # noqa: F401
        InstalledWheelImport,
        ReleasePreflightConfig,
        ReleasePreflightError,
        _is_within,
        _require_executable_file,
        _require_regular_file,
        _resolve_from_repo,
        validate_distributions,
    )


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReleasePreflightError("unable to inspect the release checkout")
    return completed.stdout.strip()


def _validate_checkout_version(config: ReleasePreflightConfig) -> None:
    pyproject = config.repo_root / "pyproject.toml"
    _require_regular_file(pyproject, "checkout pyproject.toml")
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleasePreflightError("checkout pyproject.toml is unreadable") from exc
    project = payload.get("project")
    if (
        not isinstance(project, dict)
        or project.get("version") != config.expected_version
    ):
        raise ReleasePreflightError(
            "expected package version does not match the exact checkout"
        )


def validate_clean_exact_checkout(config: ReleasePreflightConfig) -> None:
    if not GIT_SHA_RE.fullmatch(config.release_sha):
        raise ReleasePreflightError(
            "release SHA must be a lowercase 40-character Git SHA"
        )
    head = _git_output(config.repo_root, "rev-parse", "HEAD")
    if head != config.release_sha:
        raise ReleasePreflightError(
            "checked-out commit does not match the requested release SHA"
        )
    if _git_output(
        config.repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise ReleasePreflightError("release checkout is not clean")
    _validate_checkout_version(config)


def _sanitized_execution_environment(
    repo_root: Path, *, allow_checkout_source: bool
) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in {"PYTHONHOME", "PYTHONPATH"} or key.startswith("INVARLOCK_"):
            environment.pop(key, None)
    if allow_checkout_source:
        environment["PYTHONPATH"] = str(repo_root / "src")
    return environment


def _parse_import_probe(payload: str) -> InstalledWheelImport:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReleasePreflightError(
            "installed-wheel import probe did not return JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ReleasePreflightError(
            "installed-wheel import probe returned an invalid payload"
        )
    fields = {
        name: value.get(name)
        for name in (
            "module_file",
            "module_version",
            "distribution_name",
            "distribution_version",
            "distribution_root",
        )
    }
    if not all(isinstance(item, str) and item for item in fields.values()):
        raise ReleasePreflightError(
            "installed-wheel import probe omitted required identity fields"
        )
    module_file = Path(str(fields["module_file"])).resolve()
    distribution_root = Path(str(fields["distribution_root"])).resolve()
    package_paths = value.get("package_paths")
    modules = value.get("invarlock_modules")
    if (
        not isinstance(package_paths, list)
        or len(package_paths) != 1
        or not isinstance(package_paths[0], str)
        or Path(package_paths[0]).resolve() != module_file.parent
    ):
        raise ReleasePreflightError(
            "installed-wheel package path is extended outside the candidate wheel"
        )
    if (
        not isinstance(modules, dict)
        or not modules
        or not _PROBED_INVARLOCK_MODULES.issubset(modules)
        or any(
            not isinstance(name, str)
            or not isinstance(path, str)
            or not _is_within(Path(path).resolve(), distribution_root)
            for name, path in modules.items()
        )
    ):
        raise ReleasePreflightError(
            "installed-wheel InvarLock modules escaped the candidate distribution"
        )
    return InstalledWheelImport(
        module_file=module_file,
        module_version=str(fields["module_version"]),
        distribution_name=str(fields["distribution_name"]),
        distribution_version=str(fields["distribution_version"]),
        distribution_root=distribution_root,
    )


def _run_isolated_wheel_command(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=_sanitized_execution_environment(cwd, allow_checkout_source=False),
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleasePreflightError("isolated installed-wheel check failed") from exc


def _install_isolated_dependency_bridge(environment_dir: Path) -> None:
    """Expose the invoking locked toolchain's site root to the child venv.

    This reuses dependencies from the already locked release toolchain without
    resolving a second environment or using the network. A plain path line does not
    execute nested ``.pth`` files in that parent site root. The candidate package is
    installed in the child site-packages, which precedes the bridge; the import probe
    then requires its package path and loaded InvarLock modules to remain there.
    """

    dependency_roots = sorted(
        {Path(raw).resolve() for raw in site.getsitepackages() if Path(raw).is_dir()}
    )
    if not dependency_roots:
        raise ReleasePreflightError(
            "isolated installed-wheel dependency environment is unavailable"
        )
    if os.name == "nt":
        child_site_packages = environment_dir / "Lib" / "site-packages"
    else:
        child_site_packages = (
            environment_dir
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
    if not child_site_packages.is_dir():
        raise ReleasePreflightError(
            "isolated installed-wheel site-packages directory is unavailable"
        )
    try:
        (child_site_packages / "invarlock-release-dependencies.pth").write_text(
            "".join(f"{root}\n" for root in dependency_roots),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ReleasePreflightError(
            "unable to bind the isolated installed-wheel dependency environment"
        ) from exc


def _require_successful_installed_wheel_command(
    command: list[str], *, cwd: Path, label: str, timeout: int = 60
) -> subprocess.CompletedProcess[str]:
    completed = _run_isolated_wheel_command(command, cwd=cwd, timeout=timeout)
    if completed.returncode != 0:
        raise ReleasePreflightError(f"installed-wheel {label} failed")
    return completed


def _smoke_installed_wheel_cli(cli: Path, *, cwd: Path) -> None:
    """Prove the installed wheel exposes the supported public journey."""

    root_help = _require_successful_installed_wheel_command(
        [str(cli), "--help"],
        cwd=cwd,
        label="root CLI help",
    )
    for command in ("evaluate", "verify", "report"):
        if command not in root_help.stdout:
            raise ReleasePreflightError(
                f"installed-wheel root help omitted {command!r}"
            )
    for retired in ("advanced", "calibrate", "doctor", "plugins"):
        if retired in root_help.stdout:
            raise ReleasePreflightError(
                f"installed-wheel root help retained retired command {retired!r}"
            )

    for command in ("evaluate", "verify", "report"):
        _require_successful_installed_wheel_command(
            [str(cli), command, "--help"],
            cwd=cwd,
            label=f"{command} help",
        )


def _probe_installed_wheel(
    config: ReleasePreflightConfig, wheel: Path
) -> InstalledWheelImport:
    with tempfile.TemporaryDirectory(
        prefix="invarlock-release-preflight-"
    ) as directory:
        cwd = Path(directory).resolve()
        if _is_within(cwd, config.repo_root):
            raise ReleasePreflightError(
                "release preflight temporary directory must be outside checkout"
            )
        environment_dir = cwd / "installed-wheel"
        create = _run_isolated_wheel_command(
            [sys.executable, "-m", "venv", str(environment_dir)],
            cwd=cwd,
            timeout=120,
        )
        if create.returncode != 0:
            raise ReleasePreflightError(
                "unable to create isolated installed-wheel environment"
            )
        _install_isolated_dependency_bridge(environment_dir)
        wheel_python = environment_dir / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        wheel_cli = environment_dir / (
            "Scripts/invarlock.exe" if os.name == "nt" else "bin/invarlock"
        )
        _require_executable_file(wheel_python, "isolated installed-wheel Python")
        install = _run_isolated_wheel_command(
            [
                str(wheel_python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--force-reinstall",
                str(wheel),
            ],
            cwd=cwd,
            timeout=120,
        )
        if install.returncode != 0:
            raise ReleasePreflightError(
                "unable to install the candidate wheel in isolation"
            )
        _require_executable_file(wheel_cli, "isolated installed-wheel CLI")
        completed = _run_isolated_wheel_command(
            [str(wheel_python), "-I", "-c", _IMPORT_PROBE],
            cwd=cwd,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ReleasePreflightError("installed-wheel import probe failed")
        imported = _parse_import_probe(completed.stdout)
        if not _is_within(imported.module_file, environment_dir) or not _is_within(
            imported.distribution_root, environment_dir
        ):
            raise ReleasePreflightError(
                "installed-wheel import did not resolve from the isolated environment"
            )
        _smoke_installed_wheel_cli(wheel_cli, cwd=cwd)
    validate_installed_wheel_import(config, imported)
    return imported


def validate_installed_wheel_import(
    config: ReleasePreflightConfig, imported: InstalledWheelImport
) -> None:
    """Require an installed distribution of the expected version outside checkout."""
    if imported.distribution_name.casefold() != PACKAGE_NAME:
        raise ReleasePreflightError(
            "installed-wheel distribution name is not invarlock"
        )
    if (
        imported.module_version != config.expected_version
        or imported.distribution_version != config.expected_version
    ):
        raise ReleasePreflightError(
            "installed-wheel version does not match expected version"
        )
    if _is_within(imported.module_file, config.repo_root) or _is_within(
        imported.distribution_root, config.repo_root
    ):
        raise ReleasePreflightError(
            "installed-wheel import resolved inside the release checkout"
        )
    if not _is_within(imported.module_file, imported.distribution_root):
        raise ReleasePreflightError(
            "installed-wheel module is outside its distribution root"
        )


def _run_current_public_evidence_audit(config: ReleasePreflightConfig) -> None:
    public_evidence_root = (config.repo_root / "public_evidence").resolve()
    command = [
        sys.executable,
        str(config.repo_root / "scripts" / "checks" / "check_public_evidence.py"),
        "--root",
        str(public_evidence_root),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=config.repo_root,
        env=_sanitized_execution_environment(
            config.repo_root, allow_checkout_source=True
        ),
    )
    if completed.returncode != 0:
        raise ReleasePreflightError("current public-evidence audit failed")


def run_release_preflight(config: ReleasePreflightConfig) -> dict[str, Any]:
    """Validate a release candidate and return only portable result details."""
    validate_clean_exact_checkout(config)
    artifacts = validate_distributions(config)
    _probe_installed_wheel(config, artifacts.wheel)
    _run_current_public_evidence_audit(config)
    validate_clean_exact_checkout(config)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "ok": True,
        "release_sha": config.release_sha,
        "package": {"name": PACKAGE_NAME, "version": config.expected_version},
        "distributions": [
            {
                "name": artifacts.wheel.name,
                "sha256": artifacts.hashes[artifacts.wheel.name],
            },
            {
                "name": artifacts.sdist.name,
                "sha256": artifacts.hashes[artifacts.sdist.name],
            },
        ],
        "installed_wheel_import": "isolated_venv_from_candidate_wheel",
        "installed_wheel_runtime_surface": "passed",
        "current_public_evidence": "passed",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ReleasePreflightConfig:
    repo_root = args.repo_root.resolve()
    return ReleasePreflightConfig(
        repo_root=repo_root,
        release_sha=args.release_sha,
        expected_version=args.expected_version,
        dist_dir=_resolve_from_repo(repo_root, args.dist_dir),
        hash_manifest=_resolve_from_repo(repo_root, args.hash_manifest),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = run_release_preflight(_config_from_args(args))
    except ReleasePreflightError as exc:
        print(f"ERROR: release preflight rejected: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Release preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
