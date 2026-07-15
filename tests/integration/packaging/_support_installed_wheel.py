from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from invarlock.reporting import verify_contract as verify_mod
from invarlock.reporting.report_provenance import compute_report_digest
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)
from tests.cli._support_verify_runtime_provenance import (
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)
_CORE_DEPENDENCY_IMPORTS = (
    "click",
    "cryptography",
    "jsonschema",
    "markdown",
    "numpy",
    "psutil",
    "pydantic",
    "rich",
    "shellingham",
    "typer",
    "yaml",
)
_NETWORK_UNAVAILABLE_MARKERS = (
    "Failed to establish a new connection",
    "NewConnectionError",
    "Temporary failure in name resolution",
    "Name or service not known",
    "nodename nor servname provided",
)


@dataclass(frozen=True)
class InstalledWheelEnv:
    repo_root: Path
    env_dir: Path
    wheel_path: Path
    python_exe: Path
    cli_exe: Path


def _python_can_build_wheel(python_exe: Path) -> bool:
    proc = subprocess.run(
        [
            str(python_exe),
            "-c",
            (
                "from importlib import metadata as md, util\n"
                "import sys, venv\n"
                "mods=('build',)\n"
                "def has_mod(name):\n"
                "    try:\n"
                "        md.version(name)\n"
                "        return True\n"
                "    except md.PackageNotFoundError:\n"
                "        return util.find_spec(name) is not None\n"
                "raise SystemExit(0 if sys.version_info >= (3, 12) "
                "and all(has_mod(name) for name in mods) else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=python_exe.parent,
    )
    return proc.returncode == 0


def _select_python(repo_root: Path) -> Path:
    workspace_python = repo_root / (
        ".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"
    )
    if workspace_python.exists() and _python_can_build_wheel(workspace_python):
        return workspace_python

    selector_env = {
        **os.environ,
        "INVARLOCK_SELECT_PYTHON_REQUIRE_MODULES": "build",
    }
    proc = subprocess.run(
        ["/bin/bash", str(repo_root / "scripts" / "select_workspace_python.sh")],
        capture_output=True,
        text=True,
        check=False,
        env=selector_env,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        selected = Path(proc.stdout.strip())
        if _python_can_build_wheel(selected):
            return selected

    current = Path(sys.executable)
    if current.exists() and _python_can_build_wheel(current):
        return current
    pytest.skip("Could not locate a Python 3.12+ interpreter with build support.")


def _build_wheel(tmp_path: Path, python_exe: Path) -> Path:
    build_base = tmp_path / "build"
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(tmp_path),
            # Setuptools otherwise materializes into the shared repository
            # ``build/`` tree.  Pytest-xdist can start several wheel checks at
            # once, so each invocation must own every mutable build path.
            "--config-setting=--global-option=build",
            f"--config-setting=--global-option=--build-base={build_base}",
        ],
        check=True,
    )
    return next(tmp_path.glob("*.whl"))


def _create_venv(tmp_path: Path, python_exe: Path) -> tuple[Path, Path, Path]:
    env_dir = tmp_path / "venv"
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "venv",
            "--system-site-packages",
            str(env_dir),
        ],
        check=True,
    )
    if os.name == "nt":
        venv_python = env_dir / "Scripts" / "python.exe"
        cli_exe = env_dir / "Scripts" / "invarlock.exe"
    else:
        venv_python = env_dir / "bin" / "python"
        cli_exe = env_dir / "bin" / "invarlock"
    return env_dir, venv_python, cli_exe


def _run(
    executable: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_env = None if env is None else {**os.environ, **env}
    return subprocess.run(
        [str(executable), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=resolved_env,
        timeout=timeout,
    )


def _python_minor_version(python_exe: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            str(python_exe),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    major, minor = proc.stdout.strip().split(".", maxsplit=1)
    return int(major), int(minor)


def _install_core_dependencies(repo_root: Path, python_exe: Path) -> None:
    if _core_dependencies_available(python_exe):
        return

    version = _python_minor_version(python_exe)
    requirements = (
        repo_root
        / "requirements"
        / "workflows"
        / (f"core-py{version[0]}{version[1]}.txt")
    )
    if not requirements.is_file():
        pytest.skip(
            f"Missing pinned core requirements for Python {version[0]}.{version[1]}"
        )
    install = _install_requirements_file(repo_root, python_exe, requirements)
    if install.returncode == 0:
        return
    combined = f"{install.stdout}{install.stderr}"
    if _pip_failed_due_to_offline_requirements(combined):
        pytest.skip(
            "Network unavailable to install core wheel-smoke dependencies into an isolated venv."
        )
    raise AssertionError(
        f"failed to install pinned core wheel-smoke dependencies\n{combined}"
    )


def _install_requirements_file(
    repo_root: Path, python_exe: Path, requirements: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(requirements),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )


def _core_dependencies_available(python_exe: Path) -> bool:
    imports = ", ".join(repr(name) for name in _CORE_DEPENDENCY_IMPORTS)
    proc = subprocess.run(
        [
            str(python_exe),
            "-c",
            (
                "import importlib.util; "
                f"mods=({imports},); "
                "raise SystemExit(0 if all(importlib.util.find_spec(name) for name in mods) else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _pip_failed_due_to_offline_requirements(output: str) -> bool:
    return _output_indicates_network_unavailable(output)


def _output_indicates_network_unavailable(output: str) -> bool:
    return any(marker in output for marker in _NETWORK_UNAVAILABLE_MARKERS)


def _ensure_hf_smoke_dependencies(installed_wheel_env: InstalledWheelEnv) -> None:
    import_check = _run(
        installed_wheel_env.python_exe,
        ["-c", "import torch, transformers"],
        cwd=installed_wheel_env.repo_root,
    )
    if import_check.returncode == 0:
        return

    install = _run(
        installed_wheel_env.python_exe,
        [
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            f"invarlock[hf] @ {installed_wheel_env.wheel_path.as_uri()}",
        ],
        cwd=installed_wheel_env.repo_root,
        timeout=1800,
    )
    combined = f"{install.stdout}{install.stderr}"
    if install.returncode != 0 and _output_indicates_network_unavailable(combined):
        pytest.skip(
            "Network unavailable to install hf extras into the installed-wheel smoke venv."
        )
    assert install.returncode == 0, combined


def _hf_cache_root_is_writable(root: Path) -> bool:
    datasets_dir = root / "datasets"
    try:
        datasets_dir.mkdir(parents=True, exist_ok=True)
        probe = datasets_dir / ".ivl_wheel_smoke_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _hf_cache_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for value in (
        os.environ.get("INVARLOCK_SMOKE_HF_HOME"),
        os.environ.get("HF_HOME"),
        str(Path.home() / ".cache" / "huggingface"),
    ):
        if not value:
            continue
        path = Path(value).expanduser()
        if path not in candidates:
            candidates.append(path)
    return candidates


def _local_hf_smoke_cache_ready(python_exe: Path, hf_home: Path) -> bool:
    probe = _run(
        python_exe,
        [
            "-c",
            (
                "from transformers import AutoModelForCausalLM, AutoTokenizer; "
                "model_id='sshleifer/tiny-gpt2'; "
                "AutoTokenizer.from_pretrained("
                "model_id, trust_remote_code=False, local_files_only=True"
                "); "
                "AutoModelForCausalLM.from_pretrained("
                "model_id, trust_remote_code=False, local_files_only=True"
                ")"
            ),
        ],
        env={
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
            "DISABLE_SAFETENSORS_CONVERSION": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_NO_TORCHVISION": "1",
        },
        timeout=300,
    )
    return probe.returncode == 0


def _resolve_hf_smoke_env(
    python_exe: Path, tmp_root: Path
) -> tuple[dict[str, str], bool]:
    writable_candidate: Path | None = None
    for candidate in _hf_cache_root_candidates():
        if not _hf_cache_root_is_writable(candidate):
            continue
        if writable_candidate is None:
            writable_candidate = candidate
        if _local_hf_smoke_cache_ready(python_exe, candidate):
            root = candidate
            return (
                {
                    "HF_HOME": str(root),
                    "HF_HUB_CACHE": str(root / "hub"),
                    "HF_DATASETS_CACHE": str(root / "datasets"),
                    "INVARLOCK_ALLOW_NETWORK": "1",
                    "INVARLOCK_DEDUP_TEXTS": "1",
                    "INVARLOCK_TINY_RELAX": "1",
                    "DISABLE_SAFETENSORS_CONVERSION": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "TRANSFORMERS_NO_TORCHVISION": "1",
                },
                True,
            )

    root = writable_candidate if writable_candidate is not None else tmp_root / ".hf"
    return (
        {
            "HF_HOME": str(root),
            "HF_HUB_CACHE": str(root / "hub"),
            "HF_DATASETS_CACHE": str(root / "datasets"),
            "INVARLOCK_ALLOW_NETWORK": "1",
            "INVARLOCK_DEDUP_TEXTS": "1",
            "INVARLOCK_TINY_RELAX": "1",
            "DISABLE_SAFETENSORS_CONVERSION": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_NO_TORCHVISION": "1",
        },
        False,
    )


def _prefetch_hf_smoke_model(
    python_exe: Path, *, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return _run(
        python_exe,
        [
            "-c",
            (
                "from transformers import AutoModelForCausalLM, AutoTokenizer; "
                "model_id='sshleifer/tiny-gpt2'; "
                "AutoTokenizer.from_pretrained(model_id, trust_remote_code=False); "
                "AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)"
            ),
        ],
        cwd=cwd,
        env=env,
        timeout=900,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_ref(path: Path, rel_path: str) -> dict[str, str]:
    return {
        "path": rel_path,
        "digest": f"sha256:{_sha256_file(path)}",
    }


def _write_runtime_manifest(report_path: Path) -> None:
    _write_json(
        report_path.parent / RUNTIME_MANIFEST_FILENAME,
        {
            "manifest_version": 1,
            "generated_at_utc": "2026-03-21T00:00:00+00:00",
            "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
            "execution_mode": "container",
            "report": {
                "filename": report_path.name,
                "path": report_path.as_posix(),
                "sha256": _sha256_file(report_path),
            },
            "config": {
                "path": None,
                "sha256": None,
                "source": "missing",
            },
            "runtime": {
                "container_execution": True,
                "image_digest": _VALID_TEST_IMAGE_DIGEST,
                "image_ref": "invarlock-runtime:local",
                "allow_network": False,
                "allow_remote_code": False,
                "allow_third_party_plugins": False,
            },
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"text": text}) for text in texts) + "\n",
        encoding="utf-8",
    )


def _write_local_jsonl_preset(path: Path, data_file: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            dataset:
              provider:
                kind: local_jsonl
              file: {json.dumps(str(data_file))}
              split: validation
              seq_len: 16
              stride: 16
              preview_n: 2
              final_n: 2
              seed: 42
            guards:
              order: []
            eval:
              metric: {{kind: ppl_causal}}
              loss: {{type: auto}}
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_valid_report() -> dict[str, object]:
    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    return {
        "schema_version": "v1",
        "run_id": "evidence-pack-wheel-smoke",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "baseline_ref": {
            "run_id": "baseline-run",
            "model_id": "model",
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "artifacts_extra": {},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "spectral": {
            "evaluated": True,
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                spectral_contract
            ),
            "measurement_contract_match": True,
        },
        "rmt": {
            "evaluated": True,
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                rmt_contract
            ),
            "measurement_contract_match": True,
        },
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
        "evaluation_windows": {
            "final": {
                "logloss": [math.log(10.0)],
                "token_counts": [1],
            }
        },
    }


def _build_strict_report() -> dict[str, object]:
    return copy.deepcopy(_strict_provenance_gate_cert())


def _build_strict_policy_pack() -> dict[str, object]:
    return copy.deepcopy(_matching_strict_policy_pack())


def _build_strict_baseline_report() -> dict[str, object]:
    """Return the independent raw baseline required for strict PPL replay."""

    return copy.deepcopy(_matching_strict_ppl_baseline())


def _strict_baseline_report_hash() -> str:
    digest = compute_report_digest(_build_strict_baseline_report())
    assert digest is not None
    return digest


def _build_evidence_pack(pack_dir: Path) -> Path:
    final_verdict = pack_dir / "results" / "final_verdict.json"
    source_repo = pack_dir / "metadata" / "source_repo.json"
    environment = pack_dir / "metadata" / "environment.json"
    materials = pack_dir / "metadata" / "model_revisions.json"
    scenarios = pack_dir / "metadata" / "scenarios.json"
    report_rel_path = "reports/model/clean/noop/evaluation.report.json"
    report_path = pack_dir / report_rel_path

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(materials, {"models": {"org/model": {"revision": "rev1"}}})
    _write_json(
        scenarios,
        {
            "scenarios": [
                {
                    "id": "clean",
                    "strictness": "must_pass",
                    "artifact_class": "evidence_only_pack",
                    "generation": {"kind": "evidence_only"},
                }
            ]
        },
    )
    _write_json(report_path, _build_valid_report())
    _write_runtime_manifest(report_path)

    covered = [
        "results/final_verdict.json",
        "metadata/source_repo.json",
        "metadata/environment.json",
        "metadata/model_revisions.json",
        "metadata/scenarios.json",
        report_rel_path,
        str((Path(report_rel_path).parent / RUNTIME_MANIFEST_FILENAME).as_posix()),
    ]
    _write_checksums(pack_dir, covered)
    manifest = {
        "format": "evidence-pack-v1",
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_file(pack_dir / "checksums.sha256"),
        "subject": {
            "name": "final_verdict",
            **_digest_ref(final_verdict, "results/final_verdict.json"),
        },
        "invocation": {
            "config_source": _digest_ref(source_repo, "metadata/source_repo.json")
        },
        "environment": _digest_ref(environment, "metadata/environment.json"),
        "materials": [
            {
                "name": "model_revisions",
                **_digest_ref(materials, "metadata/model_revisions.json"),
            },
            {
                "name": "scenarios",
                **_digest_ref(scenarios, "metadata/scenarios.json"),
            },
        ],
    }
    _write_json(pack_dir / "manifest.json", manifest)
    return pack_dir


@pytest.fixture(scope="module")
def installed_wheel_env(tmp_path_factory: pytest.TempPathFactory) -> InstalledWheelEnv:
    repo_root = Path(__file__).resolve().parents[3]
    selected_python = _select_python(repo_root)
    wheel = _build_wheel(tmp_path_factory.mktemp("wheel-dist"), selected_python)
    env_root = tmp_path_factory.mktemp("wheel-env")
    env_dir, python_exe, cli_exe = _create_venv(env_root, selected_python)
    _install_core_dependencies(repo_root, python_exe)

    install = _run(
        python_exe,
        ["-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
        cwd=env_root,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    assert cli_exe.is_file()
    return InstalledWheelEnv(
        repo_root=repo_root,
        env_dir=env_dir,
        wheel_path=wheel,
        python_exe=python_exe,
        cli_exe=cli_exe,
    )
