from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from invarlock.reporting import verify_contract as verify_mod
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


@dataclass(frozen=True)
class InstalledWheelEnv:
    repo_root: Path
    env_dir: Path
    wheel_path: Path
    python_exe: Path
    cli_exe: Path
    runtime_verify_exe: Path


def _select_python(repo_root: Path) -> Path:
    workspace_python = repo_root / (
        ".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"
    )
    if workspace_python.exists():
        return workspace_python

    proc = subprocess.run(
        ["/bin/bash", str(repo_root / "scripts" / "select_workspace_python.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return Path(proc.stdout.strip())

    current = Path(sys.executable)
    if current.exists() and sys.version_info >= (3, 12):
        return current
    pytest.skip("Could not locate a Python 3.12+ interpreter for wheel smoke.")


def _build_wheel(tmp_path: Path, python_exe: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    shutil.rmtree(repo_root / "build", ignore_errors=True)
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
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


def _sibling_console_script(cli_exe: Path, name: str) -> Path:
    suffix = cli_exe.suffix if cli_exe.suffix else ""
    return cli_exe.with_name(f"{name}{suffix}")


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
    _install_requirements_file(repo_root, python_exe, requirements)


def _install_requirements_file(
    repo_root: Path, python_exe: Path, requirements: Path
) -> None:
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(requirements),
        ],
        check=True,
        cwd=repo_root,
    )


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
    assert install.returncode == 0, install.stdout + install.stderr


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


def _build_evidence_pack(pack_dir: Path) -> Path:
    final_verdict = pack_dir / "results" / "final_verdict.json"
    source_repo = pack_dir / "metadata" / "source_repo.json"
    environment = pack_dir / "metadata" / "environment.json"
    materials = pack_dir / "metadata" / "model_revisions.json"
    report_rel_path = "reports/model/clean/noop/evaluation.report.json"
    report_path = pack_dir / report_rel_path

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(materials, {"models": {"org/model": {"revision": "rev1"}}})
    _write_json(report_path, _build_valid_report())
    _write_runtime_manifest(report_path)

    covered = [
        "results/final_verdict.json",
        "metadata/source_repo.json",
        "metadata/environment.json",
        "metadata/model_revisions.json",
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
            }
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

    runtime_verify_exe = _sibling_console_script(cli_exe, "invarlock-runtime-verify")
    assert cli_exe.is_file()
    assert runtime_verify_exe.is_file()
    return InstalledWheelEnv(
        repo_root=repo_root,
        env_dir=env_dir,
        wheel_path=wheel,
        python_exe=python_exe,
        cli_exe=cli_exe,
        runtime_verify_exe=runtime_verify_exe,
    )
