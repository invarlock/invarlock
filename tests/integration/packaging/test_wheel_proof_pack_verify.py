from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from invarlock.reporting import verify_contract as verify_mod
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)

pytestmark = pytest.mark.integration


def _select_python(repo_root: Path) -> Path:
    current = Path(sys.executable)
    if current.exists() and sys.version_info >= (3, 12):
        return current
    proc = subprocess.run(
        ["/bin/bash", str(repo_root / "scripts" / "select_workspace_python.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        pytest.skip("Could not locate a Python 3.12+ interpreter for wheel smoke.")
    return Path(proc.stdout.strip())


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


def _run(
    executable: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_env = None if env is None else {**os.environ, **env}
    return subprocess.run(
        [str(executable), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=resolved_env,
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


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_attested_report() -> dict[str, object]:
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
        "run_id": "proof-pack-wheel-smoke",
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


def _build_proof_pack(pack_dir: Path) -> Path:
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
    _write_json(report_path, _build_attested_report())
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
        "format": "proof-pack-v1",
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


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_can_verify_proof_pack_outside_repo_tree(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    selected_python = _select_python(repo_root)
    wheel = _build_wheel(tmp_path / "dist", selected_python)
    env_dir, python_exe, cli_exe = _create_venv(tmp_path, selected_python)
    _install_core_dependencies(repo_root, python_exe)

    install = _run(
        python_exe,
        ["-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
        cwd=tmp_path,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    import_check = _run(
        python_exe,
        ["-c", "import invarlock; print(invarlock.__file__)"],
        cwd=tmp_path,
    )
    assert import_check.returncode == 0, import_check.stderr
    import_path = Path(import_check.stdout.strip()).resolve()
    assert env_dir.resolve() in import_path.parents
    assert repo_root.resolve() not in import_path.parents
    assert cli_exe.is_file()

    minimal_help = _run(
        python_exe,
        ["-m", "invarlock", "--help"],
        cwd=tmp_path,
        env={"INVARLOCK_LIGHT_IMPORT": "1"},
    )
    assert minimal_help.returncode == 0, minimal_help.stdout + minimal_help.stderr
    assert "evaluate" in minimal_help.stdout
    assert "verify" in minimal_help.stdout

    cli_app_import = _run(
        python_exe,
        [
            "-c",
            (
                "import json; "
                "import invarlock.cli.app; "
                "import invarlock.public_contracts as public_contracts; "
                "print(json.dumps(sorted(public_contracts.contract_catalog().keys())))"
            ),
        ],
        cwd=tmp_path,
        env={"INVARLOCK_LIGHT_IMPORT": "1"},
    )
    assert cli_app_import.returncode == 0, cli_app_import.stdout + cli_app_import.stderr
    exported_contracts = json.loads(cli_app_import.stdout.strip())
    assert "metric_kinds" in exported_contracts
    assert "support_matrix" in exported_contracts

    pack_dir = _build_proof_pack(tmp_path / "pack")

    verify = _run(
        cli_exe,
        [
            "advanced",
            "proof-pack",
            "verify",
            str(pack_dir),
            "--json",
        ],
        cwd=tmp_path,
        env={"INVARLOCK_ALLOW_UNATTESTED_ARTIFACTS": "1"},
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    payload = json.loads(verify.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "proof-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
