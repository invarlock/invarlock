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

pytestmark = pytest.mark.integration


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


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_exposes_core_cli_contracts_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    import_check = _run(
        installed_wheel_env.python_exe,
        ["-c", "import invarlock; print(invarlock.__file__)"],
        cwd=tmp_path,
    )
    assert import_check.returncode == 0, import_check.stderr
    import_path = Path(import_check.stdout.strip()).resolve()
    assert installed_wheel_env.env_dir.resolve() in import_path.parents
    assert installed_wheel_env.repo_root.resolve() not in import_path.parents

    root_help = _run(
        installed_wheel_env.cli_exe,
        ["--help"],
        cwd=tmp_path,
        env={"INVARLOCK_LIGHT_IMPORT": "1"},
    )
    assert root_help.returncode == 0, root_help.stdout + root_help.stderr
    assert "evaluate" in root_help.stdout
    assert "verify" in root_help.stdout
    assert "report" in root_help.stdout

    evaluate_help = _run(
        installed_wheel_env.cli_exe,
        ["evaluate", "--help"],
        cwd=tmp_path,
        env={"INVARLOCK_LIGHT_IMPORT": "1"},
    )
    assert evaluate_help.returncode == 0, evaluate_help.stdout + evaluate_help.stderr
    assert "--baseline" in evaluate_help.stdout
    assert "--subject" in evaluate_help.stdout
    assert "--report-out" in evaluate_help.stdout

    cli_app_import = _run(
        installed_wheel_env.python_exe,
        [
            "-c",
            (
                "import json; "
                "import invarlock.cli.app; "
                "import invarlock.public_contracts as public_contracts; "
                "catalog = sorted(public_contracts.contract_catalog().keys()); "
                "published_basis = {"
                "lane['lane_id']: lane['evidence'] "
                "for lane in public_contracts.load_support_matrix()['lanes'] "
                "if lane.get('support_tier') == 'published_basis'"
                "}; "
                "print(json.dumps({"
                "'catalog': catalog, "
                "'published_basis': published_basis"
                "}, sort_keys=True))"
            ),
        ],
        cwd=tmp_path,
        env={"INVARLOCK_LIGHT_IMPORT": "1"},
    )
    assert cli_app_import.returncode == 0, cli_app_import.stdout + cli_app_import.stderr
    exported_contracts = json.loads(cli_app_import.stdout.strip())
    assert "metric_kinds" in exported_contracts["catalog"]
    assert "support_matrix" in exported_contracts["catalog"]
    assert exported_contracts["published_basis"]
    for evidence in exported_contracts["published_basis"].values():
        assert evidence["evaluation_report_fixture"].startswith(
            "public_evidence/published_basis/"
        )
        assert evidence["evidence_pack_recipe"].startswith(
            "public_evidence/published_basis/"
        )
        assert "tests/fixtures/" not in evidence["evaluation_report_fixture"]
        assert "tests/fixtures/" not in evidence["evidence_pack_recipe"]

    doctor = _run(
        installed_wheel_env.cli_exe,
        ["doctor", "--json"],
        cwd=tmp_path,
        env={"INVARLOCK_LIGHT_IMPORT": "1"},
    )
    assert doctor.returncode in (0, 1), doctor.stdout + doctor.stderr
    doctor_payload = json.loads(doctor.stdout.strip().splitlines()[-1])
    assert doctor_payload["format_version"] == "doctor-v1"

    installed_public_evidence = _run(
        installed_wheel_env.python_exe,
        [
            "-c",
            (
                "import json; "
                "from importlib import resources; "
                "import invarlock.public_contracts as public_contracts; "
                "data_root = resources.files('invarlock').joinpath('_data'); "
                "resolved = {"
                "lane['lane_id']: {"
                "key: str(data_root.joinpath(*path.split('/')))"
                "for key, path in lane.get('evidence', {}).items()"
                "} "
                "for lane in public_contracts.published_basis_lanes()"
                "}; "
                "print(json.dumps(resolved, sort_keys=True))"
            ),
        ],
        cwd=tmp_path,
    )
    assert installed_public_evidence.returncode == 0, (
        installed_public_evidence.stdout + installed_public_evidence.stderr
    )
    resolved_public_evidence = json.loads(installed_public_evidence.stdout.strip())
    assert resolved_public_evidence
    for evidence in resolved_public_evidence.values():
        for path in evidence.values():
            assert Path(path).is_file(), path

    published_report = Path(
        resolved_public_evidence["gpt2-causal-hf"]["evaluation_report_fixture"]
    )
    public_html = tmp_path / "published-basis.html"
    render_public_html = _run(
        installed_wheel_env.cli_exe,
        [
            "report",
            "html",
            "-i",
            str(published_report),
            "-o",
            str(public_html),
            "--force",
        ],
        cwd=tmp_path,
    )
    assert render_public_html.returncode == 0, (
        render_public_html.stdout + render_public_html.stderr
    )
    assert public_html.is_file()
    assert "<html" in public_html.read_text(encoding="utf-8").lower()


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
@pytest.mark.slow
def test_wheel_install_runs_front_door_evaluate_verify_report_html_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    _ensure_hf_smoke_dependencies(installed_wheel_env)

    data_file = tmp_path / "smoke.jsonl"
    _write_jsonl(
        data_file,
        [
            "installed wheel front door sample one",
            "installed wheel front door sample two",
            "installed wheel front door sample three",
            "installed wheel front door sample four",
        ],
    )
    preset_path = tmp_path / "preset.yaml"
    _write_local_jsonl_preset(preset_path, data_file)

    hf_home = tmp_path / "hf-home"
    hf_home.mkdir(parents=True, exist_ok=True)
    smoke_env = {
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "INVARLOCK_ALLOW_NETWORK": "1",
        "INVARLOCK_DEDUP_TEXTS": "1",
        "INVARLOCK_TINY_RELAX": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_NO_TORCHVISION": "1",
    }

    prefetch = _run(
        installed_wheel_env.python_exe,
        [
            "-c",
            (
                "from transformers import AutoModelForCausalLM, AutoTokenizer; "
                "model_id='sshleifer/tiny-gpt2'; "
                "AutoTokenizer.from_pretrained(model_id, trust_remote_code=False); "
                "AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)"
            ),
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=900,
    )
    assert prefetch.returncode == 0, prefetch.stdout + prefetch.stderr

    report_dir = tmp_path / "front-door-report"
    evaluate = _run(
        installed_wheel_env.cli_exe,
        [
            "evaluate",
            "--allow-network",
            "--execution-mode",
            "host",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "sshleifer/tiny-gpt2",
            "--adapter",
            "auto",
            "--profile",
            "dev",
            "--preset",
            str(preset_path),
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(report_dir),
            "--quiet",
            "--no-banner",
            "--no-progress",
            "--no-color",
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=1800,
    )
    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr

    report_path = report_dir / "evaluation.report.json"
    assert report_path.is_file()
    assert (report_dir / RUNTIME_MANIFEST_FILENAME).is_file()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["schema_version"] == "v1"

    verify = _run(
        installed_wheel_env.cli_exe,
        [
            "verify",
            "--runtime-provenance",
            "host",
            "--json",
            str(report_path),
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=300,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    verify_payload = json.loads(verify.stdout.strip().splitlines()[-1])
    assert verify_payload["format_version"] == "verify-v1"
    assert verify_payload["summary"]["ok"] is True

    html_path = tmp_path / "front-door.html"
    render_html = _run(
        installed_wheel_env.cli_exe,
        [
            "report",
            "html",
            "-i",
            str(report_path),
            "-o",
            str(html_path),
            "--force",
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=300,
    )
    assert render_html.returncode == 0, render_html.stdout + render_html.stderr
    assert html_path.is_file()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_can_verify_report_runtime_and_evidence_pack_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:

    report_dir = tmp_path / "report"
    report_path = report_dir / "evaluation.report.json"
    _write_json(report_path, _build_valid_report())
    _write_runtime_manifest(report_path)

    verify_report = _run(
        installed_wheel_env.cli_exe,
        ["verify", "--json", str(report_path)],
        cwd=tmp_path,
    )
    assert verify_report.returncode == 0, verify_report.stdout + verify_report.stderr
    verify_payload = json.loads(verify_report.stdout.strip().splitlines()[-1])
    assert verify_payload["format_version"] == "verify-v1"
    assert verify_payload["summary"]["ok"] is True

    html_path = tmp_path / "evaluation.html"
    export_html = _run(
        installed_wheel_env.cli_exe,
        ["report", "html", "-i", str(report_path), "-o", str(html_path)],
        cwd=tmp_path,
    )
    assert export_html.returncode == 0, export_html.stdout + export_html.stderr
    assert html_path.is_file()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()

    runtime_verify = _run(
        installed_wheel_env.runtime_verify_exe,
        [
            "--report",
            str(report_path),
            "--manifest",
            str(report_dir / RUNTIME_MANIFEST_FILENAME),
            "--json",
        ],
        cwd=tmp_path,
    )
    assert runtime_verify.returncode == 0, runtime_verify.stdout + runtime_verify.stderr
    runtime_payload = json.loads(runtime_verify.stdout.strip().splitlines()[-1])
    assert runtime_payload["format_version"] == "runtime-verify-v1"
    assert runtime_payload["ok"] is True

    pack_dir = _build_evidence_pack(tmp_path / "pack")

    verify = _run(
        installed_wheel_env.cli_exe,
        [
            "advanced",
            "evidence-pack",
            "verify",
            str(pack_dir),
            "--json",
        ],
        cwd=tmp_path,
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "1"},
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    payload = json.loads(verify.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_verify_rejects_ambiguous_directory_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "report.json").write_text("{}", encoding="utf-8")
    _write_json(report_dir / "evaluation.report.json", _build_valid_report())

    result = _run(
        installed_wheel_env.cli_exe, ["verify", str(report_dir)], cwd=tmp_path
    )

    combined = result.stdout + result.stderr
    normalized = " ".join(combined.split())
    assert result.returncode == 2, combined
    assert (
        "contains both report.json and evaluation.report.json; pass an explicit file path."
        in normalized
    )


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_runtime_verify_failure_json_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    report_dir = tmp_path / "report"
    report_path = report_dir / "evaluation.report.json"
    _write_json(report_path, _build_valid_report())
    _write_runtime_manifest(report_path)
    manifest_path = report_dir / RUNTIME_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    result = _run(
        installed_wheel_env.runtime_verify_exe,
        [
            "--report",
            str(report_path),
            "--manifest",
            str(manifest_path),
            "--json",
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "runtime-verify-v1"
    assert payload["ok"] is False
    assert any("report digest mismatch" in error for error in payload["errors"])


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_evidence_pack_verify_reports_integrity_failure_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    pack_dir = _build_evidence_pack(tmp_path / "pack")
    _write_json(pack_dir / "results" / "final_verdict.json", {"verdict": "TAMPERED"})

    result = _run(
        installed_wheel_env.cli_exe,
        ["advanced", "evidence-pack", "verify", str(pack_dir), "--json"],
        cwd=tmp_path,
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "1"},
    )

    assert result.returncode != 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is False
    assert any(
        "checksum mismatch for results/final_verdict.json" in error
        for error in payload["errors"]
    )
