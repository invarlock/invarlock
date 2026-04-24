from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.integration.packaging._support_installed_wheel import (
    InstalledWheelEnv,
    _build_evidence_pack,
    _build_valid_report,
    _run,
    _write_json,
    _write_runtime_manifest,
)

pytestmark = pytest.mark.integration

_HELP_ENV = {
    "COLUMNS": "160",
    "INVARLOCK_LIGHT_IMPORT": "1",
    "NO_COLOR": "1",
    "TERM": "dumb",
}


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
        env=_HELP_ENV,
    )
    assert root_help.returncode == 0, root_help.stdout + root_help.stderr
    assert "evaluate" in root_help.stdout
    assert "verify" in root_help.stdout
    assert "report" in root_help.stdout

    evaluate_help = _run(
        installed_wheel_env.cli_exe,
        ["evaluate", "--help"],
        cwd=tmp_path,
        env=_HELP_ENV,
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
        installed_wheel_env.cli_exe,
        [
            "advanced",
            "runtime-verify",
            "--report",
            str(report_path),
            "--manifest",
            str(report_dir / "runtime.manifest.json"),
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
    manifest_path = report_dir / "runtime.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    result = _run(
        installed_wheel_env.cli_exe,
        [
            "advanced",
            "runtime-verify",
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
