from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.integration.packaging._support_installed_wheel import (
    _VALID_TEST_IMAGE_DIGEST,
    InstalledWheelEnv,
    _build_evidence_pack,
    _build_strict_baseline_report,
    _build_strict_policy_pack,
    _build_strict_report,
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

    verifier_imports = _run(
        installed_wheel_env.python_exe,
        [
            "-c",
            (
                "import json; "
                "from importlib import metadata, resources; "
                "import invarlock.evidence_pack_baselines as baselines; "
                "import invarlock.evidence_pack_binding as binding; "
                "import invarlock.evidence_pack_report_verification as report_verification; "
                "import invarlock.reporting.verify_bootstrap as verify_bootstrap; "
                "root = resources.files('invarlock'); "
                "required = ["
                "'adapters/hf_mixin_loading.py', "
                "'adapters/hf_mixin_snapshot_manifest.py', "
                "'core/runner_runtime/execution_phases.py', "
                "'guards/invariant_checks.py', "
                "'guards/policy_validation.py', "
                "'_data/contracts/evidence_pack_manifest.schema.json', "
                "'_data/contracts/runtime_manifest.schema.json', "
                "'_data/contracts/verify_output.schema.json', "
                "'_data/runtime/profiles/ci.yaml', "
                "'_data/runtime/profiles/release.yaml', "
                "'_data/runtime/tiers.yaml'"
                "]; "
                "print(json.dumps({"
                "'imports': [baselines.__name__, binding.__name__, "
                "report_verification.__name__, verify_bootstrap.__name__], "
                "'missing': [path for path in required "
                "if not root.joinpath(*path.split('/')).is_file()], "
                "'requires': metadata.requires('invarlock') or []"
                "}, sort_keys=True))"
            ),
        ],
        cwd=tmp_path,
    )
    assert verifier_imports.returncode == 0, (
        verifier_imports.stdout + verifier_imports.stderr
    )
    verifier_payload = json.loads(verifier_imports.stdout.strip())
    assert verifier_payload["imports"] == [
        "invarlock.evidence_pack_baselines",
        "invarlock.evidence_pack_binding",
        "invarlock.evidence_pack_report_verification",
        "invarlock.reporting.verify_bootstrap",
    ]
    assert verifier_payload["missing"] == []
    assert any(
        requirement.lower().startswith("numpy>=1.24")
        for requirement in verifier_payload["requires"]
    )

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
                "maintained_catalog = {"
                "lane['lane_id']: {"
                "'status': lane['evidence_status'], "
                "'label': lane['evidence_status_label'], "
                "'has_evidence_paths': 'evidence' in lane"
                "} "
                "for lane in public_contracts.load_support_matrix()['lanes'] "
                "if lane.get('support_tier') == 'maintained_catalog'"
                "}; "
                "print(json.dumps({"
                "'catalog': catalog, "
                "'maintained_catalog': maintained_catalog"
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
    maintained_catalog = exported_contracts["maintained_catalog"]
    assert len(maintained_catalog) == 39
    available_count = sum(
        evidence
        == {
            "status": "available",
            "label": "Available",
            "has_evidence_paths": True,
        }
        for evidence in maintained_catalog.values()
    )
    not_created_count = sum(
        evidence
        == {
            "status": "not_created",
            "label": "Evidence not yet created",
            "has_evidence_paths": False,
        }
        for evidence in maintained_catalog.values()
    )
    assert available_count + not_created_count == len(maintained_catalog)

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
                "index = public_contracts.load_public_evidence_index(); "
                "evidence_root = resources.files('invarlock').joinpath("
                "'_data', 'public_evidence'"
                "); "
                "print(json.dumps({"
                "'index': index, "
                "'full_tree_exists': any("
                "evidence_root.joinpath(name).is_dir() "
                "for name in ('catalog_evidence', 'published_basis')"
                ")"
                "}, sort_keys=True))"
            ),
        ],
        cwd=tmp_path,
    )
    assert installed_public_evidence.returncode == 0, (
        installed_public_evidence.stdout + installed_public_evidence.stderr
    )
    public_evidence_payload = json.loads(installed_public_evidence.stdout.strip())
    index = public_evidence_payload["index"]
    assert index["format_version"] == "public-evidence-index-v2"
    assert index["carrier_policy"]["installed_wheel"] == "compact_index_only"
    assert index["catalog_evidence_count"] == available_count
    assert len(index["entries"]) == available_count
    if index["entries"]:
        assert "status" not in index
        assert "status_label" not in index
    else:
        assert index["status"] == "not_created"
        assert index["status_label"] == "Evidence not yet created"
    assert public_evidence_payload["full_tree_exists"] is False


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

    strict_report_dir = tmp_path / "strict-report"
    strict_report_path = strict_report_dir / "evaluation.report.json"
    strict_baseline_path = strict_report_dir / "trusted-baseline.json"
    strict_policy_path = strict_report_dir / "trusted-policy-pack.json"
    _write_json(strict_report_path, _build_strict_report())
    _write_json(strict_baseline_path, _build_strict_baseline_report())
    _write_json(strict_policy_path, _build_strict_policy_pack())
    _write_runtime_manifest(strict_report_path)

    verify_strict_report = _run(
        installed_wheel_env.cli_exe,
        [
            "verify",
            "--assurance",
            "strict",
            "--profile",
            "ci",
            "--baseline",
            str(strict_baseline_path),
            "--policy-pack",
            str(strict_policy_path),
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--json",
            str(strict_report_path),
        ],
        cwd=tmp_path,
    )
    assert verify_strict_report.returncode == 0, (
        verify_strict_report.stdout + verify_strict_report.stderr
    )
    strict_verify_payload = json.loads(
        verify_strict_report.stdout.strip().splitlines()[-1]
    )
    assert strict_verify_payload["format_version"] == "verify-v1"
    assert strict_verify_payload["summary"]["ok"] is True

    strict_html_path = tmp_path / "strict-evaluation.html"
    render_strict_html = _run(
        installed_wheel_env.cli_exe,
        [
            "report",
            "html",
            "-i",
            str(strict_report_path),
            "-o",
            str(strict_html_path),
        ],
        cwd=tmp_path,
    )
    assert render_strict_html.returncode == 0, (
        render_strict_html.stdout + render_strict_html.stderr
    )
    assert strict_html_path.is_file()
    assert "<html" in strict_html_path.read_text(encoding="utf-8").lower()

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
