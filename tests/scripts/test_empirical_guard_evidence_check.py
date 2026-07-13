from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.scripts._support_evidence_contracts import load_evidence_contracts_module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _checker_module():
    return load_evidence_contracts_module()


def _write_valid_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    for artifact in (
        "calibration/null_sweep_report.json",
        "calibration/ve_sweep_report.json",
        "catalog-evidence/summary.json",
        "families/gpt2.json",
    ):
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"ok": true}\n', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/empirical-guard-inventory-v1",
                "authority": "diagnostic_inventory",
                "source_commands": [
                    "invarlock evaluate --config resolved-config.yaml",
                    "invarlock advanced calibrate null-sweep --config configs/calibration/null_sweep_ci.yaml",
                    "invarlock advanced calibrate ve-sweep --config configs/calibration/rmt_ve_sweep_ci.yaml",
                ],
                "guard_rows": [
                    {
                        "guard": "spectral",
                        "evidence_kind": "calibration_null_sweep",
                        "status": "indexed",
                        "model_family": "gpt2",
                        "artifact": "calibration/null_sweep_report.json",
                    },
                    {
                        "guard": "rmt",
                        "evidence_kind": "catalog_evaluation",
                        "status": "indexed",
                        "model_family": "gpt2",
                        "artifact": "catalog-evidence/summary.json",
                    },
                    {
                        "guard": "variance",
                        "evidence_kind": "calibration_ve_sweep",
                        "status": "indexed",
                        "model_family": "gpt2",
                        "artifact": "calibration/ve_sweep_report.json",
                    },
                ],
                "model_family_rows": [
                    {
                        "model_family": "gpt2",
                        "status": "indexed",
                        "artifact": "families/gpt2.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _checker_command(root: Path, *, json_output: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(_repo_root() / "scripts" / "release" / "evidence_contracts.py"),
        "empirical-inventory",
        "--root",
        str(root),
    ]
    if json_output:
        command.append("--json")
    return command


def test_empirical_guard_evidence_check_accepts_valid_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    _write_valid_bundle(root)
    module = _checker_module()

    assert module.check_empirical_guard_evidence(root=root) == []
    assert module.main(["empirical-inventory", "--root", str(root)]) == 0
    assert module.main(["empirical-inventory", "--root", str(root), "--json"]) == 0

    proc = subprocess.run(
        _checker_command(root, json_output=True),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert '"ok": true' in proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["authoritative"] is False
    assert payload["authority"] == "diagnostic_inventory"
    assert payload["scope"] == "artifact_inventory_only"


def test_empirical_guard_evidence_success_never_claims_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    _write_valid_bundle(root)

    proc = subprocess.run(
        _checker_command(root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "structurally valid" in proc.stdout
    assert "diagnostic only" in proc.stdout
    assert "cannot authorize release or calibration claims" in proc.stdout
    assert "passed" not in proc.stdout.lower()


def test_removed_empirical_command_has_no_compatibility_alias(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "scripts" / "release" / "evidence_contracts.py"),
            "empirical",
            "--root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "invalid choice: 'empirical'" in proc.stderr


def test_empirical_guard_evidence_rejects_old_self_declared_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    _write_valid_bundle(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("authority")
    for row in manifest["guard_rows"]:
        row["status"] = "empirical"
    manifest["model_family_rows"][0]["status"] = "observed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failures = _checker_module().check_empirical_guard_evidence(root=root)

    assert any("authority must be diagnostic_inventory" in item for item in failures)
    assert any("status must be indexed" in item for item in failures)


def test_empirical_guard_evidence_check_reports_missing_manifest(
    tmp_path: Path,
) -> None:
    module = _checker_module()

    failures = module.check_empirical_guard_evidence(root=tmp_path / "missing")

    assert any("manifest missing" in failure for failure in failures)
    assert any("must be a JSON object" in failure for failure in failures)


def test_empirical_guard_evidence_check_rejects_synthetic_and_missing_guards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    (root / "artifact.json").write_text('{"ok": true}\n', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/empirical-guard-inventory-v1",
                "authority": "diagnostic_inventory",
                "source_commands": ["make guard-validation-smoke"],
                "guard_rows": [
                    {
                        "guard": "spectral",
                        "evidence_kind": "catalog_evaluation",
                        "status": "indexed",
                        "synthetic": True,
                        "scope": "synthetic smoke",
                        "model_family": "gpt2",
                        "artifact": "artifact.json",
                    }
                ],
                "model_family_rows": [],
            }
        ),
        encoding="utf-8",
    )
    module = _checker_module()

    failures = module.check_empirical_guard_evidence(root=root)

    assert any(
        "source_commands must include a real evidence producer" in item
        for item in failures
    )
    assert any("must not be synthetic evidence" in item for item in failures)
    assert any(
        "missing guard rows: rmt, spectral, variance" in item for item in failures
    )
    assert any(
        "model_family_rows must be a non-empty list" in item for item in failures
    )


def test_empirical_guard_evidence_check_rejects_artifact_path_edges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    empty = root / "empty.json"
    empty.write_text("", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text('{"ok": true}', encoding="utf-8")
    rows = [
        {
            "guard": "spectral",
            "evidence_kind": "calibration_null_sweep",
            "status": "indexed",
            "model_family": "gpt2",
            "artifact": str(outside),
        },
        {
            "guard": "rmt",
            "evidence_kind": "catalog_evaluation",
            "status": "indexed",
            "model_family": "gpt2",
            "artifact": "../outside.json",
        },
        {
            "guard": "variance",
            "evidence_kind": "calibration_ve_sweep",
            "status": "indexed",
            "model_family": "gpt2",
            "artifact": "empty.json",
        },
    ]
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/empirical-guard-inventory-v1",
                "authority": "diagnostic_inventory",
                "source_commands": ["invarlock evaluate --config resolved-config.yaml"],
                "guard_rows": rows,
                "model_family_rows": [
                    {
                        "model_family": "gpt2",
                        "status": "indexed",
                        "artifact": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    module = _checker_module()

    failures = module.check_empirical_guard_evidence(root=root)

    assert any("artifact must be relative" in failure for failure in failures)
    assert any("artifact escapes evidence root" in failure for failure in failures)
    assert any("artifact must not be empty" in failure for failure in failures)
    assert any(
        "artifact must be a non-empty relative path" in failure for failure in failures
    )


def test_empirical_guard_inventory_rejects_symlink_artifact(tmp_path: Path) -> None:
    root = tmp_path / "empirical"
    _write_valid_bundle(root)
    target = root / "calibration" / "null_sweep_report.json"
    target.unlink()
    target.symlink_to(root / "calibration" / "ve_sweep_report.json")

    failures = _checker_module().check_empirical_guard_evidence(root=root)

    assert any(
        "artifact must be a regular file, not a symlink" in item for item in failures
    )


def test_empirical_guard_evidence_check_rejects_shape_edges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    artifact = root / "artifact.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "unexpected",
                "source_commands": [None],
                "guard_rows": [
                    None,
                    {
                        "guard": "unknown",
                        "evidence_kind": "synthetic_smoke",
                        "status": "fallback",
                        "artifact": "missing.json",
                    },
                ],
                "model_family_rows": [
                    None,
                    {
                        "model_family": "",
                        "status": "synthetic",
                        "artifact": "artifact.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    module = _checker_module()

    failures = module.check_empirical_guard_evidence(root=root)

    assert any(
        "schema must be invarlock/empirical-guard-inventory-v1" in item
        for item in failures
    )
    assert any("source_commands[0] must be a string" in item for item in failures)
    assert any("guard_rows[0] must be an object" in item for item in failures)
    assert any("guard_rows[1].guard must be one of" in item for item in failures)
    assert any("missing guard rows" in item for item in failures)
    assert any("model_family_rows[0] must be an object" in item for item in failures)
    assert any("model_family_rows[1].model_family must be" in item for item in failures)
    assert any(
        "model_family_rows[1].status must be indexed" in item for item in failures
    )


def test_empirical_guard_evidence_check_rejects_required_field_edges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    family = root / "family.json"
    family.write_text('{"ok": true}', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/empirical-guard-inventory-v1",
                "authority": "diagnostic_inventory",
                "source_commands": [],
                "guard_rows": [
                    {
                        "guard": "spectral",
                        "evidence_kind": "synthetic_smoke",
                        "status": "fallback",
                        "model_family": "",
                        "artifact": "missing.json",
                    }
                ],
                "model_family_rows": [
                    {
                        "model_family": "gpt2",
                        "status": "indexed",
                        "artifact": "family.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    module = _checker_module()

    failures = module.check_empirical_guard_evidence(root=root)

    assert any("source_commands must be a non-empty list" in item for item in failures)
    assert any("evidence_kind must be one of" in item for item in failures)
    assert any("status must be indexed" in item for item in failures)
    assert any("model_family must be a non-empty string" in item for item in failures)
    assert any("artifact missing" in item for item in failures)

    failures.clear()
    module._validate_guard_rows(root, {"guard_rows": "bad"}, failures)
    assert any("guard_rows must be a non-empty list" in item for item in failures)

    failures.clear()
    module._validate_source_commands(
        {"source_commands": ["invarlock evaluate --config resolved-config.yaml"]},
        failures,
    )
    assert failures == []

    failures.clear()
    module._validate_model_family_rows(root, {"model_family_rows": "bad"}, failures)
    assert any(
        "model_family_rows must be a non-empty list" in item for item in failures
    )


def test_empirical_guard_evidence_contract_owner_paths(tmp_path: Path) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    artifact = root / "artifact.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    module = _checker_module()
    failures: list[str] = []

    assert module._load_json(artifact, "artifact", failures) == {"ok": True}
    assert module._resolve_artifact(root, "artifact.json", "row", failures) == artifact

    module._validate_guard_rows(
        root,
        {
            "guard_rows": [
                None,
                {
                    "guard": "spectral",
                    "evidence_kind": "calibration_null_sweep",
                    "status": "indexed",
                    "model_family": "gpt2",
                    "artifact": "artifact.json",
                },
            ]
        },
        failures,
    )
    assert any("guard_rows[0] must be an object" in item for item in failures)
    assert any("missing guard rows: rmt, variance" in item for item in failures)

    failures.clear()
    module._validate_guard_rows(
        root,
        {
            "guard_rows": [
                {
                    "guard": "spectral",
                    "evidence_kind": "calibration_null_sweep",
                    "status": "indexed",
                    "model_family": "gpt2",
                    "artifact": "artifact.json",
                },
                {
                    "guard": "rmt",
                    "evidence_kind": "catalog_evaluation",
                    "status": "indexed",
                    "model_family": "gpt2",
                    "artifact": "artifact.json",
                },
                {
                    "guard": "variance",
                    "evidence_kind": "calibration_ve_sweep",
                    "status": "indexed",
                    "model_family": "gpt2",
                    "artifact": "artifact.json",
                },
            ]
        },
        failures,
    )
    assert failures == []

    failures.clear()
    module._validate_model_family_rows(
        root,
        {
            "model_family_rows": [
                None,
                {
                    "model_family": "gpt2",
                    "status": "indexed",
                    "artifact": "artifact.json",
                },
            ]
        },
        failures,
    )
    assert failures == ["model_family_rows[0] must be an object."]

    EmpiricalGuardEvidenceManifest = module.EmpiricalGuardEvidenceManifest

    payload_none = EmpiricalGuardEvidenceManifest(root=root, payload=None)
    assert any("must be a JSON object" in item for item in payload_none.validate())
    none_failures: list[str] = []
    payload_none._validate_source_commands(none_failures)
    payload_none._validate_guard_rows(none_failures)
    payload_none._validate_model_family_rows(none_failures)
    assert none_failures == []


def test_empirical_guard_evidence_wrapper_counts_invalid_dict_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    artifact = root / "artifact.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    module = _checker_module()
    failures: list[str] = []

    module._validate_guard_rows(
        root,
        {
            "guard_rows": [
                {
                    "guard": "spectral",
                    "evidence_kind": "calibration_null_sweep",
                    "status": "fallback",
                    "model_family": "gpt2",
                    "artifact": "artifact.json",
                }
            ]
        },
        failures,
    )

    assert any("guard_rows[0].status must be indexed" in item for item in failures)
    assert any("missing guard rows" in item for item in failures)


def test_empirical_guard_evidence_contract_empty_guard_rows(tmp_path: Path) -> None:
    module = _checker_module()
    EmpiricalGuardEvidenceManifest = module.EmpiricalGuardEvidenceManifest

    manifest = EmpiricalGuardEvidenceManifest(
        root=tmp_path,
        payload={
            "schema": "invarlock/empirical-guard-inventory-v1",
            "authority": "diagnostic_inventory",
            "source_commands": ["invarlock evaluate"],
            "guard_rows": [],
            "model_family_rows": [{"model_family": "gpt2", "status": "indexed"}],
        },
    )

    failures = manifest.validate()

    assert "empirical evidence guard_rows must be a non-empty list." in failures


def test_offline_bundle_manifest_handles_unreadable_member(monkeypatch, tmp_path: Path):
    module = _checker_module()
    OfflineBundleManifest = module.OfflineBundleManifest

    class FakeTar:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def getmembers(self):
            return [
                SimpleNamespace(
                    name="release_manifest.json",
                    isfile=lambda: True,
                )
            ]

        def extractfile(self, _member):
            return None

    monkeypatch.setattr(module.tarfile, "open", lambda *_a, **_k: FakeTar())
    failures: list[str] = []

    manifest = OfflineBundleManifest.load_from_tarball(
        tmp_path / "offline.tar.gz",
        failures,
    )

    assert manifest.payload is None
    assert any("manifest unreadable" in failure for failure in failures)


def test_empirical_guard_evidence_contract_non_object_manifest(
    tmp_path: Path,
) -> None:
    module = _checker_module()
    EmpiricalGuardEvidenceManifest = module.EmpiricalGuardEvidenceManifest

    root = tmp_path / "empirical"
    root.mkdir()
    (root / "manifest.json").write_text("[]", encoding="utf-8")
    failures: list[str] = []

    manifest = EmpiricalGuardEvidenceManifest.load(root=root, failures=failures)

    assert manifest.payload is None
    assert failures == ["empirical guard evidence manifest must be a JSON object."]

    module = _checker_module()
    check_failures = module.check_empirical_guard_evidence(root=root)
    assert check_failures == [
        "empirical guard evidence manifest must be a JSON object."
    ]


def test_empirical_guard_evidence_check_rejects_malformed_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{", encoding="utf-8")
    module = _checker_module()

    assert module.main(["empirical-inventory", "--root", str(root)]) == 1
    proc = subprocess.run(
        _checker_command(root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "not valid JSON" in proc.stderr


def test_empirical_guard_inventory_rejects_duplicate_manifest_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empirical"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"schema":"invarlock/empirical-guard-inventory-v1",'
        '"authority":"diagnostic_inventory",'
        '"authority":"release",'
        '"source_commands":[],"guard_rows":[],"model_family_rows":[]}',
        encoding="utf-8",
    )

    failures = _checker_module().check_empirical_guard_evidence(root=root)

    assert any("ambiguous JSON" in item for item in failures)
    assert any("duplicate JSON key 'authority'" in item for item in failures)
