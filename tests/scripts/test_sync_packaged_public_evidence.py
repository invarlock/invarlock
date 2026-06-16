from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "checks" / "sync_packaged_public_evidence.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    source_root = root / "public_evidence"
    support_matrix = root / "contracts" / "support_matrix.json"
    evidence_dir = source_root / "published_basis" / "demo"
    evidence_dir.mkdir(parents=True)
    (source_root / "README.md").write_text("# Public evidence\n", encoding="utf-8")
    (evidence_dir / "evaluation.report.json").write_text("report\n", encoding="utf-8")
    (evidence_dir / "runtime.manifest.json").write_text("manifest\n", encoding="utf-8")
    (evidence_dir / "evidence_pack").mkdir()
    (evidence_dir / "evidence_pack" / "manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    _write_json(
        evidence_dir / "evidence.meta.json",
        {
            "evidence_class": "strict_pass_fixture",
            "summary": "fixture report",
            "artifact_paths": {
                "evaluation_report": "evaluation.report.json",
                "runtime_manifest": "runtime.manifest.json",
                "evidence_pack": "evidence_pack",
            },
        },
    )
    _write_json(
        support_matrix,
        {
            "lanes": [
                {
                    "lane_id": "demo-lane",
                    "support_tier": "published_basis",
                    "evidence": {
                        "evaluation_report_fixture": (
                            "public_evidence/published_basis/demo/"
                            "evaluation.report.json"
                        )
                    },
                }
            ]
        },
    )
    return source_root, support_matrix


def _run(
    *,
    source_root: Path,
    support_matrix: Path,
    packaged_root: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--support-matrix",
            str(support_matrix),
            "--packaged-root",
            str(packaged_root),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_sync_packaged_public_evidence_write_and_check(tmp_path: Path) -> None:
    source_root, support_matrix = _write_fixture(tmp_path)
    packaged_root = tmp_path / "packaged"

    written = _run(
        source_root=source_root,
        support_matrix=support_matrix,
        packaged_root=packaged_root,
        args=["--write"],
    )

    assert written.returncode == 0, written.stderr
    index = json.loads(
        (packaged_root / "published_basis_index.json").read_text(encoding="utf-8")
    )
    assert index["format_version"] == "public-evidence-index-v1"
    assert index["carrier_policy"]["installed_wheel"] == "compact_index_only"
    assert index["entries"][0]["lanes"] == ["demo-lane"]
    assert index["entries"][0]["artifacts"]["evaluation_report"]["sha256"].startswith(
        "sha256:"
    )

    checked = _run(
        source_root=source_root,
        support_matrix=support_matrix,
        packaged_root=packaged_root,
        args=["--check"],
    )

    assert checked.returncode == 0, checked.stderr
    assert "in sync" in checked.stdout


def test_sync_packaged_public_evidence_rejects_drift_and_legacy_tree(
    tmp_path: Path,
) -> None:
    source_root, support_matrix = _write_fixture(tmp_path)
    packaged_root = tmp_path / "packaged"
    packaged_root.mkdir()
    _write_json(packaged_root / "published_basis_index.json", {"stale": True})
    (packaged_root / "published_basis" / "demo").mkdir(parents=True)

    checked = _run(
        source_root=source_root,
        support_matrix=support_matrix,
        packaged_root=packaged_root,
        args=["--check"],
    )

    assert checked.returncode == 1
    combined = checked.stdout + checked.stderr
    assert "legacy packaged public evidence tree must be removed" in combined
    assert "out-of-sync packaged public evidence index" in combined


def test_sync_packaged_public_evidence_write_removes_legacy_tree(
    tmp_path: Path,
) -> None:
    source_root, support_matrix = _write_fixture(tmp_path)
    packaged_root = tmp_path / "packaged"
    (packaged_root / "published_basis" / "demo").mkdir(parents=True)

    written = _run(
        source_root=source_root,
        support_matrix=support_matrix,
        packaged_root=packaged_root,
        args=["--write"],
    )

    assert written.returncode == 0, written.stderr
    assert "removed_legacy_tree=True" in written.stdout
    assert not (packaged_root / "published_basis").exists()
