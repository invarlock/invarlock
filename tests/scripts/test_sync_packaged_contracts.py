from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _run_sync_script(
    script: Path,
    *,
    source_dir: Path,
    packaged_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(script),
        "--source-dir",
        str(source_dir),
        "--packaged-dir",
        str(packaged_dir),
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(args, capture_output=True, text=True, check=False)


def test_sync_packaged_contracts_check_passes_when_dirs_match(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "checks" / "sync_packaged_contracts.py"
    source_dir = tmp_path / "contracts"
    packaged_dir = tmp_path / "packaged"
    _write_json(source_dir / "alpha.json", {"alpha": 1})
    _write_json(packaged_dir / "alpha.json", {"alpha": 1})

    proc = _run_sync_script(
        script,
        source_dir=source_dir,
        packaged_dir=packaged_dir,
        extra_args=["--check"],
    )

    assert proc.returncode == 0, proc.stderr
    assert "in sync" in proc.stdout


def test_sync_packaged_contracts_check_reports_missing_extra_and_changed(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "checks" / "sync_packaged_contracts.py"
    source_dir = tmp_path / "contracts"
    packaged_dir = tmp_path / "packaged"
    _write_json(source_dir / "alpha.json", {"alpha": 1})
    _write_json(source_dir / "beta.json", {"beta": 2})
    _write_json(packaged_dir / "alpha.json", {"alpha": 999})
    _write_json(packaged_dir / "gamma.json", {"gamma": 3})

    proc = _run_sync_script(
        script,
        source_dir=source_dir,
        packaged_dir=packaged_dir,
        extra_args=["--check"],
    )

    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "missing packaged contracts: beta.json" in combined
    assert "extra packaged contracts: gamma.json" in combined
    assert "out-of-sync packaged contracts: alpha.json" in combined


def test_sync_packaged_contracts_write_syncs_and_removes_extras(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "checks" / "sync_packaged_contracts.py"
    source_dir = tmp_path / "contracts"
    packaged_dir = tmp_path / "packaged"
    _write_json(source_dir / "alpha.json", {"alpha": 1})
    _write_json(source_dir / "beta.json", {"beta": 2})
    _write_json(packaged_dir / "alpha.json", {"alpha": 999})
    _write_json(packaged_dir / "gamma.json", {"gamma": 3})

    proc = _run_sync_script(
        script,
        source_dir=source_dir,
        packaged_dir=packaged_dir,
        extra_args=["--write"],
    )

    assert proc.returncode == 0, proc.stderr
    assert "Synchronized packaged contracts" in proc.stdout
    assert json.loads((packaged_dir / "alpha.json").read_text(encoding="utf-8")) == {
        "alpha": 1
    }
    assert json.loads((packaged_dir / "beta.json").read_text(encoding="utf-8")) == {
        "beta": 2
    }
    assert not (packaged_dir / "gamma.json").exists()


def test_sync_packaged_contracts_check_rejects_missing_dirs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "checks" / "sync_packaged_contracts.py"

    proc = _run_sync_script(
        script,
        source_dir=tmp_path / "missing-source",
        packaged_dir=tmp_path / "missing-packaged",
        extra_args=["--check"],
    )

    assert proc.returncode == 1
    assert "contract directory not found" in proc.stderr


def test_sync_packaged_contracts_write_allows_missing_packaged_dir(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "checks" / "sync_packaged_contracts.py"
    source_dir = tmp_path / "contracts"
    packaged_dir = tmp_path / "packaged"
    _write_json(source_dir / "alpha.json", {"alpha": 1})

    proc = _run_sync_script(
        script,
        source_dir=source_dir,
        packaged_dir=packaged_dir,
        extra_args=["--write"],
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads((packaged_dir / "alpha.json").read_text(encoding="utf-8")) == {
        "alpha": 1
    }
