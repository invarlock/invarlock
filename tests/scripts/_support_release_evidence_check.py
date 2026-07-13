from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from typing import Any


def release_checker_module(repo_root: Path) -> Any:
    module_path = repo_root / "scripts" / "release" / "evidence_contracts.py"
    script_dir = str(module_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "release_evidence_contracts_under_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_bundle_manifest(output_dir: Path, name: str, manifest: object) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = name.removesuffix(".tar.gz")
    manifest_path = output_dir / f"{root}.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with tarfile.open(output_dir / name, "w:gz") as tar:
        tar.add(manifest_path, arcname=f"{root}/release_manifest.json")
    manifest_path.unlink()
