#!/usr/bin/env python3
"""Assemble the standalone public contract bundle for a tagged release."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import shutil
import tempfile
import tarfile
from pathlib import Path
from typing import Any

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACTS_DIR = REPO_ROOT / "contracts"
DEFAULT_RUNTIME_DIR = REPO_ROOT / "src" / "invarlock" / "_data" / "runtime"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_kind(payload: object) -> str:
    if isinstance(payload, dict):
        return "json-object"
    if isinstance(payload, list):
        return "json-array"
    return "json-scalar"


def _contract_name(filename: str) -> str:
    name = filename.removesuffix(".schema.json")
    if name != filename:
        return name
    return filename.removesuffix(".json")


def _contract_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference: dict[str, Any] = {"path": f"contracts/{path.name}"}
    if isinstance(payload, dict):
        for key in ("format_version", "format", "core_abi", "match_policy"):
            value = payload.get(key)
            if isinstance(value, str):
                reference[key] = value
    elif isinstance(payload, list):
        reference["kind"] = "array"
        reference["item_count"] = len(payload)
    else:
        reference["kind"] = type(payload).__name__
    return reference


def _read_contracts(contracts_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    if not contracts_dir.is_dir():
        raise SystemExit(f"ERROR: contracts directory not found: {contracts_dir}")

    contract_files = sorted(
        path for path in contracts_dir.glob("*.json") if path.is_file()
    )
    if not contract_files:
        raise SystemExit("ERROR: public contract bundle requires at least one contract")

    catalog = {
        _contract_name(path.name): _contract_reference(path) for path in contract_files
    }
    return catalog, contract_files


def _read_runtime_files(runtime_dir: Path) -> tuple[Path, list[Path]]:
    if not runtime_dir.is_dir():
        raise SystemExit(f"ERROR: runtime directory not found: {runtime_dir}")

    tiers_path = runtime_dir / "tiers.yaml"
    profile_dir = runtime_dir / "profiles"
    profile_files = sorted(
        path for path in profile_dir.glob("*.yaml") if path.is_file()
    ) if profile_dir.is_dir() else []

    if not tiers_path.is_file():
        raise SystemExit(f"ERROR: runtime tiers file not found: {tiers_path}")
    if not profile_files:
        raise SystemExit("ERROR: public contract bundle requires at least one runtime profile")

    return tiers_path, profile_files


def _inventory_record(path: Path, *, relpath: str, category: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": relpath,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "category": category,
    }
    if path.suffix == ".json":
        payload["kind"] = _json_kind(json.loads(path.read_text(encoding="utf-8")))
    elif path.suffix in {".yaml", ".yml"}:
        payload["kind"] = "yaml"
    else:
        payload["kind"] = "text"
    return payload


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_readme(version: str, tag: str, repo: str) -> str:
    return (
        "InvarLock Public Contract Bundle\n\n"
        f"Release: {tag} ({version})\n"
        f"Repository: {repo}\n\n"
        "Contents:\n"
        "- contracts/*.json\n"
        "- contract_catalog.json\n"
        "- runtime/tiers.yaml\n"
        "- runtime/profiles/*.yaml\n"
        "- public_contract_bundle_manifest.json\n\n"
        "Verify by checking the bundle tarball signature, then compare the file\n"
        "hashes in public_contract_bundle_manifest.json against the extracted files.\n"
    )


def _build_tarball(bundle_root: Path, tarball_path: Path) -> None:
    files = sorted(path for path in bundle_root.rglob("*") if path.is_file())
    with tarball_path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gz_file:
            with tarfile.open(fileobj=gz_file, mode="w") as archive:
                for path in files:
                    arcname = path.relative_to(bundle_root.parent).as_posix()
                    info = archive.gettarinfo(str(path), arcname=arcname)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the standalone public contract release bundle."
    )
    parser.add_argument("--version", required=True, help="Release version without v")
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.3.12")
    parser.add_argument("--repo", required=True, help="GitHub repository slug")
    parser.add_argument(
        "--commit",
        required=True,
        help="Resolved commit SHA for the release tag",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the bundle tarball",
    )
    parser.add_argument(
        "--contracts-dir",
        default=str(DEFAULT_CONTRACTS_DIR),
        help="Override the source contracts directory",
    )
    parser.add_argument(
        "--runtime-dir",
        default=str(DEFAULT_RUNTIME_DIR),
        help="Override the shipped runtime data directory",
    )
    parser.add_argument(
        "--bundle-name",
        default="",
        help="Override the bundle root/base name",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    version = args.version
    tag = args.tag
    repo = args.repo
    commit = args.commit
    output_dir = Path(args.output_dir)
    contracts_dir = Path(args.contracts_dir)
    runtime_dir = Path(args.runtime_dir)
    bundle_name = args.bundle_name or f"invarlock-{version}-public-contract-bundle"

    catalog, contract_files = _read_contracts(contracts_dir)
    tiers_path, profile_files = _read_runtime_files(runtime_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.tmp.", dir=str(output_dir)))
    try:
        bundle_root = staging_dir / bundle_name
        bundle_root.mkdir(parents=True, exist_ok=True)

        (bundle_root / "contracts").mkdir(parents=True, exist_ok=True)
        for contract_path in contract_files:
            shutil.copy2(contract_path, bundle_root / "contracts" / contract_path.name)

        _write_text(
            bundle_root / "contract_catalog.json",
            json.dumps(catalog, sort_keys=True, indent=2) + "\n",
        )
        _write_text(
            bundle_root / "runtime" / "tiers.yaml",
            tiers_path.read_text(encoding="utf-8"),
        )
        for profile_path in profile_files:
            _write_text(
                bundle_root / "runtime" / "profiles" / profile_path.name,
                profile_path.read_text(encoding="utf-8"),
            )
        _write_text(bundle_root / "README.txt", _build_readme(version, tag, repo))

        inventory: list[dict[str, Any]] = []
        inventory.append(
            _inventory_record(
                bundle_root / "README.txt",
                relpath="README.txt",
                category="readme",
            )
        )
        inventory.append(
            _inventory_record(
                bundle_root / "contract_catalog.json",
                relpath="contract_catalog.json",
                category="catalog",
            )
        )
        inventory.extend(
            _inventory_record(
                bundle_root / "contracts" / contract_path.name,
                relpath=f"contracts/{contract_path.name}",
                category="contract",
            )
            for contract_path in contract_files
        )
        inventory.append(
            _inventory_record(
                bundle_root / "runtime" / "tiers.yaml",
                relpath="runtime/tiers.yaml",
                category="runtime-tier",
            )
        )
        inventory.extend(
            _inventory_record(
                bundle_root / "runtime" / "profiles" / profile_path.name,
                relpath=f"runtime/profiles/{profile_path.name}",
                category="runtime-profile",
            )
            for profile_path in profile_files
        )
        inventory.sort(key=lambda item: str(item["path"]))

        manifest = {
            "schema": "invarlock/public-contract-bundle-v1",
            "bundle": {
                "name": bundle_name,
                "version": version,
                "tag": tag,
                "repo": repo,
                "commit": commit,
                "generated_at": dt.datetime.now(UTC)
                .isoformat()
                .replace("+00:00", "Z"),
            },
            "contract_catalog": next(
                item for item in inventory if item["path"] == "contract_catalog.json"
            ),
            "inventory": inventory,
            "counts": {
                "contracts": len(contract_files),
                "runtime_profiles": len(profile_files),
                "files": len(inventory),
            },
        }
        _write_text(
            bundle_root / "public_contract_bundle_manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        )

        tarball_path = output_dir / f"{bundle_name}.tar.gz"
        _build_tarball(bundle_root, tarball_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Public contract bundle written to {tarball_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
