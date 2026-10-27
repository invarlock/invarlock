"""Retain resolved crate identities, license notices and a CycloneDX inventory."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vendored_provenance(package, directory):
    """Verify retained modified source separately from its upstream archive."""
    provenance_path = directory / "source-provenance.json"
    provenance = json.loads(provenance_path.read_bytes())
    for field in ("name", "version", "license"):
        if provenance[field] != package[field]:
            raise RuntimeError(f"Vendored {field} differs from Cargo metadata")

    def retained(relative):
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise RuntimeError(f"Unsafe vendored provenance path: {relative}")
        return path

    upstream_path = retained(provenance["upstream_file_hashes"])
    patched_path = retained(provenance["patched_file_hashes"])
    upstream = json.loads(upstream_path.read_bytes())
    patched = json.loads(patched_path.read_bytes())
    if upstream.keys() != patched.keys():
        raise RuntimeError("Vendored retained source inventories differ")
    changed = sorted(name for name in upstream if upstream[name] != patched[name])
    if changed != sorted(provenance["modified_upstream_files"]):
        raise RuntimeError("Vendored modified file declaration differs")
    for name, expected in patched.items():
        if sha256(retained(name)) != expected:
            raise RuntimeError(f"Vendored source changed: {name}")
    tree_sha256 = hashlib.sha256(
        json.dumps(patched, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if tree_sha256 != provenance["patched_source_tree_sha256"]:
        raise RuntimeError("Vendored source tree identity differs")
    patch_path = retained(provenance["patch"])
    if sha256(patch_path) != provenance["patch_sha256"]:
        raise RuntimeError("Vendored patch identity differs")
    return {
        **provenance,
        "provenance_path": str(provenance_path.relative_to(ROOT)),
        "provenance_sha256": sha256(provenance_path),
        "upstream_file_hashes_sha256": sha256(upstream_path),
        "patched_file_hashes_sha256": sha256(patched_path),
    }


def main() -> None:
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--manifest-path",
                str(ROOT / "Cargo.toml"),
                "--locked",
                "--offline",
                "--format-version",
                "1",
            ]
        )
    )
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    checksums = {(p["name"], p["version"]): p.get("checksum") for p in lock["package"]}
    components = []
    inventory = []
    refs = {}
    workspace_members = set(metadata["workspace_members"])
    retained_directories = set()
    for package in metadata["packages"]:
        ref = f"pkg:cargo/{package['name']}@{package['version']}"
        refs[package["id"]] = ref
        directory = Path(package["manifest_path"]).parent
        provenance = None
        if not package.get("source"):
            if package["id"] in workspace_members:
                continue
            if not directory.resolve().is_relative_to((ROOT / "vendor").resolve()):
                raise RuntimeError(f"Unclassified local dependency: {directory}")
            provenance = vendored_provenance(package, directory)
        destination = (
            ROOT / "licenses/dependencies" / f"{package['name']}-{package['version']}"
        )
        retained_directories.add(destination.name)
        destination.mkdir(parents=True, exist_ok=True)
        candidates = [
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "COPYRIGHT"))
        ]
        if package.get("license_file"):
            candidates.append(directory / package["license_file"])
        if package["name"] == "r-efi":
            candidates.append(directory / "AUTHORS")
        for source in sorted(set(candidates)):
            shutil.copyfile(source, destination / source.name)
        notices = sorted(p for p in destination.iterdir() if p.is_file())
        if not notices:
            raise RuntimeError(
                f"No retained license for {package['name']} {package['version']}"
            )
        checksum = (
            provenance["archive_sha256"]
            if provenance
            else checksums[(package["name"], package["version"])]
        )
        expression = (
            package["license"]
            .replace("MIT/Apache-2.0", "MIT OR Apache-2.0")
            .replace("Apache-2.0/MIT", "Apache-2.0 OR MIT")
        )
        component = {
            "type": "library",
            "bom-ref": ref,
            "name": package["name"],
            "version": package["version"],
            "purl": ref,
            "licenses": [{"expression": expression}],
        }
        if provenance:
            # An upstream archive checksum does not identify patched source.
            component["pedigree"] = {"notes": "Locally modified upstream dependency"}
            component["externalReferences"] = [
                {
                    "type": "distribution",
                    "url": provenance["source_url"],
                    "comment": "Original upstream registry archive; excludes local patch",
                    "hashes": [{"alg": "SHA-256", "content": checksum}],
                }
            ]
            component["properties"] = [
                {
                    "name": "source-classification",
                    "value": "vendored-third-party-modified",
                }
            ] + [
                {"name": key, "value": str(provenance[key])}
                for key in (
                    "provenance_path",
                    "provenance_sha256",
                    "patch_sha256",
                    "upstream_file_hashes_sha256",
                    "patched_file_hashes_sha256",
                    "patched_source_tree_sha256",
                    "patched_source_tree_algorithm",
                )
            ]
        elif checksum:
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        components.append(component)
        inventory.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package["license"],
                "source": (
                    "path+" + str(directory.relative_to(ROOT))
                    if provenance
                    else package["source"]
                ),
                "archive_sha256": checksum,
                **(
                    {
                        "archive_sha256_scope": "Original upstream registry archive, not modified source",
                        "source_classification": "vendored-third-party-modified",
                        "vendored_provenance": provenance,
                    }
                    if provenance
                    else {}
                ),
                "notices": [
                    {
                        "path": str(p.relative_to(ROOT)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in notices
                ],
            }
        )
    native = {p["id"]: p for p in metadata["packages"] if p["id"] in workspace_members}
    components.extend(
        {
            "type": "library",
            "bom-ref": refs[k],
            "purl": refs[k],
            "name": p["name"],
            "version": p["version"],
            "licenses": [{"expression": p["license"]}],
        }
        for k, p in native.items()
    )
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "invarlock-native-experiment",
                "version": "0.1.0",
            },
            "properties": [
                {
                    "name": "inventory-scope",
                    "value": "Resolved Cargo lock, including target-specific dependencies; not all are linked on every platform.",
                }
            ],
        },
        "components": sorted(components, key=lambda c: c["bom-ref"]),
        "dependencies": [
            {
                "ref": refs[n["id"]],
                "dependsOn": sorted(refs[d] for d in n["dependencies"]),
            }
            for n in sorted(metadata["resolve"]["nodes"], key=lambda n: refs[n["id"]])
        ],
    }
    for name, value in (
        (
            "dependency-inventory.json",
            sorted(inventory, key=lambda p: (p["name"], p["version"])),
        ),
        ("sbom.cdx.json", sbom),
    ):
        (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(f"Retained {len(inventory)} resolved crate license records.")
    orphans = sorted(
        p.name
        for p in (ROOT / "licenses/dependencies").iterdir()
        if p.is_dir() and p.name not in retained_directories
    )
    if orphans:
        print(
            "Retained license directories outside the current inventory: "
            + ", ".join(orphans)
        )


if __name__ == "__main__":
    main()
