"""Retain resolved crate identities, license notices and a CycloneDX inventory."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    for package in metadata["packages"]:
        ref = f"pkg:cargo/{package['name']}@{package['version']}"
        refs[package["id"]] = ref
        if not package.get("source"):
            continue
        directory = Path(package["manifest_path"]).parent
        destination = (
            ROOT / "licenses/dependencies" / f"{package['name']}-{package['version']}"
        )
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
        checksum = checksums[(package["name"], package["version"])]
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
        if checksum:
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        components.append(component)
        inventory.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package["license"],
                "source": package["source"],
                "archive_sha256": checksum,
                "notices": [
                    {
                        "path": str(p.relative_to(ROOT)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in notices
                ],
            }
        )
    native = {p["id"]: p for p in metadata["packages"] if not p.get("source")}
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
            for n in metadata["resolve"]["nodes"]
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


if __name__ == "__main__":
    main()
