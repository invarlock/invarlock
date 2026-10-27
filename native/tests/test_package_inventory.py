"""Keep modified third-party source identities separate from registry archives."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "package_inventory", Path(__file__).parents[1] / "tools/package_inventory.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    directory = tmp_path / "vendor/jsonschema"
    (directory / "src").mkdir(parents=True)
    originals = {
        "Cargo.toml": "package",
        "LICENSE": "MIT notice",
        "src/lib.rs": "before",
    }
    for name, text in originals.items():
        (directory / name).write_text(text)
    upstream = {name: inventory.sha256(directory / name) for name in originals}
    (directory / "src/lib.rs").write_text("after")
    patched = {name: inventory.sha256(directory / name) for name in originals}
    (directory / "fix.patch").write_text("retained patch")
    dump(directory / "upstream.json", upstream)
    dump(directory / "patched.json", patched)
    provenance = {
        "name": "jsonschema",
        "version": "0.39.0",
        "license": "MIT",
        "origin": "external upstream dependency; not InvarLock implementation",
        "source_url": "https://static.crates.io/crates/jsonschema/jsonschema-0.39.0.crate",
        "archive_sha256": "a" * 64,
        "modified_upstream_files": ["src/lib.rs"],
        "omitted_archive_files": ["Cargo.lock"],
        "patch": "fix.patch",
        "patch_sha256": inventory.sha256(directory / "fix.patch"),
        "upstream_file_hashes": "upstream.json",
        "patched_file_hashes": "patched.json",
        "patched_source_tree_sha256": hashlib.sha256(
            json.dumps(patched, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "patched_source_tree_algorithm": "SHA-256 of compact sorted file-hash JSON",
    }
    dump(directory / "source-provenance.json", provenance)
    package = {
        "id": "vendored-jsonschema",
        "name": "jsonschema",
        "version": "0.39.0",
        "source": None,
        "license": "MIT",
        "manifest_path": str(directory / "Cargo.toml"),
    }
    workspace = {
        "id": "workspace-core",
        "name": "invarlock-native-core",
        "version": "0.1.0",
        "source": None,
        "license": "Apache-2.0 AND PSF-2.0",
        "manifest_path": str(tmp_path / "core/Cargo.toml"),
    }
    metadata = {
        "packages": [package, workspace],
        "workspace_members": [workspace["id"]],
        "resolve": {
            "nodes": [
                {"id": workspace["id"], "dependencies": [package["id"]]},
                {"id": package["id"], "dependencies": []},
            ]
        },
    }
    (tmp_path / "Cargo.lock").write_text(
        '[[package]]\nname = "jsonschema"\nversion = "0.39.0"\n'
    )
    monkeypatch.setattr(
        inventory.subprocess,
        "check_output",
        lambda command: json.dumps(metadata).encode(),
    )
    return directory, package, provenance, metadata


def test_vendored_dependency_has_notice_and_distinct_patched_identity(
    vendor, tmp_path, capsys
):
    directory, package, provenance, metadata = vendor
    orphan = tmp_path / "licenses/dependencies/jsonschema-0.33.0"
    orphan.mkdir(parents=True)
    (orphan / "LICENSE").write_text("old retained notice")
    inventory.main()
    records = json.loads((tmp_path / "dependency-inventory.json").read_bytes())
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "jsonschema"
    assert record["source_classification"] == "vendored-third-party-modified"
    assert record["archive_sha256"] == provenance["archive_sha256"]
    assert "not modified source" in record["archive_sha256_scope"]
    assert (
        record["vendored_provenance"]["patched_source_tree_sha256"]
        == provenance["patched_source_tree_sha256"]
    )
    assert record["notices"] == [
        {
            "path": "licenses/dependencies/jsonschema-0.39.0/LICENSE",
            "sha256": inventory.sha256(directory / "LICENSE"),
        }
    ]
    sbom = json.loads((tmp_path / "sbom.cdx.json").read_bytes())
    (component,) = [c for c in sbom["components"] if c["name"] == "jsonschema"]
    assert "hashes" not in component
    assert (
        component["externalReferences"][0]["hashes"][0]["content"]
        == provenance["archive_sha256"]
    )
    properties = {p["name"]: p["value"] for p in component["properties"]}
    assert properties["patch_sha256"] == provenance["patch_sha256"]
    assert (
        properties["patched_source_tree_sha256"]
        == provenance["patched_source_tree_sha256"]
    )
    assert len(sbom["components"]) == 2
    assert "jsonschema-0.33.0" in capsys.readouterr().out
    assert (orphan / "LICENSE").read_text() == "old retained notice"
    first = [
        (tmp_path / name).read_bytes()
        for name in ("dependency-inventory.json", "sbom.cdx.json")
    ]
    metadata["packages"].reverse()
    metadata["resolve"]["nodes"].reverse()
    inventory.main()
    assert first == [
        (tmp_path / name).read_bytes()
        for name in ("dependency-inventory.json", "sbom.cdx.json")
    ]


@pytest.mark.parametrize("name", ["src/lib.rs", "LICENSE", "fix.patch"])
def test_changed_vendored_files_fail_before_inventory_generation(
    vendor, tmp_path, name
):
    directory, package, provenance, metadata = vendor
    (directory / name).write_text("changed")
    with pytest.raises(
        RuntimeError, match="Vendored (source changed|patch identity differs)"
    ):
        inventory.main()
    assert not (tmp_path / "dependency-inventory.json").exists()


@pytest.mark.parametrize(
    "field", ["version", "patched_source_tree_sha256", "modified_upstream_files"]
)
def test_inconsistent_provenance_fails(vendor, field):
    directory, package, provenance, metadata = vendor
    provenance[field] = [] if field == "modified_upstream_files" else "wrong"
    dump(directory / "source-provenance.json", provenance)
    with pytest.raises(RuntimeError, match="Vendored"):
        inventory.vendored_provenance(package, directory)


def test_unclassified_local_dependency_is_not_counted_as_our_implementation(
    vendor, tmp_path
):
    directory, package, provenance, metadata = vendor
    package["manifest_path"] = str(tmp_path / "unknown/Cargo.toml")
    with pytest.raises(RuntimeError, match="Unclassified local dependency"):
        inventory.main()
