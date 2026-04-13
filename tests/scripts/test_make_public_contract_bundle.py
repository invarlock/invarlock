from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_bundle(
    script: Path,
    contracts_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python",
            str(script),
            "--version",
            "0.3.12",
            "--tag",
            "v0.3.12",
            "--repo",
            "invarlock/invarlock",
            "--commit",
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "--contracts-dir",
            str(contracts_dir),
            "--runtime-dir",
            str(runtime_dir),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_make_public_contract_bundle_packages_contracts_and_runtime_data(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_public_contract_bundle.py"
    assert script.exists(), "public contract bundle generator script missing"

    contracts_dir = tmp_path / "contracts"
    runtime_dir = tmp_path / "runtime"
    output_dir = tmp_path / "out"

    _write(
        contracts_dir / "support_matrix.json",
        json.dumps({"format_version": "support-matrix-v1", "lanes": []}),
    )
    _write(
        contracts_dir / "metric_kinds.json",
        json.dumps(["ppl", "xent"]),
    )
    _write(
        contracts_dir / "policy_pack.schema.json",
        json.dumps({"title": "Policy Pack", "type": "object"}),
    )
    _write(runtime_dir / "tiers.yaml", "balanced: {}\n")
    _write(runtime_dir / "profiles" / "ci.yaml", "profile: ci\n")
    _write(runtime_dir / "profiles" / "release.yaml", "profile: release\n")

    proc = _run_bundle(script, contracts_dir, runtime_dir, output_dir)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    tarball = output_dir / "invarlock-0.3.12-public-contract-bundle.tar.gz"
    assert tarball.exists()

    with tarfile.open(tarball, "r:gz") as archive:
        names = sorted(archive.getnames())
        root = "invarlock-0.3.12-public-contract-bundle"
        assert f"{root}/README.txt" in names
        assert f"{root}/contract_catalog.json" in names
        assert f"{root}/public_contract_bundle_manifest.json" in names
        assert f"{root}/contracts/support_matrix.json" in names
        assert f"{root}/contracts/metric_kinds.json" in names
        assert f"{root}/contracts/policy_pack.schema.json" in names
        assert f"{root}/runtime/tiers.yaml" in names
        assert f"{root}/runtime/profiles/ci.yaml" in names
        assert f"{root}/runtime/profiles/release.yaml" in names

        manifest = json.loads(
            archive.extractfile(f"{root}/public_contract_bundle_manifest.json")
            .read()
            .decode("utf-8")
        )
        catalog = json.loads(
            archive.extractfile(f"{root}/contract_catalog.json").read().decode("utf-8")
        )

    assert manifest["schema"] == "invarlock/public-contract-bundle-v1"
    assert manifest["bundle"]["version"] == "0.3.12"
    assert manifest["bundle"]["tag"] == "v0.3.12"
    assert manifest["bundle"]["repo"] == "invarlock/invarlock"
    assert manifest["bundle"]["commit"] == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert "generated_at" not in manifest["bundle"]
    assert manifest["contract_catalog"]["path"] == "contract_catalog.json"
    assert manifest["counts"] == {
        "contracts": 3,
        "runtime_profiles": 2,
        "files": 8,
    }

    inventory_paths = [item["path"] for item in manifest["inventory"]]
    assert inventory_paths == sorted(inventory_paths)
    assert inventory_paths == [
        "README.txt",
        "contract_catalog.json",
        "contracts/metric_kinds.json",
        "contracts/policy_pack.schema.json",
        "contracts/support_matrix.json",
        "runtime/profiles/ci.yaml",
        "runtime/profiles/release.yaml",
        "runtime/tiers.yaml",
    ]

    support_matrix = catalog["support_matrix"]
    assert support_matrix["path"] == "contracts/support_matrix.json"
    assert support_matrix["format_version"] == "support-matrix-v1"
    assert catalog["metric_kinds"]["kind"] == "array"
    assert catalog["metric_kinds"]["item_count"] == 2
    assert catalog["policy_pack"]["path"] == "contracts/policy_pack.schema.json"


def test_make_public_contract_bundle_requires_runtime_profile(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_public_contract_bundle.py"

    contracts_dir = tmp_path / "contracts"
    runtime_dir = tmp_path / "runtime"
    output_dir = tmp_path / "out"

    _write(contracts_dir / "support_matrix.json", json.dumps({"lanes": []}))
    _write(runtime_dir / "tiers.yaml", "balanced: {}\n")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    proc = _run_bundle(script, contracts_dir, runtime_dir, output_dir)

    assert proc.returncode != 0
    assert "requires at least one runtime profile" in (proc.stderr or proc.stdout)


def test_make_public_contract_bundle_requires_contracts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_public_contract_bundle.py"

    contracts_dir = tmp_path / "contracts"
    runtime_dir = tmp_path / "runtime"
    output_dir = tmp_path / "out"

    _write(runtime_dir / "tiers.yaml", "balanced: {}\n")
    _write(runtime_dir / "profiles" / "ci.yaml", "profile: ci\n")
    contracts_dir.mkdir(parents=True, exist_ok=True)

    proc = _run_bundle(script, contracts_dir, runtime_dir, output_dir)

    assert proc.returncode != 0
    assert "requires at least one contract" in (proc.stderr or proc.stdout)


def test_make_public_contract_bundle_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_public_contract_bundle.py"

    contracts_dir = tmp_path / "contracts"
    runtime_dir = tmp_path / "runtime"
    output_one = tmp_path / "out-one"
    output_two = tmp_path / "out-two"

    _write(
        contracts_dir / "support_matrix.json",
        json.dumps({"format_version": "support-matrix-v1", "lanes": []}),
    )
    _write(
        contracts_dir / "metric_kinds.json",
        json.dumps(["ppl", "xent"]),
    )
    _write(runtime_dir / "tiers.yaml", "balanced: {}\n")
    _write(runtime_dir / "profiles" / "ci.yaml", "profile: ci\n")

    first = _run_bundle(script, contracts_dir, runtime_dir, output_one)
    second = _run_bundle(script, contracts_dir, runtime_dir, output_two)

    assert first.returncode == 0, first.stderr or first.stdout
    assert second.returncode == 0, second.stderr or second.stdout

    tarball_one = output_one / "invarlock-0.3.12-public-contract-bundle.tar.gz"
    tarball_two = output_two / "invarlock-0.3.12-public-contract-bundle.tar.gz"
    assert tarball_one.read_bytes() == tarball_two.read_bytes()
