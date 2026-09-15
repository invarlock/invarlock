from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_bundle(
    script: Path,
    dist_dir: Path,
    sbom_path: Path,
    provenance_dir: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(script),
            "--version",
            "0.3.12",
            "--tag",
            "v0.3.12",
            "--repo",
            "invarlock/invarlock",
            "--certificate-identity",
            "https://github.com/invarlock/invarlock/.github/workflows/sign.yml@refs/tags/v0.3.12",
            "--dist-dir",
            str(dist_dir),
            "--sbom",
            str(sbom_path),
            "--provenance-dir",
            str(provenance_dir),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_make_offline_bundle_packages_release_materials(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"
    assert script.exists(), "offline bundle generator script missing"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "sbom.json"

    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(
        dist_dir / "invarlock-0.3.12-py3-none-any.whl.sigstore.json",
        '{"bundle":"wheel"}',
    )
    _write(dist_dir / "invarlock-0.3.12.tar.gz", "sdist-bytes")
    _write(
        dist_dir / "invarlock-0.3.12.tar.gz.sigstore.json",
        '{"bundle":"sdist"}',
    )
    _write(dist_dir / "invarlock-0.3.12.tar.gz.crt", "crt")

    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = _run_bundle(script, dist_dir, sbom_path, provenance_dir, output_dir)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    tarball = output_dir / "invarlock-0.3.12-offline-bundle.tar.gz"
    assert tarball.exists()

    with tarfile.open(tarball, "r:gz") as archive:
        names = sorted(archive.getnames())
        root = "invarlock-0.3.12-offline-bundle"
        assert f"{root}/README.txt" in names
        assert f"{root}/public_key_hints.txt" in names
        assert f"{root}/release_manifest.json" in names
        assert f"{root}/invarlock-0.3.12-sbom.cdx.json" in names
        assert f"{root}/provenance/bundle.jsonl" in names
        assert f"{root}/dist/invarlock-0.3.12-py3-none-any.whl" in names
        assert f"{root}/dist/invarlock-0.3.12-py3-none-any.whl.sigstore.json" in names

        manifest = json.loads(
            archive.extractfile(f"{root}/release_manifest.json").read().decode("utf-8")
        )

    assert manifest["schema"] == "invarlock/release-offline-bundle-v1"
    assert manifest["bundle"]["tag"] == "v0.3.12"
    assert manifest["verification"]["certif" + "icate_identity"] == (
        "https://github.com/invarlock/invarlock/.github/workflows/sign.yml@refs/tags/v0.3.12"
    )
    assert manifest["sbom"]["path"] == "invarlock-0.3.12-sbom.cdx.json"

    distribution_paths = {item["path"] for item in manifest["distributions"]}
    assert "dist/invarlock-0.3.12-py3-none-any.whl" in distribution_paths
    assert "dist/invarlock-0.3.12.tar.gz" in distribution_paths

    wheel_record = next(
        item
        for item in manifest["distributions"]
        if item["path"] == "dist/invarlock-0.3.12-py3-none-any.whl"
    )
    assert wheel_record["sigstore_sidecars"] == [
        "dist/invarlock-0.3.12-py3-none-any.whl.sigstore.json"
    ]


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("with_ledger", [False, True])
def test_nested_addins_require_their_own_signed_inventory(
    tmp_path, missing, with_ledger
):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/release/make_offline_bundle.sh"
    )
    dist, provenance, output, sbom = (
        tmp_path / "dist",
        tmp_path / "provenance",
        tmp_path / "output",
        tmp_path / "sbom.json",
    )
    for name in ("invarlock-0.3.12.whl", "addins/invarlock_runtime_gguf-0.3.12.whl"):
        _write(dist / name, name)
        if not (missing and name.startswith("addins/")):
            _write(dist / (name + ".sigstore.json"), "{}")
    if with_ledger:
        _write(
            dist / "SHA256SUMS",
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dist).as_posix()}\n"
                for path in sorted(dist.rglob("*.whl"))
            ),
        )
    _write(provenance / "attestation.jsonl", "{}")
    _write(sbom, "{}")
    proc = _run_bundle(script, dist, sbom, provenance, output)
    if missing:
        assert proc.returncode != 0
        assert "missing Sigstore bundle" in proc.stderr
        assert not list(output.glob("*.tar.gz"))
        return
    assert proc.returncode == 0, proc.stderr
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        root = "invarlock-0.3.12-offline-bundle/"
        manifest = json.load(archive.extractfile(root + "release_manifest.json"))
        assert len(manifest["distributions"]) == 2
        records = [
            *manifest["distributions"],
            *manifest["distribution_signatures"],
            *manifest["provenance_bundles"],
            *manifest["supporting_files"],
            manifest["sbom"],
        ]
        assert {root + row["path"] for row in records} == {
            item.name
            for item in archive.getmembers()
            if item.isfile() and not item.name.endswith("/release_manifest.json")
        }
        for row in records:
            payload = archive.extractfile(root + row["path"]).read()
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
            assert len(payload) == row["size_bytes"]
        readme = archive.extractfile(root + "README.txt").read().decode()
        assert manifest["verification"]["certificate_identity"] in readme


@pytest.mark.parametrize(
    "alter", ["digest", "omission", "duplicate", "traversal", "encoding"]
)
def test_offline_bundle_rejects_invalid_distribution_ledger(tmp_path, alter):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/release/make_offline_bundle.sh"
    )
    dist, provenance, output, sbom = (
        tmp_path / "dist",
        tmp_path / "provenance",
        tmp_path / "output",
        tmp_path / "sbom.json",
    )
    name = "invarlock-0.3.12.whl"
    _write(dist / name, "wheel")
    _write(dist / (name + ".sigstore.json"), "{}")
    _write(provenance / "attestation.jsonl", "{}")
    _write(sbom, "{}")
    digest = hashlib.sha256(b"wheel").hexdigest()
    line = f"{digest}  {name}\n"
    contents = {
        "digest": f"{'0' * 64}  {name}\n",
        "omission": "",
        "duplicate": line * 2,
        "traversal": f"{digest}  ../{name}\n",
        "encoding": "",
    }[alter]
    _write(dist / "SHA256SUMS", contents)
    if alter == "encoding":
        (dist / "SHA256SUMS").write_bytes(b"\xff")
    proc = _run_bundle(script, dist, sbom, provenance, output)
    assert proc.returncode != 0
    assert "checksum ledger" in proc.stderr
    assert not list(output.glob("*.tar.gz"))


@pytest.mark.parametrize("alter", ["cross_directory_sidecar", "extra_file", "symlink"])
def test_offline_inventory_rejects_unbound_or_linked_files(tmp_path, alter):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/release/make_offline_bundle.sh"
    )
    dist, provenance, output, sbom = (
        tmp_path / "dist",
        tmp_path / "provenance",
        tmp_path / "output",
        tmp_path / "sbom.json",
    )
    _write(dist / "invarlock-0.3.12.whl", "wheel")
    _write(dist / "invarlock-0.3.12.whl.sigstore.json", "{}")
    _write(provenance / "attestation.jsonl", "{}")
    _write(sbom, "{}")
    if alter == "cross_directory_sidecar":
        _write(dist / "addins/invarlock-0.3.12.whl", "other wheel")
    elif alter == "extra_file":
        _write(dist / "unlisted.txt", "unexpected")
    else:
        (dist / "linked.whl").symlink_to(dist / "invarlock-0.3.12.whl")
        _write(dist / "linked.whl.sigstore.json", "{}")
    proc = _run_bundle(script, dist, sbom, provenance, output)
    assert proc.returncode != 0
    assert not list(output.glob("*.tar.gz"))


def test_make_offline_bundle_dry_run_writes_nothing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"
    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "sbom.json"
    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--version",
            "0.3.12",
            "--tag",
            "v0.3.12",
            "--repo",
            "invarlock/invarlock",
            "--certificate-identity",
            "https://github.com/invarlock/invarlock/.github/workflows/sign.yml@refs/tags/v0.3.12",
            "--dist-dir",
            str(dist_dir),
            "--sbom",
            str(sbom_path),
            "--provenance-dir",
            str(provenance_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "DRY RUN: would write" in proc.stdout
    assert not output_dir.exists()


def test_make_offline_bundle_rejects_path_like_bundle_name(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"
    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "sbom.json"
    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--version",
            "0.3.12",
            "--tag",
            "v0.3.12",
            "--repo",
            "invarlock/invarlock",
            "--certificate-identity",
            "https://github.com/invarlock/invarlock/.github/workflows/sign.yml@refs/tags/v0.3.12",
            "--dist-dir",
            str(dist_dir),
            "--sbom",
            str(sbom_path),
            "--provenance-dir",
            str(provenance_dir),
            "--output-dir",
            str(output_dir),
            "--bundle-name",
            "../escape",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "bundle name may only contain" in proc.stderr


def test_make_offline_bundle_requires_sigstore_bundle_per_artifact(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "sbom.json"

    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = _run_bundle(script, dist_dir, sbom_path, provenance_dir, output_dir)

    assert proc.returncode != 0
    assert "missing Sigstore bundle" in (proc.stderr or proc.stdout)


def test_make_offline_bundle_requires_provenance_files(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "sbom.json"

    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(
        dist_dir / "invarlock-0.3.12-py3-none-any.whl.sigstore.json",
        '{"bundle":"wheel"}',
    )
    provenance_dir.mkdir(parents=True, exist_ok=True)
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = _run_bundle(script, dist_dir, sbom_path, provenance_dir, output_dir)

    assert proc.returncode != 0
    assert "requires at least one provenance file" in (proc.stderr or proc.stdout)


def test_make_offline_bundle_requires_sbom(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "missing-sbom.json"

    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(
        dist_dir / "invarlock-0.3.12-py3-none-any.whl.sigstore.json",
        '{"bundle":"wheel"}',
    )
    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')

    proc = _run_bundle(script, dist_dir, sbom_path, provenance_dir, output_dir)

    assert proc.returncode != 0
    assert "SBOM file not found" in (proc.stderr or proc.stdout)


def test_make_offline_bundle_requires_real_distribution_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    output_dir = tmp_path / "out"
    sbom_path = tmp_path / "sbom.json"

    _write(
        dist_dir / "invarlock-0.3.12-py3-none-any.whl.sigstore.json",
        '{"bundle":"wheel"}',
    )
    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = _run_bundle(script, dist_dir, sbom_path, provenance_dir, output_dir)

    assert proc.returncode != 0
    assert "requires at least one wheel or sdist artifact" in (
        proc.stderr or proc.stdout
    )


def test_make_offline_bundle_supports_relative_output_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "release" / "make_offline_bundle.sh"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    sbom_path = tmp_path / "sbom.json"
    relative_output_dir = Path("release-assets")

    _write(dist_dir / "invarlock-0.3.12-py3-none-any.whl", "wheel-bytes")
    _write(
        dist_dir / "invarlock-0.3.12-py3-none-any.whl.sigstore.json",
        '{"bundle":"wheel"}',
    )
    _write(dist_dir / "invarlock-0.3.12.tar.gz", "sdist-bytes")
    _write(
        dist_dir / "invarlock-0.3.12.tar.gz.sigstore.json",
        '{"bundle":"sdist"}',
    )
    _write(provenance_dir / "bundle.jsonl", '{"provenance":"ok"}')
    _write(sbom_path, '{"bomFormat":"CycloneDX","specVersion":"1.4"}')

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--version",
            "0.3.12",
            "--tag",
            "v0.3.12",
            "--repo",
            "invarlock/invarlock",
            "--certificate-identity",
            "https://github.com/invarlock/invarlock/.github/workflows/sign.yml@refs/tags/v0.3.12",
            "--dist-dir",
            str(dist_dir),
            "--sbom",
            str(sbom_path),
            "--provenance-dir",
            str(provenance_dir),
            "--output-dir",
            str(relative_output_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (
        tmp_path / relative_output_dir / "invarlock-0.3.12-offline-bundle.tar.gz"
    ).exists()
