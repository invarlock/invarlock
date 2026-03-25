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
    assert manifest["verification"]["certificate_identity"] == (
        "repo:invarlock/invarlock@refs/tags/v0.3.12"
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
